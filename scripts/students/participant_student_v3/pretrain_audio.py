from __future__ import annotations
import argparse,json,random,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from .audio_model import CompactAudioBranch,parameter_count
from scripts.teachers.ussd_audio import AUTHOR_COMMIT,AUTHOR_TRAIN_CROP_FRAMES,SEGMENT_FRAMES
from scripts.teachers.ussd_audio.common import load_author_stats,load_feature_manifest,sha256

RUN4_SEED=1300
AUTHOR_BALANCED_SEGMENTS_PER_CLASS=468
FRAMES=SEGMENT_FRAMES

def resolve_path(row,split,local):
    if local:
        q=Path(local)/split/f"{int(row.participant_id)}.npy"
        if q.exists(): return q
    return Path(row.feature_path)

class AuthorCropSegments(Dataset):
    """Fixed author-style TRAIN crop, segmented before normalization.

    The crop pointer sequence matches run-4 style random.Random(seed) over the
    sorted participant manifest. Class balancing is fixed once from TRAIN.
    """
    def __init__(self,meta,mean,std,local=None,seed=RUN4_SEED):
        self.meta=meta.sort_values("participant_id").reset_index(drop=True)
        self.mean=mean.astype(np.float32); self.std=std.astype(np.float32)
        self.local=local; self.seed=int(seed); rng=random.Random(self.seed)
        self.arr=[]; self.crop_start=[]; self.items=[]
        for i,r in enumerate(self.meta.itertuples(index=False)):
            p=resolve_path(r,"train",local); x=np.load(p,mmap_mode="r")
            assert x.ndim==2 and x.shape[0]==130,(r.participant_id,x.shape)
            if x.shape[1] < AUTHOR_TRAIN_CROP_FRAMES:
                raise AssertionError(f"{r.participant_id}: {x.shape[1]} < {AUTHOR_TRAIN_CROP_FRAMES}")
            start=rng.randint(0,x.shape[1]-AUTHOR_TRAIN_CROP_FRAMES)
            self.arr.append(x); self.crop_start.append(start)
            n=math.ceil(AUTHOR_TRAIN_CROP_FRAMES/FRAMES)
            for s in range(n): self.items.append((i,s,int(r.label)))
        neg=[j for j,x in enumerate(self.items) if x[2]==0]
        pos=[j for j,x in enumerate(self.items) if x[2]==1]
        n=AUTHOR_BALANCED_SEGMENTS_PER_CLASS
        if len(neg)<n or len(pos)<n:
            raise AssertionError(f"Need at least {n} segments/class; got neg={len(neg)} pos={len(pos)}")
        br=random.Random(self.seed)
        self.keep=br.sample(neg,n)+br.sample(pos,n)
        self.n_per_class=n
    def __len__(self): return len(self.keep)
    def __getitem__(self,k):
        i,s,y=self.items[self.keep[k]]; x=self.arr[i]; c0=self.crop_start[i]
        a=c0+s*FRAMES; remain=AUTHOR_TRAIN_CROP_FRAMES-s*FRAMES
        take=max(0,min(FRAMES,remain)); b=a+take
        # Author order: segment/crop -> zero pad -> normalize entire segment.
        raw=np.zeros((130,FRAMES),np.float32)
        if take: raw[:,:take]=np.asarray(x[:,a:b],np.float32)
        z=(raw-self.mean)/self.std
        return torch.from_numpy(z.astype(np.float32)),torch.tensor(float(y))

def participant_segments(path,mean,std):
    x=np.load(path,mmap_mode="r"); n=math.ceil(x.shape[1]/FRAMES)
    for s in range(n):
        a=s*FRAMES; b=min((s+1)*FRAMES,x.shape[1])
        raw=np.zeros((130,FRAMES),np.float32)
        raw[:,:b-a]=np.asarray(x[:,a:b],np.float32)
        yield ((raw-mean)/std).astype(np.float32)

def participant_infer(model,row,split,local,mean,std,device,batch=128):
    path=resolve_path(row,split,local); logits=[]; embs=[]; buf=[]
    with torch.inference_mode():
        for seg in participant_segments(path,mean,std):
            buf.append(seg)
            if len(buf)==batch:
                z,e=model(torch.from_numpy(np.stack(buf)).to(device)); logits.append(z.cpu().numpy()); embs.append(e.cpu().numpy()); buf=[]
        if buf:
            z,e=model(torch.from_numpy(np.stack(buf)).to(device)); logits.append(z.cpu().numpy()); embs.append(e.cpu().numpy())
    l=np.concatenate(logits); e=np.concatenate(embs); p=1/(1+np.exp(-l))
    vf=float((p>=.5).mean())
    return {
        "participant_id":int(row.participant_id),"label":int(row.label),"n_segments":int(len(p)),
        "mean_probability":float(p.mean()),"mean_prediction":int(p.mean()>=.5),
        "vote_fraction":vf,"vote_prediction":int(np.rint(vf)),
        "embedding":np.concatenate([e.mean(0),e.std(0)]).astype(np.float32)
    }

def evaluate(model,meta,split,local,mean,std,device):
    model.eval(); rows=[]; emb=[]
    for r in meta.itertuples(index=False):
        z=participant_infer(model,r,split,local,mean,std,device); emb.append(z.pop("embedding")); rows.append(z)
    d=pd.DataFrame(rows); y=d.label.to_numpy(int); q=d.mean_probability.to_numpy(float)
    def pack(pred):
        return {"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
                "depressed_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
                "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    mm=pack(d.mean_prediction.to_numpy(int)); mv=pack(d.vote_prediction.to_numpy(int))
    m={"n":int(len(d)),"mean_probability":mm,"majority_vote":mv,"auroc_soft_mean_probability":float(roc_auc_score(y,q))}
    return d,np.stack(emb),m

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",required=True,help="USSD-compatible ComParE16 cache root")
    ap.add_argument("--author-root",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--local-audio-root",default=None); ap.add_argument("--seed",type=int,default=RUN4_SEED)
    ap.add_argument("--batch-size",type=int,default=20); ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--patience",type=int,default=20); ap.add_argument("--lr",type=float,default=3e-3)
    ap.add_argument("--lr-factor-epochs",type=int,default=2); ap.add_argument("--lr-decay",type=float,default=.9)
    a=ap.parse_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

    feat,out=Path(a.features),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    trmeta=load_feature_manifest(feat,"train"); dvmeta=load_feature_manifest(feat,"dev")
    assert len(trmeta)==107 and len(dvmeta)==34 and 440 not in set(dvmeta.participant_id.astype(int))
    assert not set(trmeta.participant_id.astype(int)) & set(dvmeta.participant_id.astype(int))

    mean,std,stat_path=load_author_stats(Path(a.author_root)); mean=mean.astype(np.float32); std=std.astype(np.float32)
    ds=AuthorCropSegments(trmeta,mean,std,a.local_audio_root,a.seed)
    g=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=g,num_workers=0,pin_memory=torch.cuda.is_available())

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=CompactAudioBranch().to(device)
    params=parameter_count(model); assert params==182529,params
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=0.0)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()

    # Save/evaluate the random initialization explicitly as epoch 0.
    # This is an audit artifact only and is NOT eligible for DEV checkpoint selection.
    epoch0_ckpt=out/"epoch0.pt"
    epoch0_dev,_,epoch0_metrics=evaluate(model,dvmeta,"dev",a.local_audio_root,mean,std,device)
    torch.save({"model_state_dict":model.state_dict(),"epoch":0,
                "selection_eligible":False,"seed":a.seed},epoch0_ckpt)
    epoch0_dev.to_csv(out/"epoch0_dev_predictions.csv",index=False)
    (out/"epoch0_metrics.json").write_text(json.dumps({
        "epoch":0,"selection_eligible":False,"dev34":epoch0_metrics,
        "seed":a.seed,"batch_size":a.batch_size,"test_opened":False
    },indent=2)+"\n")

    print("AUDIO v3 author-recipe compressed pretraining")
    print("Device:",device,"| GPU:",torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")
    print("Model device:",next(model.parameters()).device,"| params:",params)
    print("TRAIN:",len(trmeta),"DEV:",len(dvmeta),"| TEST CLOSED")
    print("Crop frames:",AUTHOR_TRAIN_CROP_FRAMES,"| crop seed:",a.seed,"| segment frames:",FRAMES)
    print("Balanced crop segments/class:",ds.n_per_class,"| total/epoch:",len(ds))
    print("Author TRAIN normalization:",stat_path)
    print("Optimizer: Adam | lr:",a.lr,"| weight_decay: 0 | batch:",a.batch_size,
          "| lr_decay:",a.lr_decay,"every",a.lr_factor_epochs,"epochs")

    best=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"
    for epoch in range(1,a.epochs+1):
        model.train(); total=n=0
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            z,_=model(x); loss=loss_fn(z,y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach().cpu())*len(y); n+=len(y)

        dd,_,m=evaluate(model,dvmeta,"dev",a.local_audio_root,mean,std,device)
        # Participant-level selection, fixed threshold. Majority vote is
        # author-compatible; soft AUROC breaks ties only.
        pm=m["majority_vote"]; key=(pm["macro_f1"],pm["depressed_f1"],m["auroc_soft_mean_probability"])
        row={"epoch":epoch,"loss":total/n,"lr":opt.param_groups[0]["lr"],
             "dev_vote_macro_f1":pm["macro_f1"],"dev_vote_depressed_f1":pm["depressed_f1"],
             "dev_mean_macro_f1":m["mean_probability"]["macro_f1"],
             "dev_auroc":m["auroc_soft_mean_probability"]}
        hist.append(row); pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={total/n:.4f} voteF1={pm['macro_f1']:.4f} depF1={pm['depressed_f1']:.4f} "
              f"meanF1={m['mean_probability']['macro_f1']:.4f} AUROC={m['auroc_soft_mean_probability']:.4f} lr={opt.param_groups[0]['lr']:.6g}")

        if key>best:
            best=key; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"selection_key":key},ckpt)
            dd.to_csv(out/"best_dev_predictions.csv",index=False)
        else:
            stale+=1

        if epoch%a.lr_factor_epochs==0:
            for pg in opt.param_groups: pg["lr"]*=a.lr_decay
        if stale>=a.patience:
            print("Early stop; best epoch",best_epoch); break

    state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"]); model.eval()
    trd,tre,tm=evaluate(model,trmeta,"train",a.local_audio_root,mean,std,device)
    dvd,dve,dm=evaluate(model,dvmeta,"dev",a.local_audio_root,mean,std,device)
    trd.to_csv(out/"train_full_predictions.csv",index=False); dvd.to_csv(out/"dev_predictions.csv",index=False)
    np.savez_compressed(out/"train_audio_embeddings.npz",participant_ids=trd.participant_id.to_numpy(int),
                        labels=trd.label.to_numpy(int),embedding=tre,probability=trd.mean_probability.to_numpy(np.float32))
    np.savez_compressed(out/"dev_audio_embeddings.npz",participant_ids=dvd.participant_id.to_numpy(int),
                        labels=dvd.label.to_numpy(int),embedding=dve,probability=dvd.mean_probability.to_numpy(np.float32))

    protocol={
      "mode":"hard-label audio branch pretraining","architecture":"ComParE16 Conv128 + LSTM128x1",
      "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
      "author_commit":AUTHOR_COMMIT,"author_normalization_reused":True,"normalization_sha256":sha256(stat_path),
      "teacher_checkpoint_loaded":False,"crop_frames":AUTHOR_TRAIN_CROP_FRAMES,"crop_seed":a.seed,
      "segmentation_frames":FRAMES,"padding_then_normalization":True,
      "class_balance":"author run-4 SUB_SAMPLE_ND_CLASS: fixed 468 segments/class from deterministic TRAIN crop; no class weights",
      "optimizer":"Adam","initial_lr":a.lr,"weight_decay":0.0,"batch_size":a.batch_size,
      "lr_schedule":f"x{a.lr_decay} every {a.lr_factor_epochs} epochs",
      "checkpoint_selection":"DEV-34 participant majority-vote macro-F1, then depressed-F1, then soft-mean AUROC",
      "epoch0_saved":True,"epoch0_selection_eligible":False,
      "threshold_search":False,"best_epoch":best_epoch,"test_opened":False
    }
    peak=float(torch.cuda.max_memory_allocated()/1024**2) if device.type=="cuda" else 0.0
    result={"train_full":tm,"dev34":dm,"epoch0_dev34":epoch0_metrics,
            "protocol":protocol,"params":params,"peak_cuda_mb":peak}
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print("\nBEST AUDIO-BRANCH DEV-34:",json.dumps(dm,indent=2))
    print("best_epoch:",best_epoch,"| peak_cuda_mb:",round(peak,2),"| TEST CLOSED.")

if __name__=="__main__": main()
