from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from .audio_model import CompactAudioBranch,parameter_count
from scripts.teachers.ussd_audio import AUTHOR_COMMIT,SEGMENT_FRAMES
from scripts.teachers.ussd_audio.common import load_author_stats,load_feature_manifest,sha256

FRAMES=SEGMENT_FRAMES

def resolve_path(row,split,local):
    if local:
        p=Path(local)/split/f"{int(row.participant_id)}.npy"
        if p.exists(): return p
    return Path(row.feature_path)

def make_segment(x,s,mean,std):
    a=s*FRAMES; b=min((s+1)*FRAMES,x.shape[1])
    raw=np.zeros((130,FRAMES),np.float32)
    raw[:,:b-a]=np.asarray(x[:,a:b],np.float32)
    return ((raw-mean)/std).astype(np.float32)

class ParticipantBags(Dataset):
    """One TRAIN item per participant; sample K ComParE16 segments per epoch."""
    def __init__(self,meta,mean,std,split="train",local=None,k=16,seed=103):
        self.meta=meta.sort_values("participant_id").reset_index(drop=True)
        self.mean=mean.astype(np.float32); self.std=std.astype(np.float32)
        self.split=split; self.local=local; self.k=int(k); self.seed=int(seed); self.epoch=0
        self.arr=[]; self.nseg=[]
        for r in self.meta.itertuples(index=False):
            p=resolve_path(r,split,local); x=np.load(p,mmap_mode="r")
            assert x.ndim==2 and x.shape[0]==130,(r.participant_id,x.shape)
            self.arr.append(x); self.nseg.append(int(math.ceil(x.shape[1]/FRAMES)))
    def set_epoch(self,e): self.epoch=int(e)
    def __len__(self): return len(self.meta)
    def __getitem__(self,i):
        r=self.meta.iloc[i]; n=self.nseg[i]
        rng=np.random.default_rng(self.seed+1000003*self.epoch+7919*int(r.participant_id))
        ids=rng.choice(n,size=self.k,replace=n<self.k)
        bag=np.stack([make_segment(self.arr[i],int(s),self.mean,self.std) for s in ids])
        return torch.from_numpy(bag),torch.tensor(float(r.label)),torch.tensor(int(r.participant_id))

def infer_participant(model,row,split,local,mean,std,device,batch=128):
    x=np.load(resolve_path(row,split,local),mmap_mode="r"); n=int(math.ceil(x.shape[1]/FRAMES))
    probs=[]; embs=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,n,batch):
            q=np.stack([make_segment(x,s,mean,std) for s in range(start,min(n,start+batch))])
            z,e=model(torch.from_numpy(q).to(device))
            probs.append(torch.sigmoid(z).cpu().numpy()); embs.append(e.cpu().numpy())
    p=np.concatenate(probs); e=np.concatenate(embs)
    mp=float(p.mean()); vf=float((p>=.5).mean())
    return {"participant_id":int(row.participant_id),"label":int(row.label),"n_segments":int(n),
            "mean_probability":mp,"mean_prediction":int(mp>=.5),
            "vote_fraction":vf,"vote_prediction":int(np.rint(vf)),
            "embedding":np.concatenate([e.mean(0),e.std(0)]).astype(np.float32)}

def evaluate(model,meta,split,local,mean,std,device):
    rows=[]; embeds=[]
    for r in meta.itertuples(index=False):
        z=infer_participant(model,r,split,local,mean,std,device); embeds.append(z.pop("embedding")); rows.append(z)
    d=pd.DataFrame(rows); y=d.label.to_numpy(int); q=d.mean_probability.to_numpy(float)
    def pack(pred):
        return {"accuracy":float(accuracy_score(y,pred)),
                "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
                "depressed_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
                "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    return d,np.stack(embeds),{
        "n":int(len(d)),
        "mean_probability":pack(d.mean_prediction.to_numpy(int)),
        "majority_vote":pack(d.vote_prediction.to_numpy(int)),
        "auroc_soft_mean_probability":float(roc_auc_score(y,q))
    }

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",required=True); ap.add_argument("--author-root",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--local-audio-root",default=None); ap.add_argument("--seed",type=int,default=103)
    ap.add_argument("--segments-per-participant",type=int,default=16); ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--epochs",type=int,default=80); ap.add_argument("--patience",type=int,default=15)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--weight-decay",type=float,default=1e-4)
    a=ap.parse_args(argv)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

    feat,out=Path(a.features),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    trmeta=load_feature_manifest(feat,"train"); dvmeta=load_feature_manifest(feat,"dev")
    assert len(trmeta)==107 and len(dvmeta)==34 and 440 not in set(dvmeta.participant_id.astype(int))
    assert not set(trmeta.participant_id.astype(int)) & set(dvmeta.participant_id.astype(int))

    mean,std,stat_path=load_author_stats(Path(a.author_root)); mean=mean.astype(np.float32); std=std.astype(np.float32)
    ds=ParticipantBags(trmeta,mean,std,"train",a.local_audio_root,a.segments_per_participant,a.seed)
    g=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=g,num_workers=0,pin_memory=torch.cuda.is_available())

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=CompactAudioBranch().to(device)
    params=parameter_count(model); assert params==182529,params
    cnt=np.bincount(trmeta.label.to_numpy(int),minlength=2); pos_weight=float(cnt[0]/cnt[1])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()

    print("AUDIO v3 participant-bag hard-label pretraining")
    print("Device:",device,"| GPU:",torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU")
    print("Model device:",next(model.parameters()).device,"| params:",params)
    print("TRAIN:",len(trmeta),"DEV:",len(dvmeta),"| TEST CLOSED")
    print("Segments/participant/epoch:",a.segments_per_participant,"| participant batch:",a.batch_size)
    print("Participant class counts:",cnt.tolist(),"| pos_weight:",pos_weight)
    print("Author TRAIN normalization:",stat_path)
    print("Optimizer: AdamW | lr:",a.lr,"| weight_decay:",a.weight_decay)
    print("Objective: BCE on logit(mean(sigmoid(segment_logits))) at PARTICIPANT level")

    best=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"; eps=1e-6
    for epoch in range(1,a.epochs+1):
        ds.set_epoch(epoch); model.train(); total=n=0
        for bags,y,_ in loader:
            bags=bags.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            B,K,C,T=bags.shape
            z,_=model(bags.reshape(B*K,C,T)); p=torch.sigmoid(z).reshape(B,K).mean(1)
            bag_logit=torch.logit(p.clamp(eps,1-eps))
            loss=loss_fn(bag_logit,y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach().cpu())*B; n+=B

        dd,_,m=evaluate(model,dvmeta,"dev",a.local_audio_root,mean,std,device)
        mm=m["mean_probability"]; key=(mm["macro_f1"],mm["depressed_f1"],m["auroc_soft_mean_probability"])
        row={"epoch":epoch,"loss":total/n,
             "dev_mean_macro_f1":mm["macro_f1"],"dev_mean_depressed_f1":mm["depressed_f1"],
             "dev_vote_macro_f1":m["majority_vote"]["macro_f1"],
             "dev_auroc":m["auroc_soft_mean_probability"]}
        hist.append(row); pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={total/n:.4f} meanF1={mm['macro_f1']:.4f} depF1={mm['depressed_f1']:.4f} "
              f"voteF1={m['majority_vote']['macro_f1']:.4f} AUROC={m['auroc_soft_mean_probability']:.4f}")

        if key>best:
            best=key; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"selection_key":key},ckpt)
            dd.to_csv(out/"best_dev_predictions.csv",index=False)
        else:
            stale+=1
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

    protocol={"mode":"hard-label participant-bag audio pretraining",
              "architecture":"ComParE16 Conv128 + LSTM128x1",
              "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
              "author_commit":AUTHOR_COMMIT,"author_normalization_reused":True,
              "normalization_sha256":sha256(stat_path),"teacher_checkpoint_loaded":False,
              "segment_frames":FRAMES,"padding_then_normalization":True,
              "segments_per_participant_per_epoch":a.segments_per_participant,
              "objective":"participant BCE on logit(mean sigmoid segment logits)",
              "train_class_weight_source":"TRAIN-107 only","pos_weight":pos_weight,
              "optimizer":"AdamW","lr":a.lr,"weight_decay":a.weight_decay,
              "checkpoint_selection":"DEV-34 mean-probability macro-F1, then depressed-F1, then AUROC",
              "threshold_search":False,"best_epoch":best_epoch,"test_opened":False}
    peak=float(torch.cuda.max_memory_allocated()/1024**2) if device.type=="cuda" else 0.0
    result={"train_full":tm,"dev34":dm,"protocol":protocol,"params":params,"peak_cuda_mb":peak}
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print("\nBEST AUDIO-BAG DEV-34:",json.dumps(dm,indent=2))
    print("best_epoch:",best_epoch,"| peak_cuda_mb:",round(peak,2),"| TEST CLOSED.")

if __name__=="__main__": main()
