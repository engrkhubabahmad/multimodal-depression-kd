from __future__ import annotations
import argparse,json,random,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from .audio_model import CompactAudioBranch,parameter_count
from scripts.teachers.ussd_audio.common import load_author_stats

FRAMES=384

class Segments(Dataset):
    def __init__(self,manifest,mean,std,split,local=None,k=16,seed=103):
        self.m=pd.read_csv(manifest).sort_values("participant_id").reset_index(drop=True)
        self.mean=mean; self.std=std; self.split=split; self.local=Path(local) if local else None; self.k=int(k); self.seed=int(seed); self.epoch=0
        self.arr=[]; self.nseg=[]
        for r in self.m.itertuples():
            p=(self.local/split/f"{int(r.participant_id)}.npy") if self.local else Path(r.feature_path)
            x=np.load(p,mmap_mode="r"); assert x.ndim==2 and x.shape[0]==130
            self.arr.append(x); self.nseg.append(int(math.ceil(x.shape[1]/FRAMES)))
        self.set_epoch(0)
    def set_epoch(self,e):
        self.epoch=int(e); items=[]
        for i,(r,n) in enumerate(zip(self.m.itertuples(),self.nseg)):
            rng=np.random.default_rng(self.seed+1000003*self.epoch+7919*int(r.participant_id))
            ids=rng.choice(n,size=self.k,replace=n<self.k)
            items.extend((i,int(j)) for j in ids)
        random.Random(self.seed+self.epoch).shuffle(items); self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,j):
        i,s=self.items[j]; r=self.m.iloc[i]; x=self.arr[i]; a=s*FRAMES; b=min((s+1)*FRAMES,x.shape[1])
        z=np.zeros((130,FRAMES),np.float32); z[:,:b-a]=(np.asarray(x[:,a:b],np.float32)-self.mean)/self.std
        return torch.from_numpy(z),torch.tensor(float(r.label)),torch.tensor(int(r.participant_id))

def all_participant(model,row,path,mean,std,device,batch=128):
    x=np.load(path,mmap_mode="r"); n=int(math.ceil(x.shape[1]/FRAMES)); logits=[]; embs=[]
    with torch.inference_mode():
        for start in range(0,n,batch):
            q=[]
            for s in range(start,min(n,start+batch)):
                a=s*FRAMES; b=min((s+1)*FRAMES,x.shape[1]); z=np.zeros((130,FRAMES),np.float32)
                z[:,:b-a]=(np.asarray(x[:,a:b],np.float32)-mean)/std; q.append(z)
            zz,e=model(torch.from_numpy(np.stack(q)).float().to(device)); logits.append(zz.cpu().numpy()); embs.append(e.cpu().numpy())
    l=np.concatenate(logits); e=np.concatenate(embs); p=1/(1+np.exp(-l))
    return {"participant_id":int(row.participant_id),"label":int(row.label),"probability":float(p.mean()),
            "vote_fraction":float((p>=.5).mean()),"embedding":np.concatenate([e.mean(0),e.std(0)]).astype(np.float32),"segments":n}

def evaluate(model,meta,split,local,mean,std,device):
    model.eval(); rows=[]; embeds=[]
    for r in meta.itertuples():
        path=(Path(local)/split/f"{int(r.participant_id)}.npy") if local else Path(r.feature_path)
        z=all_participant(model,r,path,mean,std,device); embeds.append(z.pop("embedding")); rows.append(z)
    d=pd.DataFrame(rows); y=d.label.to_numpy(int); p=d.probability.to_numpy(float); pred=(p>=.5).astype(int)
    m={"n":len(d),"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro")),
       "depressed_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),"auroc":float(roc_auc_score(y,p)),
       "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}
    return d,np.stack(embeds),m

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--features",required=True); ap.add_argument("--author-root",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--local-audio-root",default=None); ap.add_argument("--seed",type=int,default=103); ap.add_argument("--segments-per-participant",type=int,default=16)
    ap.add_argument("--batch-size",type=int,default=64); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--weight-decay",type=float,default=1e-4)
    a=ap.parse_args(argv); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    feat,out=Path(a.features),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    mean,std,stat_path=load_author_stats(Path(a.author_root)); mean=mean.astype(np.float32); std=std.astype(np.float32)
    trm=feat/"train_manifest.csv"; dvm=feat/"dev_manifest.csv"; trmeta=pd.read_csv(trm); dvmeta=pd.read_csv(dvm)
    assert len(trmeta)==107 and len(dvmeta)==34 and 440 not in set(dvmeta.participant_id.astype(int))
    ds=Segments(trm,mean,std,"train",a.local_audio_root,a.segments_per_participant,a.seed)
    g=torch.Generator().manual_seed(a.seed); loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=0,generator=g)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=CompactAudioBranch().to(device)
    params=parameter_count(model); assert params==182529,params
    y=trmeta.label.to_numpy(int); cnt=np.bincount(y,minlength=2); pos_weight=float(cnt[0]/cnt[1])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    best=(-1.,-1.,-1.); best_epoch=0; stale=0; ckpt=out/"best.pt"; hist=[]
    print("AUDIO v3 hard-label segment pretraining | params:",params,"| segments/participant/epoch:",a.segments_per_participant)
    print("Author TRAIN normalization:",stat_path)
    for epoch in range(1,a.epochs+1):
        ds.set_epoch(epoch); model.train(); total=n=0
        for x,yb,_ in loader:
            x,yb=x.to(device),yb.to(device); z,_=model(x); loss=loss_fn(z,yb)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach().cpu())*len(yb); n+=len(yb)
        dd,_,m=evaluate(model,dvmeta,"dev",a.local_audio_root,mean,std,device)
        key=(m["macro_f1"],m["depressed_f1"],m["auroc"]); row={"epoch":epoch,"loss":total/n,**m}; hist.append(row); pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={total/n:.4f} dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")
        if key>best:
            best=key; best_epoch=epoch; stale=0; torch.save({"model_state_dict":model.state_dict(),"epoch":epoch},ckpt); dd.to_csv(out/"best_dev_predictions.csv",index=False)
        else:
            stale+=1
            if stale>=a.patience: print("Early stop; best epoch",best_epoch); break
    state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"])
    trd,tre,tm=evaluate(model,trmeta,"train",a.local_audio_root,mean,std,device); dvd,dve,dm=evaluate(model,dvmeta,"dev",a.local_audio_root,mean,std,device)
    np.savez_compressed(out/"train_audio_embeddings.npz",participant_ids=trd.participant_id.to_numpy(int),labels=trd.label.to_numpy(int),embedding=tre,probability=trd.probability.to_numpy(np.float32))
    np.savez_compressed(out/"dev_audio_embeddings.npz",participant_ids=dvd.participant_id.to_numpy(int),labels=dvd.label.to_numpy(int),embedding=dve,probability=dvd.probability.to_numpy(np.float32))
    protocol={"mode":"hard-label audio branch pretraining","architecture":"ComParE16 Conv128 + LSTM128x1","train_participants":107,"dev_participants":34,
              "participant_440_excluded":True,"author_normalization_reused":True,"teacher_checkpoint_loaded":False,
              "segments_per_participant_per_epoch":a.segments_per_participant,"dev_aggregation":"mean segment probability over all DEV segments",
              "best_epoch":best_epoch,"test_opened":False}
    (out/"metrics.json").write_text(json.dumps({"train":tm,"dev34":dm,"protocol":protocol},indent=2)+"\n")
    print("\nBEST AUDIO-BRANCH DEV-34:",json.dumps(dm,indent=2)); print("best_epoch:",best_epoch,"| TEST CLOSED.")
if __name__=="__main__": main()
