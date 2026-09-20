from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader
from scripts.students.participant_student.metrics import metric_dict
from .data import RichParticipantDataset,collate
from .model import AudioSegmentEncoder

class AudioOnly(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=AudioSegmentEncoder(d_model=96)
        self.score=nn.Linear(96,1)
        self.head=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Dropout(.35),nn.Linear(32,1))
    def forward(self,audio,mask):
        b,s,c,t=audio.shape; flat=audio.reshape(b*s,c,t); valid=mask.reshape(-1)
        # Encode only real segments. Padding never enters BatchNorm statistics.
        z=self.enc(flat[valid].unsqueeze(1)).squeeze(1)
        allz=torch.zeros((b*s,96),device=audio.device,dtype=z.dtype); allz[valid]=z; allz=allz.reshape(b,s,96)
        e=self.score(allz).squeeze(-1).masked_fill(~mask,-1e9); w=torch.softmax(e,dim=1)
        pooled=(allz*w.unsqueeze(-1)).sum(1)
        return self.head(pooled).squeeze(-1)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def predict(model,loader,device):
    model.eval(); rows=[]
    with torch.inference_mode():
        for b in loader:
            z=model(b["audio"].to(device),b["mask"].to(device)); q=torch.sigmoid(z).cpu().numpy()
            for pid,y,p in zip(b["participant_id"].numpy(),b["label"].numpy(),q):
                rows.append({"participant_id":int(pid),"label":int(y),"probability":float(p)})
    return pd.DataFrame(rows)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--local-audio-root",default=None); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=103); p.add_argument("--epochs",type=int,default=60); p.add_argument("--patience",type=int,default=15)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--weight-decay",type=float,default=5e-4)
    a=p.parse_args(argv); seed_all(a.seed); feat,out=Path(a.features),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((feat/"feature_summary.json").read_text()); assert not summary["test_opened"]
    trd=RichParticipantDataset(feat/"train_manifest.csv",feat/"train_text_tfidf.npy",feat/"audio_standardizer.npz","train",a.seed,local_audio_root=a.local_audio_root)
    dvd=RichParticipantDataset(feat/"dev_manifest.csv",feat/"dev_text_tfidf.npy",feat/"audio_standardizer.npz","dev",a.seed,local_audio_root=a.local_audio_root)
    g=torch.Generator().manual_seed(a.seed)
    tr=DataLoader(trd,batch_size=a.batch_size,shuffle=True,generator=g,collate_fn=collate,num_workers=0)
    dv=DataLoader(dvd,batch_size=max(1,a.batch_size//2),shuffle=False,collate_fn=collate,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=AudioOnly().to(device)
    params=sum(x.numel() for x in model.parameters() if x.requires_grad)
    y=trd.meta.label.to_numpy(int); cnt=np.bincount(y,minlength=2); pw=float(cnt[0]/cnt[1])
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=device)); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    best=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]
    print("AUDIO-ONLY diagnostic | params:",params,"| TEST CLOSED")
    for epoch in range(1,a.epochs+1):
        trd.set_epoch(epoch); model.train(); total=n=0
        for b in tr:
            yy=b["label"].to(device); z=model(b["audio"].to(device),b["mask"].to(device)); loss=loss_fn(z,yy)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach().cpu())*len(yy); n+=len(yy)
        d=predict(model,dv,device); m=metric_dict(d.label,d.probability,.5); key=(m["macro_f1"],m["depressed_f1"],m["auroc"])
        row={"epoch":epoch,"loss":total/n,**{f"dev_{k}":v for k,v in m.items() if k!="confusion_matrix"}}; hist.append(row)
        pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={total/n:.4f} dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")
        if key>best:
            best=key; best_epoch=epoch; stale=0; torch.save({"model_state_dict":model.state_dict(),"epoch":epoch},out/"best.pt")
            d.to_csv(out/"best_dev_predictions.csv",index=False); (out/"best_metrics.json").write_text(json.dumps(m,indent=2)+"\n")
        else:
            stale+=1
            if stale>=a.patience: break
    print("\nBEST AUDIO-ONLY DEV-34:",json.dumps(json.loads((out/"best_metrics.json").read_text()),indent=2))
    print("best_epoch:",best_epoch,"| TEST CLOSED")

if __name__=="__main__": main()
