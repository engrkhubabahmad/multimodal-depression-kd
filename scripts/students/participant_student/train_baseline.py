from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from .model import ParticipantReLiMPNet,parameter_count
from .metrics import metric_dict,save_bundle
from .complexity import profile_model,save_complexity

class Participants(Dataset):
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z: self.d={k:z[k] for k in z.files}
    def __len__(self): return len(self.d["labels"])
    def __getitem__(self,i):
        return {k:self.d[k][i] for k in ["participant_ids","labels","audio","text","segment_mask","n_segments"]}

def collate(batch):
    s=max(int(x["n_segments"]) for x in batch)
    return {"participant_id":torch.tensor([x["participant_ids"] for x in batch]).long(),
        "label":torch.tensor([x["labels"] for x in batch]).float(),
        "audio":torch.from_numpy(np.stack([x["audio"][:s] for x in batch])).float(),
        "text":torch.from_numpy(np.stack([x["text"][:s] for x in batch])).long(),
        "mask":torch.from_numpy(np.stack([x["segment_mask"][:s] for x in batch])).bool()}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def predict(model,loader,device,model_name):
    model.eval(); rows=[]
    with torch.inference_mode():
        for b in loader:
            z=model(b["audio"].to(device),b["text"].to(device),b["mask"].to(device)); p=torch.sigmoid(z).cpu().numpy()
            for pid,y,q in zip(b["participant_id"].numpy(),b["label"].numpy(),p):
                rows.append({"participant_id":int(pid),"label":int(y),"probability":float(q),"split":loader.dataset.split_name if hasattr(loader.dataset,"split_name") else "unknown","model":model_name})
    return pd.DataFrame(rows)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=103); p.add_argument("--epochs",type=int,default=120); p.add_argument("--patience",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=1e-4)
    a=p.parse_args(argv); seed_all(a.seed); feat,out=Path(a.features),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((feat/"feature_summary.json").read_text()); assert summary["test_opened"] is False
    vocab=json.loads((feat/"vocab.json").read_text()); train=Participants(feat/"train_features.npz"); dev=Participants(feat/"dev_features.npz")
    train.split_name="train"; dev.split_name="dev"; assert len(train)==107 and len(dev)==34
    y=train.d["labels"].astype(int); counts=np.bincount(y,minlength=2); pos_weight=float(counts[0]/counts[1])
    g=torch.Generator().manual_seed(a.seed)
    tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=g,collate_fn=collate,num_workers=0)
    tr_eval=DataLoader(train,batch_size=a.batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    dv=DataLoader(dev,batch_size=a.batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=ParticipantReLiMPNet(len(vocab),use_reliability_fusion=False).to(device)
    print("Device:",device,"| Parameters:",parameter_count(model),"| pos_weight:",pos_weight)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    best_key=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"
    for epoch in range(1,a.epochs+1):
        model.train(); total=n=0
        for b in tr:
            yy=b["label"].to(device); z=model(b["audio"].to(device),b["text"].to(device),b["mask"].to(device)); loss=loss_fn(z,yy)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach().cpu())*len(yy); n+=len(yy)
        d=predict(model,dv,device,"student_no_kd"); m=metric_dict(d.label,d.probability,.5); key=(m["macro_f1"],m["depressed_f1"],m["auroc"])
        row={"epoch":epoch,"train_loss":total/max(1,n),**{f"dev_{k}":v for k,v in m.items() if k!="confusion_matrix"}}; hist.append(row)
        pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")
        if key>best_key:
            best_key=key; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"threshold":.5,"vocab_size":len(vocab),"seed":a.seed,
                "student":"ParticipantReLiMPNet","training_config":vars(a),"feature_summary":summary},ckpt)
        else:
            stale+=1
            if stale>=a.patience: print("Early stop; best epoch",best_epoch); break
    state=torch.load(ckpt,map_location="cpu",weights_only=False); model.load_state_dict(state["model_state_dict"]); model.to(device)
    train_pred=predict(model,tr_eval,device,"student_no_kd"); dev_pred=predict(model,dv,device,"student_no_kd")
    tm=save_bundle(train_pred,out,"train",.5); dm=save_bundle(dev_pred,out,"dev",.5)
    protocol={"student":"ParticipantReLiMPNet","mode":"no_kd","hard_labels_only":True,"teacher_targets_loaded":False,
        "train_participants":107,"dev_participants":34,"participant_440_excluded":True,"checkpoint_selection":"DEV-34 participant macro-F1; threshold fixed at 0.5",
        "best_epoch":best_epoch,"test_opened":False,"test_prepared":False}
    (out/"metrics.json").write_text(json.dumps({"train":tm,"dev34":dm,"protocol":protocol},indent=2)+"\n")
    (out/"training_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    counts=dev.d["n_segments"]; target=int(np.median(counts)); idx=int(np.argmin(np.abs(counts-target))); s=int(counts[idx])
    audio=torch.from_numpy(dev.d["audio"][idx:idx+1,:s]).float(); text=torch.from_numpy(dev.d["text"][idx:idx+1,:s]).long(); mask=torch.from_numpy(dev.d["segment_mask"][idx:idx+1,:s]).bool()
    comp=profile_model(model,audio,text,mask,device,ckpt); comp["profile_participant_id"]=int(dev.d["participant_ids"][idx]); comp["profile_segments_basis"]="DEV median valid segment count"
    save_complexity(out/"model_complexity.json",comp)
    print("\nBEST DEV-34:",json.dumps(dm,indent=2)); print("\nCOMPLEXITY:",json.dumps(comp,indent=2)); print("\nTEST CLOSED.")
    return {"metrics":dm,"complexity":comp}
if __name__=="__main__": main()
