from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
from scripts.students.participant_student.model import ParticipantReLiMPNet,parameter_count
from scripts.students.participant_student.metrics import metric_dict,save_bundle
from scripts.students.participant_student.complexity import profile_model,save_complexity

class Participants(Dataset):
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z: self.d={k:z[k] for k in z.files}
        self.teacher=None; self.split_name="unknown"
    def attach_teacher(self,frame):
        m=frame.set_index("participant_id")
        ids=self.d["participant_ids"].astype(int)
        assert set(ids)==set(m.index.astype(int)) and len(ids)==len(m)
        m=m.loc[ids]
        assert np.array_equal(m.label.to_numpy(int),self.d["labels"].astype(int))
        self.teacher={
            "audio_probability":m.audio_probability.to_numpy(np.float32),
            "audio_logit":m.audio_logit.to_numpy(np.float32),
            "text_probability":m.text_probability.to_numpy(np.float32),
            "text_logit":m.text_logit.to_numpy(np.float32),
        }
    def __len__(self): return len(self.d["labels"])
    def __getitem__(self,i):
        r={k:self.d[k][i] for k in ["participant_ids","labels","audio","text","segment_mask","n_segments"]}
        if self.teacher is not None:
            for k,v in self.teacher.items(): r[k]=v[i]
        return r

def collate(batch):
    s=max(int(x["n_segments"]) for x in batch)
    out={"participant_id":torch.tensor([x["participant_ids"] for x in batch]).long(),
         "label":torch.tensor([x["labels"] for x in batch]).float(),
         "audio":torch.from_numpy(np.stack([x["audio"][:s] for x in batch])).float(),
         "text":torch.from_numpy(np.stack([x["text"][:s] for x in batch])).long(),
         "mask":torch.from_numpy(np.stack([x["segment_mask"][:s] for x in batch])).bool()}
    for k in ["audio_probability","audio_logit","text_probability","text_logit"]:
        if k in batch[0]: out[k]=torch.tensor([x[k] for x in batch]).float()
    return out

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_teacher_targets(audio_path,text_path):
    a=pd.read_csv(audio_path); t=pd.read_csv(text_path)
    req_a={"participant_id","label","kd_probability","kd_logit"}; req_t={"participant_id","label","text_probability","text_logit"}
    assert req_a<=set(a.columns),sorted(req_a-set(a.columns)); assert req_t<=set(t.columns),sorted(req_t-set(t.columns))
    assert len(a)==107 and a.participant_id.nunique()==107
    assert len(t)==107 and t.participant_id.nunique()==107
    assert set(a.participant_id.astype(int))==set(t.participant_id.astype(int))
    a=a[["participant_id","label","kd_probability","kd_logit"]].rename(columns={"kd_probability":"audio_probability","kd_logit":"audio_logit"})
    t=t[["participant_id","label","text_probability","text_logit"]]
    d=a.merge(t,on=["participant_id","label"],how="inner",validate="one_to_one").sort_values("participant_id").reset_index(drop=True)
    for p,z in [("audio_probability","audio_logit"),("text_probability","text_logit")]:
        q=1/(1+np.exp(-d[z].to_numpy(float)))
        assert np.allclose(q,d[p].to_numpy(float),atol=1e-5)
    assert d[["audio_probability","audio_logit","text_probability","text_logit"]].notna().all().all()
    return d

def kd_loss(student_logit,y,audio_logit,text_logit,pos_weight,temperature,alpha):
    hard=torch.nn.functional.binary_cross_entropy_with_logits(student_logit,y,pos_weight=pos_weight)
    T=float(temperature)
    qa=torch.sigmoid(audio_logit/T); qt=torch.sigmoid(text_logit/T)
    teacher=.5*(qa+qt)
    soft=torch.nn.functional.binary_cross_entropy_with_logits(student_logit/T,teacher)*(T*T)
    total=(1-alpha)*hard+alpha*soft
    return total,hard.detach(),soft.detach(),teacher.detach()

def predict(model,loader,device,name):
    model.eval(); rows=[]
    with torch.inference_mode():
        for b in loader:
            z=model(b["audio"].to(device),b["text"].to(device),b["mask"].to(device)); p=torch.sigmoid(z).cpu().numpy()
            for pid,y,q in zip(b["participant_id"].numpy(),b["label"].numpy(),p):
                rows.append({"participant_id":int(pid),"label":int(y),"probability":float(q),
                             "split":loader.dataset.split_name,"model":name})
    return pd.DataFrame(rows)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--features",required=True); p.add_argument("--audio-train-targets",required=True); p.add_argument("--text-train-targets",required=True)
    p.add_argument("--baseline-output",required=True); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=103); p.add_argument("--epochs",type=int,default=120); p.add_argument("--patience",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--temperature",type=float,default=2.0); p.add_argument("--lambda-kd",type=float,default=.5)
    a=p.parse_args(argv); assert 0<=a.lambda_kd<=1 and a.temperature>0
    seed_all(a.seed); feat,out=Path(a.features),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((feat/"feature_summary.json").read_text()); assert summary["test_opened"] is False and summary["test_prepared"] is False
    vocab=json.loads((feat/"vocab.json").read_text()); train=Participants(feat/"train_features.npz"); dev=Participants(feat/"dev_features.npz")
    train.split_name="train"; dev.split_name="dev"; assert len(train)==107 and len(dev)==34 and 440 not in set(dev.d["participant_ids"].astype(int))
    targets=read_teacher_targets(a.audio_train_targets,a.text_train_targets); train.attach_teacher(targets)
    targets.to_csv(out/"train_locked_teacher_targets.csv",index=False)

    y=train.d["labels"].astype(int); counts=np.bincount(y,minlength=2); pos_weight_value=float(counts[0]/counts[1])
    g=torch.Generator().manual_seed(a.seed)
    tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=g,collate_fn=collate,num_workers=0)
    tr_eval=DataLoader(train,batch_size=a.batch_size,shuffle=False,collate_fn=collate,num_workers=0)
    dv=DataLoader(dev,batch_size=a.batch_size,shuffle=False,collate_fn=collate,num_workers=0)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ParticipantReLiMPNet(len(vocab),use_reliability_fusion=False).to(device)
    params=parameter_count(model); baseline_complexity=json.loads((Path(a.baseline_output)/"model_complexity.json").read_text())
    assert params["total"]==baseline_complexity["total_parameters"],(params,baseline_complexity["total_parameters"])
    print("Device:",device,"| Parameters:",params,"| SAME AS NO-KD: yes | pos_weight:",pos_weight_value)
    print("Standard KD | T=",a.temperature,"| lambda_kd=",a.lambda_kd,"| teacher weights audio=text=0.5")

    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    pos_weight=torch.tensor(pos_weight_value,device=device)
    best_key=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"
    for epoch in range(1,a.epochs+1):
        model.train(); total=hard_total=soft_total=0.; n=0
        for b in tr:
            yy=b["label"].to(device); z=model(b["audio"].to(device),b["text"].to(device),b["mask"].to(device))
            loss,hard,soft,_=kd_loss(z,yy,b["audio_logit"].to(device),b["text_logit"].to(device),pos_weight,a.temperature,a.lambda_kd)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=len(yy); total+=float(loss.detach().cpu())*bs; hard_total+=float(hard.cpu())*bs; soft_total+=float(soft.cpu())*bs; n+=bs
        d=predict(model,dv,device,"student_standard_kd"); m=metric_dict(d.label,d.probability,.5); key=(m["macro_f1"],m["depressed_f1"],m["auroc"])
        row={"epoch":epoch,"train_loss":total/n,"train_hard_loss":hard_total/n,"train_kd_loss":soft_total/n,
             **{f"dev_{k}":v for k,v in m.items() if k!="confusion_matrix"}}
        hist.append(row); pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} hard={row['train_hard_loss']:.4f} kd={row['train_kd_loss']:.4f} "
              f"dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")
        if key>best_key:
            best_key=key; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"threshold":.5,"vocab_size":len(vocab),"seed":a.seed,
                "student":"ParticipantReLiMPNet","mode":"standard_kd","temperature":a.temperature,"lambda_kd":a.lambda_kd,
                "teacher_fusion":"equal per-teacher softened binary KD; audio=text=0.5","training_config":vars(a),"feature_summary":summary},ckpt)
        else:
            stale+=1
            if stale>=a.patience: print("Early stop; best epoch",best_epoch); break

    state=torch.load(ckpt,map_location="cpu",weights_only=False); model.load_state_dict(state["model_state_dict"]); model.to(device)
    train_pred=predict(model,tr_eval,device,"student_standard_kd"); dev_pred=predict(model,dv,device,"student_standard_kd")
    tm=save_bundle(train_pred,out,"train",.5); dm=save_bundle(dev_pred,out,"dev",.5)
    protocol={"student":"ParticipantReLiMPNet","mode":"standard_kd","same_architecture_as_no_kd":True,
        "teacher_targets":"locked TRAIN-107 participant targets only","teacher_weights":{"audio":.5,"text":.5},
        "temperature":a.temperature,"lambda_kd":a.lambda_kd,"hard_loss":"class-weighted BCE","kd_loss":"equal softened binary BCE, multiplied by T^2",
        "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
        "checkpoint_selection":"DEV-34 participant macro-F1; threshold fixed at 0.5","dev_teacher_targets_used_for_training":False,
        "test_opened":False,"test_prepared":False,"best_epoch":best_epoch}
    (out/"metrics.json").write_text(json.dumps({"train":tm,"dev34":dm,"protocol":protocol},indent=2)+"\n")
    (out/"training_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")

    counts=dev.d["n_segments"]; target=int(np.median(counts)); idx=int(np.argmin(np.abs(counts-target))); s=int(counts[idx])
    audio=torch.from_numpy(dev.d["audio"][idx:idx+1,:s]).float(); text=torch.from_numpy(dev.d["text"][idx:idx+1,:s]).long(); mask=torch.from_numpy(dev.d["segment_mask"][idx:idx+1,:s]).bool()
    comp=profile_model(model,audio,text,mask,device,ckpt); comp["profile_participant_id"]=int(dev.d["participant_ids"][idx])
    comp["profile_segments_basis"]="DEV median valid segment count"; comp["same_architecture_as_no_kd"]=True
    save_complexity(out/"model_complexity.json",comp)
    assert comp["total_parameters"]==baseline_complexity["total_parameters"]

    baseline_metrics=json.loads((Path(a.baseline_output)/"metrics.json").read_text())["dev34"]
    comparison=pd.DataFrame([
        {"model":"No-KD Student","accuracy":baseline_metrics["accuracy"],"macro_f1":baseline_metrics["macro_f1"],"depressed_f1":baseline_metrics["depressed_f1"],"auroc":baseline_metrics["auroc"],"parameters":baseline_complexity["total_parameters"],"checkpoint_mb":baseline_complexity["checkpoint_size_mb"],"profiled_gflops":baseline_complexity["profiled_gflops"]},
        {"model":"Standard-KD Student","accuracy":dm["accuracy"],"macro_f1":dm["macro_f1"],"depressed_f1":dm["depressed_f1"],"auroc":dm["auroc"],"parameters":comp["total_parameters"],"checkpoint_mb":comp["checkpoint_size_mb"],"profiled_gflops":comp["profiled_gflops"]},
    ])
    comparison.to_csv(out/"student_comparison_dev34.csv",index=False)
    print("\nBEST STANDARD-KD DEV-34:",json.dumps(dm,indent=2))
    print("\nCOMPLEXITY:",json.dumps(comp,indent=2))
    print("\n",comparison.to_string(index=False))
    print("\nTEST CLOSED.")
    return {"metrics":dm,"complexity":comp}

if __name__=="__main__": main()
