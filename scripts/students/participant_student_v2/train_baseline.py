from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from scripts.students.participant_student.metrics import metric_dict,save_bundle
from .data import RichParticipantDataset,collate
from .model import RichParticipantStudent,parameter_count
from .complexity import profile_model,save as save_complexity

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def predict(model,loader,device,name):
    model.eval(); rows=[]
    with torch.inference_mode():
        for b in loader:
            z=model(b["audio"].to(device),b["text"].to(device),b["mask"].to(device)); p=torch.sigmoid(z).cpu().numpy()
            for pid,y,q in zip(b["participant_id"].numpy(),b["label"].numpy(),p):
                rows.append({"participant_id":int(pid),"label":int(y),"probability":float(q),"split":loader.dataset.split,"model":name})
    return pd.DataFrame(rows)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--output",required=True)
    p.add_argument("--local-audio-root",default=None); p.add_argument("--seed",type=int,default=103)
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--patience",type=int,default=20); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--weight-decay",type=float,default=5e-4)
    a=p.parse_args(argv); seed_all(a.seed); feat,out=Path(a.features),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((feat/"feature_summary.json").read_text()); assert summary["test_opened"] is False and summary["test_prepared"] is False
    train=RichParticipantDataset(feat/"train_manifest.csv",feat/"train_text_tfidf.npy",feat/"audio_standardizer.npz","train",a.seed,local_audio_root=a.local_audio_root)
    dev=RichParticipantDataset(feat/"dev_manifest.csv",feat/"dev_text_tfidf.npy",feat/"audio_standardizer.npz","dev",a.seed,local_audio_root=a.local_audio_root)
    assert len(train)==107 and len(dev)==34 and 440 not in set(dev.meta.participant_id.astype(int))
    y=train.meta.label.to_numpy(int); counts=np.bincount(y,minlength=2); pos_weight=float(counts[0]/counts[1])
    g=torch.Generator().manual_seed(a.seed)
    tr=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=g,collate_fn=collate,num_workers=0)
    tr_eval=DataLoader(train,batch_size=max(1,a.batch_size//2),shuffle=False,collate_fn=collate,num_workers=0)
    dv=DataLoader(dev,batch_size=max(1,a.batch_size//2),shuffle=False,collate_fn=collate,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=RichParticipantStudent(text_dim=summary["text_features"]).to(device)
    params=parameter_count(model); assert params["total"]<300000,params
    print("Device:",device,"| Rich student parameters:",params,"| Audio teacher params=1,153,537 | Text teacher params=16,128")
    print("Student/audio-teacher parameter ratio:",params["total"]/1153537)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    best_key=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"
    for epoch in range(1,a.epochs+1):
        train.set_epoch(epoch); model.train(); total=n=0
        for b in tr:
            yy=b["label"].to(device); z=model(b["audio"].to(device),b["text"].to(device),b["mask"].to(device)); loss=loss_fn(z,yy)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach().cpu())*len(yy); n+=len(yy)
        d=predict(model,dv,device,"rich_student_no_kd"); m=metric_dict(d.label,d.probability,.5); key=(m["macro_f1"],m["depressed_f1"],m["auroc"])
        row={"epoch":epoch,"train_loss":total/max(1,n),**{f"dev_{k}":v for k,v in m.items() if k!="confusion_matrix"}}; hist.append(row)
        pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")
        if key>best_key:
            best_key=key; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"threshold":.5,"seed":a.seed,
                        "student":"RichParticipantStudent-v2","training_config":vars(a),"feature_summary":summary},ckpt)
        else:
            stale+=1
            if stale>=a.patience: print("Early stop; best epoch",best_epoch); break
    state=torch.load(ckpt,map_location="cpu",weights_only=False); model.load_state_dict(state["model_state_dict"]); model.to(device)
    train.set_epoch(0)
    tm=save_bundle(predict(model,tr_eval,device,"rich_student_no_kd"),out,"train",.5)
    dm=save_bundle(predict(model,dv,device,"rich_student_no_kd"),out,"dev",.5)
    protocol={"student":"RichParticipantStudent-v2","mode":"no_kd","hard_labels_only":True,"teacher_predictions_loaded":False,
              "audio_representation":"cached USSD-compatible ComParE16 input features, independently TRAIN-normalized",
              "text_representation":"independently fitted TRAIN-only top-250 TF-IDF",
              "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
              "checkpoint_selection":"DEV-34 participant macro-F1; threshold fixed at 0.5","best_epoch":best_epoch,
              "test_opened":False,"test_prepared":False}
    (out/"metrics.json").write_text(json.dumps({"train":tm,"dev34":dm,"protocol":protocol},indent=2)+"\n")
    (out/"training_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    b=next(iter(DataLoader(dev,batch_size=1,shuffle=False,collate_fn=collate,num_workers=0)))
    comp=profile_model(model,b["audio"],b["text"],b["mask"],device,ckpt); comp["audio_teacher_parameters"]=1153537; comp["text_teacher_parameters"]=16128
    comp["parameter_ratio_vs_audio_teacher"]=comp["total_parameters"]/1153537; save_complexity(out/"model_complexity.json",comp)
    design={"student_parameters":params["total"],"audio_teacher_parameters":1153537,"text_teacher_parameters":16128,
            "student_is_smaller_than_audio_teacher":params["total"]<1153537,
            "student_uses_teacher_predictions_at_inference":False,
            "student_uses_teacher_hidden_states":False}
    (out/"design_audit.json").write_text(json.dumps(design,indent=2)+"\n")
    print("\nBEST RICH NO-KD DEV-34:",json.dumps(dm,indent=2)); print("\nCOMPLEXITY:",json.dumps(comp,indent=2)); print("\nTEST CLOSED.")
    return {"metrics":dm,"complexity":comp}

if __name__=="__main__": main()
