from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
from .fusion_model import FrozenBranchFusion,parameter_count

AUDIO_BRANCH_PARAMS=182529
TEXT_BRANCH_PARAMS=16128
EXPECTED_FUSION_PARAMS=55617

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

def load_npz(path):
    z=np.load(path)
    need={"participant_ids","labels","embedding","probability"}
    assert need<=set(z.files),(path,z.files)
    pid=z["participant_ids"].astype(int); y=z["labels"].astype(int)
    e=z["embedding"].astype(np.float32); p=z["probability"].astype(np.float32)
    assert len(pid)==len(y)==len(e)==len(p)
    assert len(set(pid.tolist()))==len(pid)
    assert np.isfinite(e).all() and np.isfinite(p).all()
    return pid,y,e,p

def align(audio_path,text_path,expected_n,split):
    ap,ay,ae,aprob=load_npz(audio_path); tp,ty,te,tprob=load_npz(text_path)
    A={int(p):i for i,p in enumerate(ap)}; T={int(p):i for i,p in enumerate(tp)}
    assert set(A)==set(T),f"{split}: audio/text participant ID mismatch"
    ids=np.array(sorted(A),dtype=int)
    assert len(ids)==expected_n,(split,len(ids),expected_n)
    aidx=np.array([A[int(p)] for p in ids]); tidx=np.array([T[int(p)] for p in ids])
    ya=ay[aidx]; yt=ty[tidx]
    assert np.array_equal(ya,yt),f"{split}: audio/text label mismatch"
    if split=="dev": assert 440 not in set(ids.tolist())
    return ids,ya,ae[aidx],te[tidx],aprob[aidx],tprob[tidx]

def fit_zscore(x):
    m=x.mean(0,dtype=np.float64).astype(np.float32)
    s=x.std(0,dtype=np.float64).astype(np.float32)
    s=np.where(s<1e-6,1.0,s).astype(np.float32)
    return m,s

def zscore(x,m,s):
    z=(x-m)/s
    assert np.isfinite(z).all()
    return z.astype(np.float32)

def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=(p>=.5).astype(int)
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "depressed_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
            "auroc":float(roc_auc_score(y,p)),
            "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def predict(model,A,T,device,batch=64):
    model.eval(); out=[]
    with torch.inference_mode():
        for i in range(0,len(A),batch):
            a=torch.from_numpy(A[i:i+batch]).to(device)
            t=torch.from_numpy(T[i:i+batch]).to(device)
            out.append(torch.sigmoid(model(a,t)).cpu().numpy())
    return np.concatenate(out)

def verify_branch_protocols(text_dir,audio_dir):
    tm=json.loads((text_dir/"metrics.json").read_text())
    am=json.loads((audio_dir/"metrics.json").read_text())
    tp=tm["protocol"]; ap=am["protocol"]
    assert tp["test_opened"] is False and ap["test_opened"] is False
    assert tp["train_participants"]==107 and tp["dev_participants"]==34
    assert ap["train_participants"]==107 and ap["dev_participants"]==34
    assert tp["participant_440_excluded"] is True and ap["participant_440_excluded"] is True
    va=tp.get("vocabulary_overlap_audit",{})
    if va.get("teacher_vectorizer_found"):
        assert va.get("overlap")==250 and abs(float(va.get("overlap_fraction",0))-1.0)<1e-12
    assert int(am.get("params",AUDIO_BRANCH_PARAMS))==AUDIO_BRANCH_PARAMS
    return tm,am

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--text-branch",required=True); p.add_argument("--audio-branch",required=True); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=103); p.add_argument("--epochs",type=int,default=80); p.add_argument("--patience",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--weight-decay",type=float,default=5e-4)
    p.add_argument("--dropout",type=float,default=.35)
    a=p.parse_args(argv); seed_all(a.seed)
    td,ad,out=Path(a.text_branch),Path(a.audio_branch),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tmetrics,ametrics=verify_branch_protocols(td,ad)

    tr=align(ad/"train_audio_embeddings.npz",td/"train_text_embeddings.npz",107,"train")
    dv=align(ad/"dev_audio_embeddings.npz",td/"dev_text_embeddings.npz",34,"dev")
    tr_ids,ytr,Atr,Ttr,atr_prob,ttr_prob=tr; dv_ids,ydv,Adv,Tdv,adv_prob,tdv_prob=dv
    assert not set(tr_ids.tolist()) & set(dv_ids.tolist())

    amean,astd=fit_zscore(Atr); tmean,tstd=fit_zscore(Ttr)
    Atr=zscore(Atr,amean,astd); Adv=zscore(Adv,amean,astd)
    Ttr=zscore(Ttr,tmean,tstd); Tdv=zscore(Tdv,tmean,tstd)
    np.savez_compressed(out/"train_only_standardizers.npz",audio_mean=amean,audio_std=astd,text_mean=tmean,text_std=tstd)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FrozenBranchFusion(dropout=a.dropout).to(device); params=parameter_count(model)
    assert params==EXPECTED_FUSION_PARAMS,(params,EXPECTED_FUSION_PARAMS)
    total_student=params+AUDIO_BRANCH_PARAMS+TEXT_BRANCH_PARAMS

    cnt=np.bincount(ytr,minlength=2); pos_weight=float(cnt[0]/cnt[1])
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)

    ds=TensorDataset(torch.from_numpy(Atr),torch.from_numpy(Ttr),torch.from_numpy(ytr.astype(np.float32)))
    g=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,generator=g,num_workers=0)

    print("MULTIMODAL v3 NO-KD | frozen branches + trainable fusion")
    print("Device:",device,"| fusion params:",params,"| full student params:",total_student)
    print("TRAIN:",len(ytr),"DEV:",len(ydv),"| class counts:",cnt.tolist(),"| pos_weight:",pos_weight)
    print("Audio embedding:",Atr.shape[1],"->96 | Text embedding:",Ttr.shape[1],"->96 | fusion 384->64->1")
    print("TRAIN-only z-score per modality | threshold=0.5 | TEST CLOSED")
    print("Optimizer: AdamW | lr:",a.lr,"| weight_decay:",a.weight_decay,"| batch:",a.batch_size)

    best=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"
    for epoch in range(1,a.epochs+1):
        model.train(); total=n=0
        for ab,tb,yb in loader:
            ab,tb,yb=ab.to(device),tb.to(device),yb.to(device)
            z=model(ab,tb); loss=loss_fn(z,yb)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach().cpu())*len(yb); n+=len(yb)

        pdev=predict(model,Adv,Tdv,device); m=metrics(ydv,pdev)
        key=(m["macro_f1"],m["depressed_f1"],m["auroc"])
        row={"epoch":epoch,"loss":total/n,**{f"dev_{k}":v for k,v in m.items() if k!="confusion_matrix"}}
        hist.append(row); pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={total/n:.4f} dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")

        if key>best:
            best=key; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,
                        "audio_mean":amean,"audio_std":astd,"text_mean":tmean,"text_std":tstd,
                        "config":{"audio_dim":256,"text_dim":64,"d_model":96,"hidden":64,"dropout":a.dropout}},ckpt)
            pd.DataFrame({"participant_id":dv_ids,"label":ydv,"probability":pdev,
                          "prediction":(pdev>=.5).astype(int),
                          "audio_branch_probability":adv_prob,
                          "text_branch_probability":tdv_prob}).to_csv(out/"best_dev_predictions.csv",index=False)
        else:
            stale+=1
            if stale>=a.patience:
                print("Early stop; best epoch",best_epoch); break

    state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state["model_state_dict"]); model.eval()
    ptr=predict(model,Atr,Ttr,device); pdv=predict(model,Adv,Tdv,device)
    mt,md=metrics(ytr,ptr),metrics(ydv,pdv)

    def save_pred(path,ids,y,p,ap,tp):
        pd.DataFrame({"participant_id":ids,"label":y,"probability":p,"prediction":(p>=.5).astype(int),
                      "audio_branch_probability":ap,"text_branch_probability":tp}).to_csv(path,index=False)
    save_pred(out/"train_predictions.csv",tr_ids,ytr,ptr,atr_prob,ttr_prob)
    save_pred(out/"dev_predictions.csv",dv_ids,ydv,pdv,adv_prob,tdv_prob)

    protocol={
      "mode":"No-KD multimodal v3","branch_policy":"frozen pretrained hard-label branches; fusion head only trainable",
      "audio_branch":"v3 author-recipe compressed hard-label segment branch; best checkpoint embeddings",
      "text_branch":"v3 independent InducT-style top250 hard-label branch; best checkpoint embeddings",
      "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
      "input_standardization":"per-dimension mean/std fit on TRAIN-107 embeddings only",
      "train_class_weight_source":"TRAIN-107 only","pos_weight":pos_weight,
      "fusion":"audio256->96; text64->96; [a,t,a*t,abs(a-t)] 384->64->1",
      "fusion_params":params,"audio_branch_params":AUDIO_BRANCH_PARAMS,"text_branch_params":TEXT_BRANCH_PARAMS,
      "full_student_params":total_student,
      "optimizer":"AdamW","lr":a.lr,"weight_decay":a.weight_decay,"batch_size":a.batch_size,
      "checkpoint_selection":"DEV-34 macro-F1, then depressed-F1, then AUROC",
      "threshold":0.5,"threshold_search":False,"best_epoch":best_epoch,
      "teacher_targets_used":False,"teacher_checkpoint_loaded":False,"test_opened":False
    }
    (out/"metrics.json").write_text(json.dumps({"train":mt,"dev34":md,"protocol":protocol,
                                                "source_branch_metrics":{"text_dev34":tmetrics["dev34"],"audio_dev34":ametrics["dev34"]}},indent=2)+"\n")
    (out/"backbone_config.json").write_text(json.dumps({"audio_dim":256,"text_dim":64,"projection_dim":96,"fusion_input_dim":384,
                                                        "hidden_dim":64,"dropout":a.dropout,"fusion_params":params,
                                                        "full_student_params":total_student},indent=2)+"\n")
    print("\nBEST MULTIMODAL v3 NO-KD DEV-34:",json.dumps(md,indent=2))
    print("best_epoch:",best_epoch,"| fusion_params:",params,"| full_student_params:",total_student,"| TEST CLOSED.")

if __name__=="__main__": main()
