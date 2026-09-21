from __future__ import annotations
import argparse,hashlib,json,random
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

def state_sha256(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

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
    ids=np.array(sorted(A),dtype=int); assert len(ids)==expected_n,(split,len(ids),expected_n)
    ai=np.array([A[int(p)] for p in ids]); ti=np.array([T[int(p)] for p in ids])
    assert np.array_equal(ay[ai],ty[ti]),f"{split}: audio/text label mismatch"
    if split=="dev": assert 440 not in set(ids.tolist())
    return ids,ay[ai],ae[ai],te[ti],aprob[ai],tprob[ti]

def zscore(x,m,s):
    z=(x-m)/s; assert np.isfinite(z).all(); return z.astype(np.float32)

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
            out.append(torch.sigmoid(model(torch.from_numpy(A[i:i+batch]).to(device),
                                           torch.from_numpy(T[i:i+batch]).to(device))).cpu().numpy())
    return np.concatenate(out)

def load_teacher_targets(text_path,audio_path,ids,y,T):
    tx=pd.read_csv(text_path); au=pd.read_csv(audio_path)
    for name,d,prob,logit in [
        ("text",tx,"text_probability","text_logit"),
        ("audio",au,"kd_probability","kd_logit")
    ]:
        assert len(d)==107 and d.participant_id.nunique()==107,(name,len(d))
        assert d[[prob,logit]].notna().all().all()
        q=d[prob].to_numpy(float); z=d[logit].to_numpy(float)
        assert np.all((q>0)&(q<1)),name
        assert np.allclose(1/(1+np.exp(-z)),q,atol=1e-6,rtol=0),f"{name}: probability/logit mismatch"

    TX={int(r.participant_id):r for r in tx.itertuples(index=False)}
    AU={int(r.participant_id):r for r in au.itertuples(index=False)}
    assert set(ids.tolist())==set(TX)==set(AU),"teacher/student TRAIN participant mismatch"
    tz=np.array([float(TX[int(p)].text_logit) for p in ids],dtype=np.float64)
    az=np.array([float(AU[int(p)].kd_logit) for p in ids],dtype=np.float64)
    ty=np.array([int(TX[int(p)].label) for p in ids]); ay=np.array([int(AU[int(p)].label) for p in ids])
    assert np.array_equal(ty,y) and np.array_equal(ay,y),"teacher/student TRAIN label mismatch"

    st=float(np.median(np.abs(tz))); sa=float(np.median(np.abs(az)))
    assert st>0 and sa>0 and np.isfinite([st,sa]).all()
    ut=tz/st; ua=az/sa
    qt=1/(1+np.exp(-(ut/T))); qa=1/(1+np.exp(-(ua/T)))
    qeq=.5*(qt+qa)
    return {
        "text_raw_logit":tz.astype(np.float32),"audio_raw_logit":az.astype(np.float32),
        "text_scaled_logit":ut.astype(np.float32),"audio_scaled_logit":ua.astype(np.float32),
        "text_soft":qt.astype(np.float32),"audio_soft":qa.astype(np.float32),
        "equal_soft":qeq.astype(np.float32),
        "text_scale":st,"audio_scale":sa
    }

def verify_sources(text_dir,audio_dir,teacher_text_dir,teacher_audio_dir,no_kd):
    tm=json.loads((text_dir/"metrics.json").read_text()); am=json.loads((audio_dir/"metrics.json").read_text())
    nta=json.loads((teacher_text_dir/"text_target_audit.json").read_text())
    naa=json.loads((teacher_audio_dir/"train_run4_crop_audit.json").read_text())
    nm=json.loads((no_kd/"metrics.json").read_text()); bc=json.loads((no_kd/"backbone_config.json").read_text())
    assert tm["protocol"]["test_opened"] is False and am["protocol"]["test_opened"] is False
    assert nta["test_opened"] is False and naa["test_opened"] is False and nm["protocol"]["test_opened"] is False
    assert nta["train_participants"]==107 and naa["participants"]==107
    assert bc["fusion_params"]==EXPECTED_FUSION_PARAMS and bc["full_student_params"]==254274
    assert bc["audio_dim"]==256 and bc["text_dim"]==64 and bc["projection_dim"]==96 and bc["hidden_dim"]==64
    return nm,bc

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--text-branch",required=True); p.add_argument("--audio-branch",required=True)
    p.add_argument("--teacher-text",required=True); p.add_argument("--teacher-audio",required=True)
    p.add_argument("--no-kd-reference",required=True); p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=103); p.add_argument("--epochs",type=int,default=80); p.add_argument("--patience",type=int,default=20)
    p.add_argument("--temperature",type=float,default=2.0); p.add_argument("--kd-weight",type=float,default=.5)
    a=p.parse_args(argv)
    assert a.temperature>0 and 0<=a.kd_weight<=1
    seed_all(a.seed)

    td,ad=Path(a.text_branch),Path(a.audio_branch)
    ttd,tad=Path(a.teacher_text),Path(a.teacher_audio)
    ref,out=Path(a.no_kd_reference),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    no_kd,bc=verify_sources(td,ad,ttd,tad,ref)
    base=no_kd["protocol"]
    lr=float(base["lr"]); wd=float(base["weight_decay"]); batch=int(base["batch_size"]); dropout=float(bc["dropout"])

    tr=align(ad/"train_audio_embeddings.npz",td/"train_text_embeddings.npz",107,"train")
    dv=align(ad/"dev_audio_embeddings.npz",td/"dev_text_embeddings.npz",34,"dev")
    tr_ids,ytr,Atr,Ttr,atr_prob,ttr_prob=tr; dv_ids,ydv,Adv,Tdv,adv_prob,tdv_prob=dv
    assert not set(tr_ids.tolist())&set(dv_ids.tolist())

    sc=np.load(ref/"train_only_standardizers.npz")
    amean,astd=sc["audio_mean"].astype(np.float32),sc["audio_std"].astype(np.float32)
    tmean,tstd=sc["text_mean"].astype(np.float32),sc["text_std"].astype(np.float32)
    Atr=zscore(Atr,amean,astd); Adv=zscore(Adv,amean,astd)
    Ttr=zscore(Ttr,tmean,tstd); Tdv=zscore(Tdv,tmean,tstd)

    target=load_teacher_targets(ttd/"train_text_kd_targets.csv",
                                tad/"train_run4_crop_kd_targets.csv",
                                tr_ids,ytr,a.temperature)
    qkd=target["equal_soft"]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FrozenBranchFusion(dropout=dropout).to(device); params=parameter_count(model)
    assert params==EXPECTED_FUSION_PARAMS
    init_sha=state_sha256(model)

    cnt=np.bincount(ytr,minlength=2); pos_weight=float(cnt[0]/cnt[1])
    hard_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device))
    kd_fn=torch.nn.BCEWithLogitsLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)

    ds=TensorDataset(torch.from_numpy(Atr),torch.from_numpy(Ttr),
                     torch.from_numpy(ytr.astype(np.float32)),torch.from_numpy(qkd))
    g=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,generator=g,num_workers=0)

    print("MULTIMODAL v3 STANDARD KD | exact frozen-branch fusion backbone")
    print("Device:",device,"| fusion params:",params,"| full student params:",params+AUDIO_BRANCH_PARAMS+TEXT_BRANCH_PARAMS)
    print("TRAIN:",len(ytr),"DEV:",len(ydv),"| seed:",a.seed,"| init_sha256:",init_sha)
    print("Reference optimizer: AdamW | lr:",lr,"| weight_decay:",wd,"| batch:",batch,"| dropout:",dropout)
    print("KD: T=",a.temperature,"| lambda=",a.kd_weight,"| equal audio/text teacher weight")
    print("TRAIN-only teacher scales: audio median|z|=",target["audio_scale"],"| text median|z|=",target["text_scale"])
    print("Teacher q: sigmoid((z/median|z|)/T); student KD: BCEWithLogits(student_logit/T,q) * T^2")
    print("Threshold=0.5 | DEV checkpoint selection unchanged | TEST CLOSED")

    pd.DataFrame({
        "participant_id":tr_ids,"label":ytr,
        "audio_teacher_raw_logit":target["audio_raw_logit"],
        "text_teacher_raw_logit":target["text_raw_logit"],
        "audio_teacher_scaled_logit":target["audio_scaled_logit"],
        "text_teacher_scaled_logit":target["text_scaled_logit"],
        "audio_teacher_soft":target["audio_soft"],
        "text_teacher_soft":target["text_soft"],
        "standard_kd_target":qkd
    }).to_csv(out/"train_teacher_targets_normalized.csv",index=False)
    (out/"teacher_normalization.json").write_text(json.dumps({
        "temperature":a.temperature,"audio_scale_median_abs_train_logit":target["audio_scale"],
        "text_scale_median_abs_train_logit":target["text_scale"],
        "scale_source":"TRAIN-107 teacher logits only",
        "standard_kd_teacher_weight":{"audio":0.5,"text":0.5},
        "formula":"q_m=sigmoid((z_m/median_TRAIN(abs(z_m)))/T); q_standard=(q_audio+q_text)/2",
        "dev_used_for_scale":False,"test_opened":False
    },indent=2)+"\n")

    best=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"
    for epoch in range(1,a.epochs+1):
        model.train(); total=hard_total=kd_total=n=0
        for ab,tb,yb,qb in loader:
            ab,tb,yb,qb=ab.to(device),tb.to(device),yb.to(device),qb.to(device)
            z=model(ab,tb)
            hard=hard_fn(z,yb)
            kd=kd_fn(z/a.temperature,qb)*(a.temperature**2)
            loss=(1-a.kd_weight)*hard+a.kd_weight*kd
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs=len(yb); total+=float(loss.detach().cpu())*bs; hard_total+=float(hard.detach().cpu())*bs; kd_total+=float(kd.detach().cpu())*bs; n+=bs

        pdev=predict(model,Adv,Tdv,device); m=metrics(ydv,pdev); key=(m["macro_f1"],m["depressed_f1"],m["auroc"])
        row={"epoch":epoch,"loss":total/n,"hard_loss":hard_total/n,"kd_loss":kd_total/n,
             **{f"dev_{k}":v for k,v in m.items() if k!="confusion_matrix"}}
        hist.append(row); pd.DataFrame(hist).to_csv(out/"history.csv",index=False)
        print(f"epoch={epoch:03d} loss={row['loss']:.4f} hard={row['hard_loss']:.4f} kd={row['kd_loss']:.4f} "
              f"dev_macroF1={m['macro_f1']:.4f} depF1={m['depressed_f1']:.4f} AUROC={m['auroc']:.4f}")

        if key>best:
            best=key; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"seed":a.seed,
                        "temperature":a.temperature,"kd_weight":a.kd_weight,
                        "initial_state_sha256":init_sha,"backbone_config":bc,
                        "teacher_scales":{"audio":target["audio_scale"],"text":target["text_scale"]}},ckpt)
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
      "mode":"Standard KD multimodal v3",
      "backbone_reference":str(ref),"same_frozen_branch_embeddings":True,"same_train_only_standardizers":True,
      "same_fusion_architecture":True,"same_optimizer_hparams":True,
      "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
      "seed":a.seed,"initial_state_sha256":init_sha,
      "fusion_params":params,"full_student_params":params+AUDIO_BRANCH_PARAMS+TEXT_BRANCH_PARAMS,
      "optimizer":"AdamW","lr":lr,"weight_decay":wd,"batch_size":batch,"dropout":dropout,
      "hard_loss":"TRAIN-only class-weighted BCEWithLogitsLoss","pos_weight":pos_weight,
      "kd_loss":"T^2 * BCEWithLogits(student_logit/T, equal-weight normalized teacher soft target)",
      "kd_weight":a.kd_weight,"temperature":a.temperature,
      "teacher_logit_scaling":"per modality median_TRAIN(abs(logit)); no DEV-derived scale",
      "teacher_combination":"equal 0.5 audio + 0.5 text after scale normalization",
      "checkpoint_selection":"DEV-34 macro-F1, then depressed-F1, then AUROC",
      "threshold":0.5,"threshold_search":False,"best_epoch":best_epoch,
      "dev_teacher_targets_used_for_training":False,"test_opened":False
    }
    (out/"metrics.json").write_text(json.dumps({"train":mt,"dev34":md,"protocol":protocol,
      "no_kd_reference_dev34":no_kd["dev34"]},indent=2)+"\n")
    print("\nBEST MULTIMODAL v3 STANDARD-KD DEV-34:",json.dumps(md,indent=2))
    print("best_epoch:",best_epoch,"| init_sha256:",init_sha,"| TEST CLOSED.")

if __name__=="__main__": main()
