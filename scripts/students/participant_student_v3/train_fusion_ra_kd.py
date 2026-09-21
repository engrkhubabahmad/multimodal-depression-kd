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

    ct=1-np.exp(-np.abs(ut)); ca=1-np.exp(-np.abs(ua))
    den=ct+ca
    wt=np.divide(ct,den,out=np.full_like(ct,.5),where=den>1e-12)
    wa=np.divide(ca,den,out=np.full_like(ca,.5),where=den>1e-12)
    qra=wt*qt+wa*qa

    assert np.allclose(wt+wa,1.0,atol=1e-8)
    assert np.all((qra>0)&(qra<1))
    return {
        "text_raw_logit":tz.astype(np.float32),"audio_raw_logit":az.astype(np.float32),
        "text_scaled_logit":ut.astype(np.float32),"audio_scaled_logit":ua.astype(np.float32),
        "text_soft":qt.astype(np.float32),"audio_soft":qa.astype(np.float32),
        "text_reliability":ct.astype(np.float32),"audio_reliability":ca.astype(np.float32),
        "text_weight":wt.astype(np.float32),"audio_weight":wa.astype(np.float32),
        "ra_soft":qra.astype(np.float32),
        "text_scale":st,"audio_scale":sa
    }

def verify_sources(text_dir,audio_dir,teacher_text_dir,teacher_audio_dir,no_kd,std_kd):
    tm=json.loads((text_dir/"metrics.json").read_text()); am=json.loads((audio_dir/"metrics.json").read_text())
    nta=json.loads((teacher_text_dir/"text_target_audit.json").read_text())
    naa=json.loads((teacher_audio_dir/"train_run4_crop_audit.json").read_text())
    nm=json.loads((no_kd/"metrics.json").read_text()); bc=json.loads((no_kd/"backbone_config.json").read_text())
    sm=json.loads((std_kd/"metrics.json").read_text())
    sn=json.loads((std_kd/"teacher_normalization.json").read_text())

    assert tm["protocol"]["test_opened"] is False and am["protocol"]["test_opened"] is False
    assert nta["test_opened"] is False and naa["test_opened"] is False
    assert nm["protocol"]["test_opened"] is False and sm["protocol"]["test_opened"] is False
    assert nta["train_participants"]==107 and naa["participants"]==107
    assert bc["fusion_params"]==EXPECTED_FUSION_PARAMS and bc["full_student_params"]==254274
    assert bc["audio_dim"]==256 and bc["text_dim"]==64 and bc["projection_dim"]==96 and bc["hidden_dim"]==64
    assert sm["protocol"]["same_frozen_branch_embeddings"] is True
    assert sm["protocol"]["same_train_only_standardizers"] is True
    assert sm["protocol"]["same_fusion_architecture"] is True
    assert sm["protocol"]["same_optimizer_hparams"] is True
    assert sn["dev_used_for_scale"] is False and sn["test_opened"] is False
    return nm,bc,sm,sn

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--text-branch",required=True); p.add_argument("--audio-branch",required=True)
    p.add_argument("--teacher-text",required=True); p.add_argument("--teacher-audio",required=True)
    p.add_argument("--no-kd-reference",required=True); p.add_argument("--standard-kd-reference",required=True)
    p.add_argument("--output",required=True); p.add_argument("--seed",type=int,default=103)
    p.add_argument("--epochs",type=int,default=80); p.add_argument("--patience",type=int,default=20)
    a=p.parse_args(argv); seed_all(a.seed)

    td,ad=Path(a.text_branch),Path(a.audio_branch)
    ttd,tad=Path(a.teacher_text),Path(a.teacher_audio)
    ref,std,out=Path(a.no_kd_reference),Path(a.standard_kd_reference),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    no_kd,bc,std_metrics,std_norm=verify_sources(td,ad,ttd,tad,ref,std)
    base=no_kd["protocol"]; sprotocol=std_metrics["protocol"]

    T=float(sprotocol["temperature"]); kd_weight=float(sprotocol["kd_weight"])
    lr=float(base["lr"]); wd=float(base["weight_decay"]); batch=int(base["batch_size"]); dropout=float(bc["dropout"])
    assert int(sprotocol["seed"])==a.seed,"RA-KD seed must match Standard KD"
    assert abs(float(sprotocol["lr"])-lr)<1e-15 and abs(float(sprotocol["weight_decay"])-wd)<1e-15
    assert int(sprotocol["batch_size"])==batch and abs(float(sprotocol["dropout"])-dropout)<1e-15

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
                                tr_ids,ytr,T)
    assert np.isclose(target["audio_scale"],float(std_norm["audio_scale_median_abs_train_logit"]),rtol=0,atol=1e-12)
    assert np.isclose(target["text_scale"],float(std_norm["text_scale_median_abs_train_logit"]),rtol=0,atol=1e-12)
    qkd=target["ra_soft"]

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FrozenBranchFusion(dropout=dropout).to(device); params=parameter_count(model)
    assert params==EXPECTED_FUSION_PARAMS
    init_sha=state_sha256(model)
    std_init=str(sprotocol["initial_state_sha256"])
    assert init_sha==std_init,f"Initialization mismatch: RA={init_sha} Standard={std_init}"

    cnt=np.bincount(ytr,minlength=2); pos_weight=float(cnt[0]/cnt[1])
    hard_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device))
    kd_fn=torch.nn.BCEWithLogitsLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)

    ds=TensorDataset(torch.from_numpy(Atr),torch.from_numpy(Ttr),
                     torch.from_numpy(ytr.astype(np.float32)),torch.from_numpy(qkd))
    g=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,generator=g,num_workers=0)

    print("MULTIMODAL v3 RA-KD | exact Standard-KD backbone and normalized teachers")
    print("Device:",device,"| fusion params:",params,"| full student params:",params+AUDIO_BRANCH_PARAMS+TEXT_BRANCH_PARAMS)
    print("TRAIN:",len(ytr),"DEV:",len(ydv),"| seed:",a.seed,"| init_sha256:",init_sha)
    print("Matched Standard-KD init:",init_sha==std_init)
    print("Optimizer: AdamW | lr:",lr,"| weight_decay:",wd,"| batch:",batch,"| dropout:",dropout)
    print("KD: T=",T,"| lambda=",kd_weight)
    print("TRAIN-only teacher scales: audio median|z|=",target["audio_scale"],"| text median|z|=",target["text_scale"])
    print("Reliability: c_m=1-exp(-|z_m|/s_m); normalized per participant across modalities")
    print("Mean RA weights: audio=",float(target["audio_weight"].mean()),"| text=",float(target["text_weight"].mean()))
    print("Threshold=0.5 | DEV checkpoint selection unchanged | TEST CLOSED")

    table=pd.DataFrame({
        "participant_id":tr_ids,"label":ytr,
        "audio_teacher_raw_logit":target["audio_raw_logit"],"text_teacher_raw_logit":target["text_raw_logit"],
        "audio_teacher_scaled_logit":target["audio_scaled_logit"],"text_teacher_scaled_logit":target["text_scaled_logit"],
        "audio_teacher_soft":target["audio_soft"],"text_teacher_soft":target["text_soft"],
        "audio_reliability":target["audio_reliability"],"text_reliability":target["text_reliability"],
        "audio_weight":target["audio_weight"],"text_weight":target["text_weight"],
        "ra_kd_target":qkd
    })
    table.to_csv(out/"train_teacher_targets_reliability.csv",index=False)
    (out/"reliability_definition.json").write_text(json.dumps({
        "temperature":T,"kd_weight":kd_weight,
        "audio_scale_median_abs_train_logit":target["audio_scale"],
        "text_scale_median_abs_train_logit":target["text_scale"],
        "scale_source":"TRAIN-107 teacher logits only",
        "soft_target_formula":"q_m=sigmoid((z_m/s_m)/T)",
        "reliability_formula":"c_m=1-exp(-abs(z_m)/s_m)",
        "weight_formula":"w_m=c_m/(c_audio+c_text); 0.5/0.5 fallback only if denominator <=1e-12",
        "ra_target_formula":"q_RA=w_audio*q_audio+w_text*q_text",
        "mean_audio_weight":float(target["audio_weight"].mean()),
        "mean_text_weight":float(target["text_weight"].mean()),
        "dev_used_for_scale_or_reliability":False,"test_opened":False
    },indent=2)+"\n")

    best=(-1.,-1.,-1.); best_epoch=0; stale=0; hist=[]; ckpt=out/"best.pt"
    for epoch in range(1,a.epochs+1):
        model.train(); total=hard_total=kd_total=n=0
        for ab,tb,yb,qb in loader:
            ab,tb,yb,qb=ab.to(device),tb.to(device),yb.to(device),qb.to(device)
            z=model(ab,tb)
            hard=hard_fn(z,yb)
            kd=kd_fn(z/T,qb)*(T**2)
            loss=(1-kd_weight)*hard+kd_weight*kd
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
                        "temperature":T,"kd_weight":kd_weight,"initial_state_sha256":init_sha,
                        "backbone_config":bc,"teacher_scales":{"audio":target["audio_scale"],"text":target["text_scale"]},
                        "mean_reliability_weights":{"audio":float(target["audio_weight"].mean()),
                                                    "text":float(target["text_weight"].mean())}},ckpt)
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
      "mode":"Reliability-Aware KD multimodal v3",
      "no_kd_reference":str(ref),"standard_kd_reference":str(std),
      "same_frozen_branch_embeddings":True,"same_train_only_standardizers":True,
      "same_fusion_architecture":True,"same_optimizer_hparams":True,
      "same_initialization_as_standard_kd":True,
      "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
      "seed":a.seed,"initial_state_sha256":init_sha,
      "fusion_params":params,"full_student_params":params+AUDIO_BRANCH_PARAMS+TEXT_BRANCH_PARAMS,
      "optimizer":"AdamW","lr":lr,"weight_decay":wd,"batch_size":batch,"dropout":dropout,
      "hard_loss":"TRAIN-only class-weighted BCEWithLogitsLoss","pos_weight":pos_weight,
      "kd_loss":"T^2 * BCEWithLogits(student_logit/T, reliability-weighted normalized teacher target)",
      "kd_weight":kd_weight,"temperature":T,
      "teacher_logit_scaling":"identical to Standard KD: per modality median_TRAIN(abs(logit))",
      "reliability":"c_m=1-exp(-abs(z_m)/s_m); normalize across audio/text per TRAIN participant",
      "teacher_combination":"participant-specific reliability-weighted q_audio/q_text",
      "mean_audio_weight":float(target["audio_weight"].mean()),"mean_text_weight":float(target["text_weight"].mean()),
      "checkpoint_selection":"DEV-34 macro-F1, then depressed-F1, then AUROC",
      "threshold":0.5,"threshold_search":False,"best_epoch":best_epoch,
      "dev_teacher_targets_used_for_training":False,"dev_used_for_reliability":False,"test_opened":False
    }
    (out/"metrics.json").write_text(json.dumps({
        "train":mt,"dev34":md,"protocol":protocol,
        "no_kd_reference_dev34":no_kd["dev34"],
        "standard_kd_reference_dev34":std_metrics["dev34"]
    },indent=2)+"\n")

    print("\nBEST MULTIMODAL v3 RA-KD DEV-34:",json.dumps(md,indent=2))
    print("best_epoch:",best_epoch,"| init_sha256:",init_sha)
    print("mean_audio_weight:",float(target["audio_weight"].mean()),"| mean_text_weight:",float(target["text_weight"].mean()),"| TEST CLOSED.")

if __name__=="__main__": main()
