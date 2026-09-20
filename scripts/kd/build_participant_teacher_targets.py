"""Align locked participant-level audio and text teacher targets.

TRAIN: USSD run-4 author-style crop + Idiap training-graph document nodes.
DEV:   USSD full-stream author evaluation + Idiap saved A_dev.
TEST is never read or created.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd

def conf(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    h=-(p*np.log(p)+(1-p)*np.log(1-p))/math.log(2.0)
    return np.clip(1-h,1e-3,1.0)

def prep_audio(path,split):
    d=pd.read_csv(path); req={"participant_id","label","kd_probability","kd_logit"}
    if not req<=set(d.columns): raise ValueError(f"{path}: missing {sorted(req-set(d.columns))}")
    return d[["participant_id","label","kd_probability","kd_logit"]].rename(columns={"kd_probability":"audio_probability","kd_logit":"audio_logit"})

def prep_text(path,split):
    d=pd.read_csv(path); req={"participant_id","label","text_probability","text_logit"}
    if not req<=set(d.columns): raise ValueError(f"{path}: missing {sorted(req-set(d.columns))}")
    return d[["participant_id","label","text_probability","text_logit"]]

def build(audio,text,split,expected,out):
    a=prep_audio(audio,split); t=prep_text(text,split)
    if a.participant_id.duplicated().any() or t.participant_id.duplicated().any(): raise AssertionError(f"{split}: duplicate participant")
    d=a.merge(t,on=["participant_id","label"],how="inner",validate="one_to_one")
    if len(d)!=len(a) or len(d)!=len(t) or len(d)!=expected: raise AssertionError(f"{split}: alignment mismatch audio={len(a)} text={len(t)} merged={len(d)} expected={expected}")
    d["split"]=split
    d["audio_confidence"]=conf(d.audio_probability); d["text_confidence"]=conf(d.text_probability)
    d["teacher_disagreement"]=(d.audio_probability-d.text_probability).abs()
    d["equal_teacher_probability"]=0.5*(d.audio_probability+d.text_probability)
    z=d.audio_confidence+d.text_confidence
    d["audio_reliability_weight"]=d.audio_confidence/z; d["text_reliability_weight"]=d.text_confidence/z
    d["reliability_teacher_probability"]=d.audio_reliability_weight*d.audio_probability+d.text_reliability_weight*d.text_probability
    d["equal_teacher_logit"]=np.log(np.clip(d.equal_teacher_probability,1e-6,1-1e-6)/np.clip(1-d.equal_teacher_probability,1e-6,1-1e-6))
    d["reliability_teacher_logit"]=np.log(np.clip(d.reliability_teacher_probability,1e-6,1-1e-6)/np.clip(1-d.reliability_teacher_probability,1e-6,1-1e-6))
    cols=["participant_id","split","label","audio_probability","audio_logit","text_probability","text_logit",
          "audio_confidence","text_confidence","teacher_disagreement","audio_reliability_weight","text_reliability_weight",
          "equal_teacher_probability","equal_teacher_logit","reliability_teacher_probability","reliability_teacher_logit"]
    d=d[cols].sort_values("participant_id").reset_index(drop=True); p=out/f"{split}_participant_teacher_targets.csv"; d.to_csv(p,index=False)
    q={"n":len(d),"audio_prob_mean":float(d.audio_probability.mean()),"text_prob_mean":float(d.text_probability.mean()),
       "mean_disagreement":float(d.teacher_disagreement.mean()),"median_disagreement":float(d.teacher_disagreement.median()),
       "mean_audio_confidence":float(d.audio_confidence.mean()),"mean_text_confidence":float(d.text_confidence.mean()),
       "mean_audio_weight":float(d.audio_reliability_weight.mean()),"mean_text_weight":float(d.text_reliability_weight.mean())}
    return d,p,q

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--audio-root",required=True); p.add_argument("--text-root",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(argv); ar,tr,out=Path(a.audio_root),Path(a.text_root),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    train,tp,tq=build(ar/"train_run4_crop_kd_targets.csv",tr/"train_text_kd_targets.csv","train",107,out)
    dev,dp,dq=build(ar/"dev_predictions.csv",tr/"dev_text_predictions.csv","dev",34,out)
    if 440 in set(dev.participant_id): raise AssertionError("DEV participant 440 must remain excluded")
    if set(train.participant_id)&set(dev.participant_id): raise AssertionError("TRAIN/DEV overlap")
    qc={"protocol":"participant-level locked-teacher alignment","train_participants":107,"dev_participants":34,"participant_440_excluded":True,"test_opened":False,
        "audio_train_view":"USSD run-4 seed-1300 6662-frame author-style crop","audio_dev_view":"USSD full-stream author aggregation",
        "text_train_view":"Idiap checkpoint training-graph document nodes","text_dev_view":"Idiap checkpoint saved A_dev",
        "reliability":"label-free normalized binary entropy confidence; no DEV-label-derived teacher weighting",
        "train":tq,"dev":dq,"train_output":str(tp),"dev_output":str(dp)}
    (out/"qc.json").write_text(json.dumps(qc,indent=2)+"\n"); print(json.dumps(qc,indent=2)); return qc
if __name__=="__main__": main()
