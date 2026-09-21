from __future__ import annotations
import argparse,json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix

from scripts.teachers.idiap_text.export_targets import main as export_text
from scripts.teachers.ussd_audio.audit_train_crop import main as export_audio_train
from scripts.teachers.ussd_audio.audit_frozen import main as export_audio_dev

ATOL=1e-6

def sigmoid(z):
    z=np.asarray(z,float)
    return 1.0/(1.0+np.exp(-z))

def binary_metrics(y,p,prob):
    y=np.asarray(y,int); p=np.asarray(p,int); prob=np.asarray(prob,float)
    return {
        "n":int(len(y)),
        "accuracy":float(accuracy_score(y,p)),
        "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
        "auroc":float(roc_auc_score(y,prob)),
        "confusion_matrix":confusion_matrix(y,p,labels=[0,1]).tolist(),
    }

def assert_frame_equal_numeric(saved,recomputed,keys,atol=ATOL):
    s=saved.sort_values("participant_id").reset_index(drop=True)
    r=recomputed.sort_values("participant_id").reset_index(drop=True)
    assert list(s.participant_id.astype(int))==list(r.participant_id.astype(int))
    for k in keys:
        assert k in s and k in r,k
        if np.issubdtype(s[k].dtype,np.number) and np.issubdtype(r[k].dtype,np.number):
            if not np.allclose(s[k].to_numpy(float),r[k].to_numpy(float),atol=atol,rtol=0,equal_nan=True):
                d=np.max(np.abs(s[k].to_numpy(float)-r[k].to_numpy(float)))
                raise AssertionError(f"{k}: saved/recomputed max abs diff={d}")
        else:
            assert s[k].astype(str).tolist()==r[k].astype(str).tolist(),k

def check_text(out):
    tr=pd.read_csv(out/"train_text_kd_targets.csv")
    dv=pd.read_csv(out/"dev_text_predictions.csv")
    audit=json.loads((out/"text_target_audit.json").read_text())
    assert len(tr)==107 and tr.participant_id.nunique()==107
    assert len(dv)==34 and dv.participant_id.nunique()==34
    assert 440 not in set(dv.participant_id.astype(int))
    assert not set(tr.participant_id.astype(int)) & set(dv.participant_id.astype(int))
    for name,d in [("text_train",tr),("text_dev",dv)]:
        assert d[["text_probability","text_logit"]].notna().all().all()
        assert ((d.text_probability>0)&(d.text_probability<1)).all()
        assert np.allclose(sigmoid(d.text_logit),d.text_probability,atol=ATOL,rtol=0),name
        assert np.array_equal((d.text_probability.to_numpy()>=.5).astype(int),d.text_prediction.to_numpy(int)),name
    mt=binary_metrics(tr.label,tr.text_prediction,tr.text_probability)
    md=binary_metrics(dv.label,dv.text_prediction,dv.text_probability)
    for k in ["n","accuracy","macro_f1","auroc","confusion_matrix"]:
        if k=="confusion_matrix":
            assert mt[k]==audit["train_in_sample"][k] and md[k]==audit["dev34"][k]
        else:
            assert np.allclose(mt[k],audit["train_in_sample"][k],atol=ATOL,rtol=0)
            assert np.allclose(md[k],audit["dev34"][k],atol=ATOL,rtol=0)
    assert audit["train_participants"]==107 and audit["dev_participants"]==34 and audit["test_opened"] is False
    return tr,dv,audit,mt,md

def check_audio(train_out,dev_out):
    tr=pd.read_csv(train_out/"train_run4_crop_kd_targets.csv")
    dv=pd.read_csv(dev_out/"dev_participant_predictions.csv")
    ta=json.loads((train_out/"train_run4_crop_audit.json").read_text())
    da=json.loads((dev_out/"audit.json").read_text())
    assert len(tr)==107 and tr.participant_id.nunique()==107
    assert len(dv)==34 and dv.participant_id.nunique()==34 and 440 not in set(dv.participant_id.astype(int))
    assert not set(tr.participant_id.astype(int)) & set(dv.participant_id.astype(int))
    for name,d in [("audio_train",tr),("audio_dev",dv)]:
        assert d[["kd_probability","kd_logit"]].notna().all().all()
        assert ((d.kd_probability>0)&(d.kd_probability<1)).all()
        assert np.allclose(sigmoid(d.kd_logit),d.kd_probability,atol=ATOL,rtol=0),name
        assert np.array_equal(np.rint(d.author_vote_fraction.to_numpy()).astype(int),d.author_prediction.to_numpy(int)),name
    # Author metrics use participant majority-vote predictions; AUROC uses soft mean probability.
    def am(d):
        y=d.label.to_numpy(int); p=d.author_prediction.to_numpy(int); q=d.kd_probability.to_numpy(float)
        return {"n":len(d),"accuracy":float(accuracy_score(y,p)),
                "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
                "auroc_soft_mean_probability":float(roc_auc_score(y,q)),
                "confusion_matrix":confusion_matrix(y,p,labels=[0,1]).tolist()}
    mt,md=am(tr),am(dv)
    for k in ["n","accuracy","macro_f1","auroc_soft_mean_probability","confusion_matrix"]:
        if k=="confusion_matrix":
            assert mt[k]==ta["metrics"][k] and md[k]==da["local_dev34"][k]
        else:
            assert np.allclose(mt[k],ta["metrics"][k],atol=ATOL,rtol=0)
            assert np.allclose(md[k],da["local_dev34"][k],atol=ATOL,rtol=0)
    assert ta["participants"]==107 and ta["test_opened"] is False
    assert da["participant_440_excluded"] is True and da["test_opened"] is False
    return tr,dv,ta,da,mt,md

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--daic-root",required=True)
    p.add_argument("--idiap-source",required=True)
    p.add_argument("--ussd-author-root",required=True)
    p.add_argument("--ussd-features",required=True)
    p.add_argument("--text-output",required=True)
    p.add_argument("--audio-train-output",required=True)
    p.add_argument("--audio-dev-output",required=True)
    p.add_argument("--deep",action="store_true",help="Re-infer all teacher outputs and compare row-for-row.")
    a=p.parse_args(argv)
    text_out=Path(a.text_output); atr_out=Path(a.audio_train_output); adv_out=Path(a.audio_dev_output)

    trt,dvt,taudit,mt_txt,md_txt=check_text(text_out)
    tra,dva,atraudit,adaudit,mt_aud,md_aud=check_audio(atr_out,adv_out)

    result={
      "saved_artifacts_verified":True,
      "text":{"train":mt_txt,"dev34":md_txt,
              "train_view":taudit["train_view"],"dev_view":taudit["dev_view"],
              "label_mismatch_ids":taudit.get("author_local_train_label_mismatches",[])},
      "audio":{"train":mt_aud,"dev34":md_aud,
               "train_protocol":atraudit["protocol"],"dev_protocol":adaudit["protocol"],
               "checkpoint_sha256":adaudit.get("checkpoint_sha256"),
               "normalization_sha256":adaudit.get("normalization_sha256")},
      "test_opened":False,
      "deep_reinference_verified":False
    }

    if a.deep:
        with tempfile.TemporaryDirectory(prefix="teacher_reverify_") as td:
            td=Path(td); tx=td/"text"; at=td/"audio_train"; ad=td/"audio_dev"
            export_text(["--daic-root",a.daic_root,"--source-root",a.idiap_source,"--output",str(tx)])
            export_audio_train(["--features",a.ussd_features,"--author-root",a.ussd_author_root,"--output",str(at)])
            export_audio_dev(["--features",a.ussd_features,"--author-root",a.ussd_author_root,"--output",str(ad)])
            rtrt=pd.read_csv(tx/"train_text_kd_targets.csv"); rdvt=pd.read_csv(tx/"dev_text_predictions.csv")
            rtra=pd.read_csv(at/"train_run4_crop_kd_targets.csv"); rdva=pd.read_csv(ad/"dev_participant_predictions.csv")
            assert_frame_equal_numeric(trt,rtrt,["participant_id","label","text_probability","text_logit","text_prediction"])
            assert_frame_equal_numeric(dvt,rdvt,["participant_id","label","text_probability","text_logit","text_prediction"])
            assert_frame_equal_numeric(tra,rtra,["participant_id","label","n_segments","author_vote_fraction","author_prediction","kd_probability","kd_logit","crop_start","crop_frames","run_seed"])
            assert_frame_equal_numeric(dva,rdva,["participant_id","label","n_segments","author_vote_fraction","author_prediction","kd_probability","kd_logit"])
            result["deep_reinference_verified"]=True

    print(json.dumps(result,indent=2))
    print("\nLOCKED TEACHER AUDIT: PASS")
    print("Saved probability/logit relations verified; metrics recomputed; TEST CLOSED.")
    return result

if __name__=="__main__": main()
