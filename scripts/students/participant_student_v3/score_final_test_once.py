from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,classification_report

from .pretrain_text import one
from scripts.teachers.ussd_audio.common import sha256

def metrics(y,p,pred):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=np.asarray(pred,int)
    return {
      "n":int(len(y)),
      "accuracy":float(accuracy_score(y,pred)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "depressed_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
      "auroc":float(roc_auc_score(y,p)),
      "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
      "classification_report":classification_report(
          y,pred,labels=[0,1],target_names=["Non-depressed","Depressed"],
          output_dict=True,zero_division=0)
    }

def read_ground_truth(root,expected_ids):
    p=one(root,"full_test_split.csv")
    d=pd.read_csv(p); c={x.casefold():x for x in d.columns}
    if "participant_id" not in c: raise ValueError(f"{p}: missing Participant_ID")
    label_col=c.get("phq_binary",c.get("phq8_binary"))
    if label_col is None: raise ValueError(f"{p}: expected PHQ_Binary or PHQ8_Binary")
    g=pd.DataFrame({
        "participant_id":d[c["participant_id"]].astype(int),
        "label":d[label_col].astype(int)
    }).drop_duplicates("participant_id").sort_values("participant_id").reset_index(drop=True)
    if len(g)!=47: raise AssertionError(f"Expected 47 labeled TEST participants; got {len(g)}")
    if set(g.participant_id)!=set(map(int,expected_ids)):
        raise AssertionError("full_test_split IDs != frozen blind student prediction IDs")
    if not set(g.label.unique())<={0,1}: raise AssertionError("TEST labels are not binary")
    return g,p,label_col

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--daic-root",required=True)
    ap.add_argument("--blind-output",required=True)
    ap.add_argument("--dev-freeze",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args(argv)
    root,blind_dir,freeze,out=Path(a.daic_root),Path(a.blind_output),Path(a.dev_freeze),Path(a.output)
    out.mkdir(parents=True,exist_ok=True)

    marker=out/"FINAL_TEST_SCORED.json"
    if marker.exists() or (out/"FINAL_TEST_METRICS.json").exists():
        raise RuntimeError("FINAL TEST already scored. No repeat evaluation permitted.")

    pred_path=blind_dir/"blind_student_only_predictions.csv"
    man_path=blind_dir/"blind_student_only_manifest.json"
    if not pred_path.exists() or not man_path.exists():
        raise FileNotFoundError(
            "Student-only blind freeze missing. Run freeze_student_only_test_predictions.py BEFORE opening TEST labels."
        )

    manifest=json.loads(man_path.read_text())
    fr=json.loads((freeze/"final_dev_selection.json").read_text())
    assert manifest["status"]=="STUDENT-ONLY BLIND TEST PREDICTIONS FROZEN; labels never loaded"
    assert manifest["test_labels_loaded"] is False and manifest["fitting_on_test"] is False
    assert manifest["threshold_search_on_test"] is False and float(manifest["threshold"])==0.5
    assert manifest["unimodal_test_metrics_permitted"] is False
    assert fr["selected_multimodal_condition"]=="standard_kd"
    assert fr["test_opened"] is False and fr["no_more_dev_tuning_after_freeze"] is True
    assert manifest["dev_freeze_sha256"]==sha256(freeze/"final_dev_selection.json")
    assert manifest["student_only_predictions_sha256"]==sha256(pred_path)

    pred=pd.read_csv(pred_path).sort_values("participant_id").reset_index(drop=True)
    assert list(pred.columns)==["participant_id","student_probability","student_prediction"]
    assert len(pred)==47 and pred.participant_id.nunique()==47
    assert np.array_equal(
        (pred.student_probability.to_numpy(float)>=.5).astype(int),
        pred.student_prediction.to_numpy(int)
    )
    pred_sha=sha256(pred_path)

    # This is the one and only point where TEST labels are opened.
    gt,gt_path,label_col=read_ground_truth(root,pred.participant_id.tolist())
    d=pred.merge(gt,on="participant_id",how="inner",validate="one_to_one").sort_values("participant_id")
    final=metrics(
        d.label.to_numpy(int),
        d.student_probability.to_numpy(float),
        d.student_prediction.to_numpy(int)
    )

    d.to_csv(out/"final_test_student_predictions_scored.csv",index=False)
    result={
      "protocol_status":"FINAL TEST OPENED ONCE AFTER DEV FREEZE",
      "test_participants":47,
      "ground_truth_file":str(gt_path),
      "ground_truth_label_column":label_col,
      "student_only_blind_predictions_sha256":pred_sha,
      "dev_freeze_sha256":manifest["dev_freeze_sha256"],
      "selected_condition":"standard_kd_student",
      "threshold":0.5,
      "threshold_search":False,
      "fitting_on_test":False,
      "checkpoint_selection_on_test":False,
      "teacher_targets_used_on_test":False,
      "unimodal_branch_test_metrics_computed":False,
      "student_test":final,
      "test_opened":True,
      "no_post_test_model_changes_permitted":True
    }
    (out/"FINAL_TEST_METRICS.json").write_text(json.dumps(result,indent=2)+"\n")
    marker.write_text(json.dumps({
      "test_opened":True,
      "student_only_blind_predictions_sha256":pred_sha,
      "selected_condition":"standard_kd_student",
      "final_metrics_file":"FINAL_TEST_METRICS.json",
      "no_repeat_evaluation":True
    },indent=2)+"\n")

    print(json.dumps(result,indent=2))
    print("\nFINAL STUDENT TEST EVALUATION: COMPLETE")
    print("No further model/threshold/hyperparameter changes are permitted.")

if __name__=="__main__": main()
