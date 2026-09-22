from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scripts.teachers.ussd_audio.common import sha256

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--blind-output",required=True)
    ap.add_argument("--dev-freeze",required=True)
    a=ap.parse_args(argv)
    out,freeze=Path(a.blind_output),Path(a.dev_freeze)

    if (out/"FINAL_TEST_SCORED.json").exists() or (out/"FINAL_TEST_METRICS.json").exists():
        raise RuntimeError("TEST has already been scored; refusing to alter blind predictions.")

    src=out/"blind_test_predictions.csv"
    man=out/"blind_test_manifest.json"
    if not src.exists() or not man.exists():
        raise FileNotFoundError("Missing blind TEST predictions/manifest.")

    m=json.loads(man.read_text())
    fr=json.loads((freeze/"final_dev_selection.json").read_text())
    assert m["status"]=="BLIND TEST PREDICTIONS FROZEN; labels never loaded"
    assert m["test_labels_loaded"] is False
    assert m["fitting_on_test"] is False
    assert m["threshold_search_on_test"] is False
    assert float(m["selected_threshold"])==0.5
    assert fr["selected_multimodal_condition"]=="standard_kd"
    assert fr["test_opened"] is False and fr["no_more_dev_tuning_after_freeze"] is True
    assert m["dev_freeze_sha256"]==sha256(freeze/"final_dev_selection.json")

    d=pd.read_csv(src)
    assert len(d)==47 and d.participant_id.nunique()==47
    assert "label" not in {c.casefold() for c in d.columns}

    if {"student_probability","student_prediction"}<=set(d.columns):
        prob=d["student_probability"].to_numpy(float)
        pred=d["student_prediction"].to_numpy(int)
        source_schema="student_only"
    elif {"standard_kd_probability","standard_kd_prediction"}<=set(d.columns):
        prob=d["standard_kd_probability"].to_numpy(float)
        pred=d["standard_kd_prediction"].to_numpy(int)
        source_schema="legacy_standard_kd_with_diagnostics"
    else:
        raise AssertionError(f"Could not find selected student columns in {src}")

    assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
    assert np.array_equal((prob>=.5).astype(int),pred),"Stored prediction != fixed threshold 0.5"

    clean=pd.DataFrame({
        "participant_id":d["participant_id"].astype(int),
        "student_probability":prob,
        "student_prediction":pred
    }).sort_values("participant_id").reset_index(drop=True)

    dst=out/"blind_student_only_predictions.csv"
    clean.to_csv(dst,index=False)

    audit={
      "status":"STUDENT-ONLY BLIND TEST PREDICTIONS FROZEN; labels never loaded",
      "test_participants":47,
      "selected_condition":"standard_kd_student",
      "threshold":0.5,
      "source_schema":source_schema,
      "source_blind_predictions_sha256":sha256(src),
      "student_only_predictions_sha256":sha256(dst),
      "source_manifest_sha256":sha256(man),
      "dev_freeze_sha256":sha256(freeze/"final_dev_selection.json"),
      "selected_checkpoint_sha256":m["selected_checkpoint_sha256"],
      "audio_branch_checkpoint_sha256":m["audio_branch_checkpoint_sha256"],
      "text_inference_state_sha256":m["text_inference_state_sha256"],
      "train_standardizers_sha256":m["train_standardizers_sha256"],
      "teacher_targets_used_on_test":False,
      "test_labels_loaded":False,
      "fitting_on_test":False,
      "threshold_search_on_test":False,
      "unimodal_test_metrics_permitted":False
    }
    (out/"blind_student_only_manifest.json").write_text(json.dumps(audit,indent=2)+"\n")

    print(json.dumps(audit,indent=2))
    print("\nSTUDENT-ONLY BLIND FREEZE: PASS")
    print("No TEST labels were loaded.")
    print("Original blind file preserved unchanged.")

if __name__=="__main__": main()
