from __future__ import annotations
import argparse,json
from pathlib import Path

def load(p):
    p=Path(p); d=json.loads((p/"metrics.json").read_text())
    assert d["protocol"]["test_opened"] is False,(p,"TEST already opened")
    return d

def key(m):
    return (float(m["macro_f1"]),float(m["depressed_f1"]),float(m["auroc"]))

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--text-branch",required=True)
    ap.add_argument("--no-kd",required=True)
    ap.add_argument("--standard-kd",required=True)
    ap.add_argument("--ra-kd",required=True)
    ap.add_argument("--output",required=True)
    a=ap.parse_args(argv)

    text=load(a.text_branch)
    nokd=load(a.no_kd)
    std=load(a.standard_kd)
    ra=load(a.ra_kd)

    # Controlled multimodal comparison invariants.
    for d in [nokd,std,ra]:
        p=d["protocol"]
        assert p["train_participants"]==107 and p["dev_participants"]==34
        assert p["participant_440_excluded"] is True
        assert p["threshold"]==0.5 and p["threshold_search"] is False

    sp=std["protocol"]; rp=ra["protocol"]
    assert sp["seed"]==rp["seed"]==103
    assert sp["initial_state_sha256"]==rp["initial_state_sha256"]
    assert sp["fusion_params"]==rp["fusion_params"]==55617
    assert sp["full_student_params"]==rp["full_student_params"]==254274
    assert float(sp["temperature"])==float(rp["temperature"])==2.0
    assert float(sp["kd_weight"])==float(rp["kd_weight"])==0.5
    assert sp["same_frozen_branch_embeddings"] is True and rp["same_frozen_branch_embeddings"] is True
    assert sp["same_train_only_standardizers"] is True and rp["same_train_only_standardizers"] is True
    assert sp["same_fusion_architecture"] is True and rp["same_fusion_architecture"] is True
    assert sp["same_optimizer_hparams"] is True and rp["same_optimizer_hparams"] is True

    variants={
        "no_kd":nokd["dev34"],
        "standard_kd":std["dev34"],
        "ra_kd":ra["dev34"],
    }
    selected=max(variants,key=lambda n:key(variants[n]))

    # This should be Standard KD for the frozen DEV-34 results.
    assert selected=="standard_kd",(selected,variants)

    result={
      "protocol_status":"DEV stage frozen; TEST unopened",
      "train_participants":107,
      "dev_participants":34,
      "participant_440_excluded":True,
      "selection_rule":"lexicographic DEV-34 (macro_f1, depressed_f1, auroc), fixed threshold 0.5",
      "multimodal_variants":variants,
      "selected_multimodal_condition":selected,
      "selected_multimodal_dev34":variants[selected],
      "selected_checkpoint_dir":str(Path(a.standard_kd)),
      "selected_checkpoint_file":str(Path(a.standard_kd)/"best.pt"),
      "initial_state_sha256":sp["initial_state_sha256"],
      "fusion_params":sp["fusion_params"],
      "full_student_params":sp["full_student_params"],
      "temperature":sp["temperature"],
      "kd_weight":sp["kd_weight"],
      "text_only_dev34":text["dev34"],
      "text_only_is_stronger_on_dev_macro_f1":float(text["dev34"]["macro_f1"])>float(variants[selected]["macro_f1"]),
      "ra_kd_interpretation":"negative/tied ablation on macro-F1 and depressed-F1; lower AUROC than Standard KD",
      "test_opened":False,
      "no_more_dev_tuning_after_freeze":True
    }

    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/"final_dev_selection.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    print("\nDEV FREEZE: PASS")
    print("Selected multimodal student: STANDARD KD")
    print("TEST remains CLOSED.")

if __name__=="__main__": main()
