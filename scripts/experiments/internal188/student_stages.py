"""Canonical TRAIN/DEV student stages; never read canonical TEST media."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def prepare(experiment):
    experiment = Path(experiment)
    split = pd.read_csv(experiment / "split" / "manifest.csv")
    coverage = pd.read_csv(experiment / "coverage" / "participant_manifest.csv")
    expected = {"train":107, "val":34, "student_test":47}
    if split.split.value_counts().to_dict() != expected:
        raise ValueError("Expected canonical split TRAIN-107/DEV-34/TEST-47")
    allowed = split[split.split.isin(["train", "val"])][["participant_id", "label", "split"]]
    if len(coverage) != 141 or coverage.participant_id.duplicated().any():
        raise ValueError("Coverage must contain exactly 141 unique TRAIN/DEV participants")
    joined = coverage.merge(allowed, on="participant_id", suffixes=("_cache", ""), validate="one_to_one")
    if len(joined) != 141 or not (joined.label_cache.astype(int) == joined.label.astype(int)).all():
        raise ValueError("Coverage IDs or labels differ from the frozen split")
    if not (joined.split_cache == joined.split).all() or 440 in set(joined.participant_id.astype(int)):
        raise ValueError("Coverage split mismatch or excluded participant 440 present")
    if not joined.feature_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("At least one ComParE16 array is missing")
    out = experiment / "students" / "canonical_v1" / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    result = joined[["participant_id", "label", "split", "feature_path"]].copy()
    result.loc[result.split.eq("val"), "split"] = "dev"
    path = out / "participant_manifest.csv"
    if path.is_file() and not pd.read_csv(path).equals(result):
        raise ValueError(f"Existing student manifest differs: {path}")
    result.to_csv(path, index=False)
    (out / "audit.json").write_text(json.dumps({"train":107,"dev":34,"test_opened":False,
        "source_coverage":str(experiment / "coverage" / "participant_manifest.csv")},indent=2)+"\n")
    return out


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--experiment",required=True,type=Path)
    p.add_argument("--daic-root",required=True,type=Path)
    p.add_argument("--stage",choices=["prepare","text"],required=True)
    p.add_argument("--idiap-source",type=Path)
    a=p.parse_args(argv); e=a.experiment; base=e/"students"/"canonical_v1"
    inputs=prepare(e)
    if a.stage=="prepare": print("Verified student inputs:",inputs); return
    text_dir,audio_dir=base/"plain_text",base/"plain_audio"
    if a.stage=="text":
        if not a.idiap_source or not a.idiap_source.is_dir(): raise FileNotFoundError("Provide --idiap-source")
        from scripts.students.participant_student_v3.pretrain_text import main as train
        train(["--daic-root",str(a.daic_root),"--idiap-source",str(a.idiap_source),"--output",str(text_dir),"--seed","103"])


if __name__=="__main__": main()
