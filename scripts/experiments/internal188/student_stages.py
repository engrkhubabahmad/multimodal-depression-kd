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
    p.add_argument("--stage",choices=["prepare","text","audio","no_kd","ra_kd"],required=True)
    p.add_argument("--idiap-source",type=Path); p.add_argument("--ussd-source",type=Path)
    a=p.parse_args(argv); e=a.experiment; base=e/"students"/"canonical_v1"
    inputs=prepare(e)
    if a.stage=="prepare": print("Verified student inputs:",inputs); return
    text_dir,audio_dir=base/"plain_text",base/"plain_audio"
    if a.stage=="text":
        if not a.idiap_source or not a.idiap_source.is_dir(): raise FileNotFoundError("Provide --idiap-source")
        from scripts.students.participant_student_v3.pretrain_text import main as train
        train(["--daic-root",str(a.daic_root),"--idiap-source",str(a.idiap_source),"--output",str(text_dir),"--seed","103"])
    elif a.stage=="audio":
        if not a.ussd_source or not a.ussd_source.is_dir(): raise FileNotFoundError("Provide --ussd-source for normalization only")
        from scripts.students.participant_student_v3.pretrain_audio import main as train
        train(["--features",str(inputs),"--author-root",str(a.ussd_source),"--output",str(audio_dir)])
    elif a.stage=="no_kd":
        for directory, file in ((text_dir,"train_text_embeddings.npz"),(audio_dir,"train_audio_embeddings.npz")):
            if not (directory/file).is_file(): raise FileNotFoundError(directory/file)
        from scripts.students.participant_student_v3.train_fusion_no_kd import main as train
        train(["--text-branch",str(text_dir),"--audio-branch",str(audio_dir),
               "--output",str(base/"no_kd"),"--seed","103"])
    else:
        raise RuntimeError("RA-KD is blocked: NUSD run-2 TRAIN-107 targets have not been exported and verified. Existing v3 KD code expects USSD run-4 targets; never substitute them.")


if __name__=="__main__": main()
