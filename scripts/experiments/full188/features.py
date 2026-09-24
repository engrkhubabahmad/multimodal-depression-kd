"""Reuse canonical ComParE16 arrays; extract missing participants for full-188."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .split import digest


def verified_split(path):
    root = Path(path); marker = json.loads((root / "complete.json").read_text())
    if digest(root / "manifest.csv") != marker["manifest_sha256"]:
        raise ValueError("Split manifest changed")
    data = pd.read_csv(root / "manifest.csv")
    if data.split.value_counts().to_dict() != {"train": 150, "val": 19, "student_test": 19}:
        raise ValueError("Unexpected full-188 split")
    return data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--canonical-features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare16-config", type=Path, required=True)
    p.add_argument("--smile-extract", default="SMILExtract")
    a = p.parse_args(argv)
    manifest = verified_split(a.split_dir)
    old_path = a.canonical_features / "participant_manifest.csv"
    old = pd.read_csv(old_path)
    if not {"participant_id", "label", "feature_path"} <= set(old.columns):
        raise ValueError("Canonical feature manifest lacks IDs, labels, or paths")
    if old.participant_id.duplicated().any() or len(old) not in (141, 188) or 440 in set(old.participant_id):
        raise ValueError("Expected 141 canonical or 188 full-protocol cached participants")
    indexed = old.set_index("participant_id"); reused = []; missing = []; copied = 0
    for r in manifest.itertuples(index=False):
        if r.participant_id in indexed.index:
            row = indexed.loc[r.participant_id]
            if int(row.label) != int(r.label): raise ValueError(f"Cached label mismatch: {r.participant_id}")
            path = Path(row.feature_path)
            if not path.is_file(): raise FileNotFoundError(path)
            x = np.load(path, mmap_mode="r", allow_pickle=False)
            if x.ndim != 2 or x.shape[0] != 130 or x.shape[1] == 0:
                raise ValueError(f"Invalid cached ComParE16 feature: {path}")
            if len(old) == 188 and path.resolve(strict=True).is_relative_to(a.canonical_features.resolve()):
                dst = a.output / "cached_original_test" / path.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists(): shutil.copy2(path, dst)
                if digest(dst) != digest(path): raise ValueError(f"Cached audio copy differs: {dst}")
                path = dst
                copied += 1
            reused.append({"participant_id": int(r.participant_id), "label": int(r.label),
                           "split": r.split, "feature_path": str(path), "frames": int(x.shape[1]),
                           "provenance": "canonical label-independent ComParE16 cache"})
        else:
            missing.append({"participant_id": int(r.participant_id), "label": int(r.label),
                            "split": r.split})
    if len(missing) != 188 - len(old):
        raise ValueError(f"Unexpected missing feature count: {len(missing)}")
    if missing:
        if not a.compare16_config.is_file(): raise FileNotFoundError(a.compare16_config)
        config_sha256 = digest(a.compare16_config)
    else:
        # All arrays already exist. No OpenSMILE executable or config is needed
        # for copying and verifying the cache from the previous experiment.
        previous = a.canonical_features / "provenance.json"
        if not previous.is_file(): raise FileNotFoundError(previous)
        config_sha256 = json.loads(previous.read_text())["compare16_config_sha256"]
    a.output.mkdir(parents=True, exist_ok=True)
    fresh = []
    if missing:
        from scripts.teachers.ussd_audio.prepare_compare16 import process_split, resolve_executable
        smile = resolve_executable(a.smile_extract)
        # Extract exactly the missing raw recordings, one participant at a time.
        fresh = process_split(a.daic_root, pd.DataFrame(missing)[["participant_id", "label"]],
                              "new_original_test", a.output, smile, a.compare16_config, False)
    split_by_id = {r["participant_id"]: r["split"] for r in missing}
    rows = reused + [{"participant_id": r["participant_id"], "label": r["label"],
                       "split": split_by_id[r["participant_id"]], "feature_path": r["feature_path"],
                       "frames": r["frames"], "provenance": "fresh original TEST ComParE16 extraction"}
                      for r in fresh]
    d = pd.DataFrame(rows).sort_values("participant_id")
    if len(d) != 188 or d.participant_id.nunique() != 188 or not np.array_equal(
            d.participant_id.to_numpy(), manifest.participant_id.to_numpy()):
        raise ValueError("Feature manifest does not cover exact 188-person split")
    if not np.array_equal(d.label.to_numpy(), manifest.label.to_numpy()):
        raise ValueError("Feature labels differ from split")
    d.to_csv(a.output / "participant_manifest.csv", index=False)
    (a.output / "provenance.json").write_text(json.dumps({
        "split_manifest_sha256": digest(a.split_dir / "manifest.csv"),
        "canonical_feature_manifest_sha256": digest(old_path),
        "compare16_config_sha256": config_sha256,
        "reuse_count": len(reused), "fresh_count": len(fresh), "copied_to_independent_cache": copied,
        "note": "Feature extraction is label-independent. No learned normalization reused; trainers fit it on TRAIN-132."}, indent=2) + "\n")
    print("Verified full-188 features: reused", len(reused), "extracted", len(fresh))


if __name__ == "__main__": main()
