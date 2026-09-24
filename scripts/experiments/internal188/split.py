"""Create a 113/37/38 split, preserving the prior untouched internal TEST roster."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""): h.update(part)
    return h.hexdigest()


def table(path, id_name, label_name=None):
    raw = pd.read_csv(path); raw.columns = raw.columns.str.strip().str.lower()
    if id_name not in raw or (label_name and label_name not in raw):
        raise ValueError(f"Missing required columns in {path}: {list(raw.columns)}")
    columns = [id_name] + ([label_name] if label_name else [])
    d = raw[columns].copy(); d.columns = ["participant_id"] + (["label"] if label_name else [])
    if d.isna().any().any() or (d.participant_id != d.participant_id.astype(int)).any():
        raise ValueError(f"Invalid IDs/labels in {path}")
    d = d.astype(int)
    if d.participant_id.duplicated().any() or (label_name and not d.label.isin([0, 1]).all()):
        raise ValueError(f"Duplicate IDs or invalid binary labels in {path}")
    return d


def prepare(root, prior_split_dir, seed=103):
    metadata = Path(root) / "metadata"
    paths = {name: metadata / name for name in (
        "train_split_Depression_AVEC2017.csv", "dev_split_Depression_AVEC2017.csv",
        "test_split_Depression_AVEC2017.csv", "full_test_split.csv")}
    train = table(paths["train_split_Depression_AVEC2017.csv"], "participant_id", "phq8_binary")
    dev_all = table(paths["dev_split_Depression_AVEC2017.csv"], "participant_id", "phq8_binary")
    if 440 not in dev_all.participant_id.to_numpy(): raise ValueError("Expected corrupt participant 440 in official DEV metadata")
    dev = dev_all.loc[dev_all.participant_id.ne(440)].copy()
    test_ids = table(paths["test_split_Depression_AVEC2017.csv"], "participant_id")
    test_labels = table(paths["full_test_split.csv"], "participant_id", "phq_binary")
    if not set(test_ids.participant_id) == set(test_labels.participant_id):
        raise ValueError("full_test_split.csv IDs differ from official TEST roster")
    full = pd.read_csv(paths["full_test_split.csv"])
    full.columns = full.columns.str.strip().str.lower()
    if "phq_score" not in full or not np.array_equal(
        (full.phq_score.to_numpy(float) >= 10).astype(int), full.phq_binary.to_numpy(int)):
        raise ValueError("TEST binary labels disagree with PHQ threshold 10")
    if [len(train), len(dev_all), len(dev), len(test_ids)] != [107, 35, 34, 47]:
        raise ValueError("Unexpected canonical split counts")
    groups = {"canonical_train": train, "canonical_dev": dev, "canonical_test": test_labels}
    for name, data in groups.items():
        if not set(data.label) <= {0, 1}: raise ValueError(f"Invalid labels: {name}")
    ids = [set(d.participant_id) for d in groups.values()]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Canonical participant overlap")
    combined = pd.concat([d.assign(source_split=name) for name, d in groups.items()], ignore_index=True)
    if len(combined) != 188 or 440 in set(combined.participant_id):
        raise ValueError("Expected 188 usable labeled participants excluding 440")
    prior_split_dir = Path(prior_split_dir)
    prior_marker = json.loads((prior_split_dir / 'complete.json').read_text())
    prior_path = prior_split_dir / 'manifest.csv'
    if digest(prior_path) != prior_marker['manifest_sha256']:
        raise ValueError('Prior split manifest changed')
    prior = pd.read_csv(prior_path)
    if prior.split.value_counts().to_dict() != {'train': 150, 'val': 19, 'student_test': 19}:
        raise ValueError('Expected prior 150/19/19 split')
    if (not combined.set_index('participant_id').label.sort_index().equals(
            prior.set_index('participant_id').label.sort_index())):
        raise ValueError('Prior split IDs/labels differ from 188-person pool')
    old_train = combined.loc[combined.participant_id.isin(prior.loc[prior.split.eq('train'), 'participant_id'])]
    old_val = combined.loc[combined.participant_id.isin(prior.loc[prior.split.eq('val'), 'participant_id'])]
    old_test = combined.loc[combined.participant_id.isin(prior.loc[prior.split.eq('student_test'), 'participant_id'])]
    # Previous VAL participants enter TRAIN so the new VAL does not reuse the
    # 19 labels already examined in earlier model-selection experiments.
    extra_test, remain = train_test_split(old_train, train_size=19, random_state=seed,
                                          stratify=old_train.label)
    val, fit_from_old_train = train_test_split(remain, train_size=37, random_state=seed,
                                               stratify=remain.label)
    first = pd.concat([fit_from_old_train, old_val])
    student_test = pd.concat([old_test, extra_test])
    parts = [d.assign(split=name) for name, d in
             (("train", first), ("val", val), ("student_test", student_test))]
    result = pd.concat(parts).sort_values("participant_id").reset_index(drop=True)
    if result.split.value_counts().to_dict() != {"train": 113, "val": 37, "student_test": 38}:
        raise ValueError("Split sizes changed")
    return result, paths, digest(prior_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--prior-split-dir', type=Path, required=True)
    parser.add_argument("--seed", type=int, default=103)
    a = parser.parse_args(argv)
    if a.seed not in (42, 103): raise ValueError("Supported protocol seeds are 42 and 103")
    manifest, paths, prior_sha = prepare(a.daic_root, a.prior_split_dir, a.seed)
    signature = {"seed": a.seed, "split": [113, 37, 38], 'prior_split_sha256': prior_sha,
                 "source_sha256": {name: digest(path) for name, path in paths.items()},
                 "code_sha256": digest(Path(__file__))}
    marker = a.output / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["signature"] != signature or digest(a.output / "manifest.csv") != saved["manifest_sha256"]:
            raise ValueError("Existing split differs. Use a new output directory")
        print("Verified existing 113/37/38 split:", a.output); return
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty directory")
    a.output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(a.output / "manifest.csv", index=False)
    marker.write_text(json.dumps({"signature": signature,
        "manifest_sha256": digest(a.output / "manifest.csv"),
        "class_counts": {name: group.label.value_counts().sort_index().to_dict()
                         for name, group in manifest.groupby("split")},
        "full_test_label_provenance": "Local full_test_split.csv; roster and PHQ threshold verified, external origin not independently verified",
        'prior_internal_test_preserved': True, 'previous_val_moved_to_train': True,
        "warning": "This split was chosen after viewing prior VAL results. The original 47-person TEST roster is redistributed; no official AVEC test claim."}, indent=2) + "\n")
    print(f"Created seed-{a.seed} 113/37/38 manifest:", a.output)


if __name__ == "__main__": main()
