"""Fresh participant-level 70/15/15 seed-42 sanity baseline on labeled DAIC TRAIN+DEV."""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from .common import cached_or_create, complete, scores, sha, write_json
from .compact_fusion import fit_head, predict
from .export_text import document, split_table, transcript_paths


def split_70_15_15(table, seed=42):
    """99/21/21 from 141 labeled participants, stratified in two seeded stages."""
    if len(table) != 141 or table.participant_id.duplicated().any():
        raise ValueError("Expected 141 distinct labeled participants (107 TRAIN, 34 DEV)")
    train, rest = train_test_split(table, train_size=99, random_state=seed,
                                   stratify=table.label)
    dev, holdout = train_test_split(rest, train_size=21, random_state=seed,
                                    stratify=rest.label)
    parts = []
    for name, frame in (("train", train), ("dev", dev), ("holdout", holdout)):
        parts.append(frame.assign(experimental_split=name))
    result = pd.concat(parts).sort_values("participant_id").reset_index(drop=True)
    if len(result) != 141 or result.participant_id.nunique() != 141:
        raise ValueError("Split lost or duplicated participants")
    return result


def audio_summary(path):
    # One participant at a time, with the feature source read through mmap.
    x = np.load(path, mmap_mode="r", allow_pickle=False)
    if x.ndim != 2 or x.shape[0] != 130 or x.shape[1] == 0:
        raise ValueError(f"Invalid ComParE16 feature shape: {path}")
    # One 130-feature frame block at a time to limit peak RAM.
    n = 0; total = np.zeros(130); total2 = np.zeros(130)
    for start in range(0, x.shape[1], 4096):
        block = np.asarray(x[:, start:start + 4096], dtype=np.float64)
        if not np.isfinite(block).all(): raise ValueError(f"Non-finite audio features: {path}")
        n += block.shape[1]; total += block.sum(1); total2 += np.square(block).sum(1)
    mean = total / n
    return np.concatenate([mean, np.sqrt(np.maximum(total2 / n - mean ** 2, 0.))])


def audio_paths(features, table, local=None):
    manifest = pd.read_csv(features / "participant_manifest.csv")
    if not {"participant_id", "label", "feature_path", "split"} <= set(manifest.columns):
        raise ValueError("Incomplete ComParE16 feature manifest")
    if not set(manifest.split.astype(str).str.lower()) <= {"train", "dev"}:
        raise ValueError("Feature manifest contains a forbidden source split")
    if manifest.participant_id.duplicated().any() or set(manifest.participant_id) != set(table.participant_id):
        raise ValueError("Audio feature coverage differs from 141 labeled participants")
    indexed = manifest.set_index("participant_id"); found = {}
    for pid, label in table[["participant_id", "label"]].itertuples(index=False, name=None):
        row = indexed.loc[pid]
        if int(row.label) != int(label): raise ValueError(f"Audio label mismatch: {pid}")
        path = Path(row.feature_path)
        if local:
            candidate = Path(local) / str(row.split).lower() / f"{int(pid)}.npy"
            if candidate.is_file(): path = candidate
        if not path.is_file(): raise FileNotFoundError(path)
        found[pid] = path
    return found


def run(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--local-audio-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    if a.seed != 42: raise ValueError("This fixed protocol uses seed 42; new seeds need a new protocol")
    orig, sources = [], {}
    for split, size in (("train", 107), ("dev", 34)):
        d, path = split_table(a.daic_root, split)
        if len(d) != size: raise ValueError(f"Unexpected canonical {split} size")
        orig.append(d.assign(source_split=split)); sources[str(path)] = sha(path)
    all_rows = pd.concat(orig).sort_values("participant_id").reset_index(drop=True)
    if all_rows.participant_id.duplicated().any() or set(all_rows.label) != {0, 1}:
        raise ValueError("Invalid labeled participant pool")
    split = split_70_15_15(all_rows, a.seed)
    paths = transcript_paths(a.daic_root, all_rows.participant_id)
    audio = audio_paths(a.features, all_rows, a.local_audio_root)
    sources[str(a.features / "participant_manifest.csv")] = sha(a.features / "participant_manifest.csv")
    for path in (*paths.values(), *audio.values()): sources[str(path)] = sha(path)
    signature = {"version": 1, "method": "fresh_70_15_15_sanity_baseline",
                 "seed": a.seed, "inputs": sources,
                 "code": {name: sha(Path(__file__).with_name(name)) for name in
                          ("split42_baseline.py", "common.py", "compact_fusion.py", "export_text.py")}}
    if cached_or_create(a.output, signature):
        print("Verified completed seed-42 run:", a.output); return
    split.to_csv(a.output / "split_manifest.csv", index=False)
    train = split.loc[split.experimental_split.eq("train")]
    dev = split.loc[split.experimental_split.eq("dev")]
    # All learned text statistics, scaling and coefficients use TRAIN only.
    docs = {int(pid): document(paths[int(pid)]) for pid in split.participant_id}
    v = TfidfVectorizer(max_features=250, sublinear_tf=True, ngram_range=(1, 1))
    v.fit([docs[int(pid)] for pid in train.participant_id])
    with (a.output / "train_vectorizer.pkl").open("wb") as f: pickle.dump(v, f)
    print("Summarizing 141 audio feature files one participant at a time")
    audio_vectors = {int(pid): audio_summary(audio[int(pid)]) for pid in split.participant_id}
    inputs = {}
    for name, frame in (("train", train), ("dev", dev)):
        ids = frame.participant_id.to_numpy(int)
        inputs[name] = {"ids": ids, "labels": frame.label.to_numpy(int),
                        "text": v.transform([docs[int(pid)] for pid in ids]).toarray(),
                        "audio": np.stack([audio_vectors[int(pid)] for pid in ids])}
    modes = ("text", "audio", "fusion")
    heads = {}; rows = []; report = {}
    for mode in modes:
        tr = inputs["train"]
        x = tr[mode] if mode != "fusion" else np.column_stack([tr["text"], tr["audio"]])
        head, diag = fit_head(x, tr["labels"], np.zeros(len(tr["labels"])), l2=1.)
        heads[mode] = head
        np.savez_compressed(a.output / f"{mode}_head.npz", **head)
        report[mode] = {"fit": diag}
    # DEV inspection happens after all heads are fixed. Holdout is not evaluated here.
    for mode in modes:
        for name in ("train", "dev"):
            data = inputs[name]
            x = data[mode] if mode != "fusion" else np.column_stack([data["text"], data["audio"]])
            prob = predict(heads[mode], x, np.zeros(len(data["labels"])))
            m = scores(data["labels"], prob); report[mode][name] = m
            rows.append({"mode": mode, "split": name, "parameters": len(heads[mode]["theta"]),
                         **{k: val for k, val in m.items() if not isinstance(val, (dict, list))}})
            pd.DataFrame({"participant_id": data["ids"], "label": data["labels"],
                          "probability": prob, "prediction": (prob >= .5).astype(int)
                          }).to_csv(a.output / f"{mode}_{name}_predictions.csv", index=False)
            print(f"{mode} {name}: macroF1={m['macro_f1']:.4f} depressedF1={m['depressed_f1']:.4f} CM={m['confusion_matrix']}")
    pd.DataFrame(rows).to_csv(a.output / "metrics.csv", index=False)
    write_json(a.output / "audit.json", {"protocol": {
        "seed": 42, "pool": "canonical TRAIN-107 + DEV-34; participant 440 excluded",
        "counts": split.experimental_split.value_counts().to_dict(),
        "ratio": "99/21/21 of 141 labeled participants (approximately 70/15/15)",
        "test_source_used": False, "internal_holdout_scored": False, "oof": False,
        "previous_checkpoints_used": False, "train_only_vocabulary_and_scaler": True,
        "precomputed_audio_features": "ComParE16, label-independent, participant mean/std",
        "upstream_feature_extraction_must_be_label_free": True,
        "l2": 1., "threshold": .5,
        "caution": "Exploratory split; historical work inspected canonical DEV. This is not an independent publication TEST score or KD."},
        "metrics": report})
    complete(a.output, signature)
    print("Completed seed-42 fresh split baseline; internal holdout unscored:", a.output)


if __name__ == "__main__": run()
