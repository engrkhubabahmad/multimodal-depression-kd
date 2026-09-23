from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cached_or_create(out, signature):
    """Resume verified completed work; never silently replace an old run."""
    out = Path(out); marker = out / "complete.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["signature"] != signature:
            raise ValueError("Inputs or settings changed. Use a new output directory.")
        for name, digest in old["outputs"].items():
            if not (out / name).is_file() or sha(out / name) != digest:
                raise ValueError(f"Cached output changed or missing: {name}")
        return True
    if out.exists() and any(out.iterdir()):
        raise ValueError("Incomplete/nonempty output directory. Use a new output directory.")
    out.mkdir(parents=True, exist_ok=True)
    return False


def complete(out, signature):
    out = Path(out)
    write_json(out / "complete.json", {"signature": signature, "outputs": {
        p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "complete.json"
    }})


def load_embeddings(path, expected_n=None):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in ("participant_ids", "labels", "embedding", "probability")}
    ids, y, e, p = (d[k] for k in ("participant_ids", "labels", "embedding", "probability"))
    if ids.ndim != 1 or y.shape != ids.shape or p.shape != ids.shape or e.ndim != 2 or len(e) != len(ids):
        raise ValueError(f"Invalid embedding shapes: {path}")
    if not np.isfinite(ids).all() or not np.equal(ids, ids.astype(int)).all():
        raise ValueError("Invalid participant IDs")
    if len(set(ids.tolist())) != len(ids) or (expected_n is not None and len(ids) != expected_n):
        raise ValueError(f"Participant count/uniqueness mismatch: {path}")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(e).all() or not np.isfinite(p).all():
        raise ValueError(f"Invalid labels/features/probabilities: {path}")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    order = np.argsort(ids)
    return {k: v[order] for k, v in d.items()}


def check_splits(train, dev):
    if set(train["participant_ids"]) & set(dev["participant_ids"]):
        raise ValueError("TRAIN/DEV participant overlap")
    if 440 in dev["participant_ids"]:
        raise ValueError("DEV must exclude participant 440")
    for name, d, counts in [("TRAIN", train, [77, 30]), ("DEV", dev, [23, 11])]:
        if np.bincount(d["labels"].astype(int), minlength=2).tolist() != counts:
            raise ValueError(f"Unexpected {name} class counts")


def scores(y, p):
    pred = (p >= .5).astype(int)
    return {"n": len(y), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "depressed_f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "predicted_positive": int(pred.sum()),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y, pred, labels=[0, 1],
                target_names=["Non-depressed", "Depressed"], output_dict=True, zero_division=0)}


def fit_scaler(x):
    mean = x.mean(axis=0); std = x.std(axis=0)
    return mean, np.where(std < 1e-6, 1., std)


def shift_summary(train, dev):
    mean, std = fit_scaler(train)
    z = (dev - mean) / std
    return {"train_norm_median": float(np.median(np.linalg.norm(train, axis=1))),
            "dev_norm_median": float(np.median(np.linalg.norm(dev, axis=1))),
            "dev_abs_z_p95": float(np.quantile(abs(z), .95)),
            "dev_fraction_beyond_3_train_sd": float((abs(z) > 3).mean())}
