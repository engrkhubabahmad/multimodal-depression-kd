"""Fixed-protocol TRAIN-only regularized fusion; report DEV without selecting a winner."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from .common import (cached_or_create, check_splits, complete, fit_scaler,
                     load_embeddings, scores, sha, shift_summary, write_json)

MODES = ("text_linear", "audio_linear", "fusion_linear", "text_residual")


def features(mode, text, audio):
    if mode == "text_linear": return text
    if mode in ("audio_linear", "text_residual"): return audio
    if mode == "fusion_linear": return np.concatenate([text, audio], axis=1)
    raise ValueError(f"Unknown mode: {mode}")


def offset(mode, probability):
    return logit(np.clip(probability, 1e-6, 1 - 1e-6)) if mode == "text_residual" else np.zeros(len(probability))


def fit_head(x, y, base, l2=1.0):
    """Minimize class-balanced mean BCE + l2/2 * ||theta||^2.

    All coefficients, including the intercept/correction bias, are penalized.
    Only TRAIN arrays enter this function. No epoch/DEV selection is performed.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if l2 <= 0 or not np.isfinite(l2): raise ValueError("l2 must be finite and positive")
    if set(y.tolist()) != {0., 1.}: raise ValueError("Both TRAIN classes required")
    mean, scale = fit_scaler(x)
    design = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    weight = np.where(y == 1, .5 / (y == 1).sum(), .5 / (y == 0).sum())

    def objective(theta):
        z = base + design @ theta
        loss = np.sum(weight * (np.logaddexp(0., z) - y * z)) + .5 * l2 * (theta @ theta)
        grad = design.T @ (weight * (expit(z) - y)) + l2 * theta
        return loss, grad

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Head optimization failed: {result.message}")
    return {"mean": mean, "scale": scale, "theta": result.x}, {
        "iterations": int(result.nit), "objective": float(result.fun),
        "gradient_inf_norm": float(np.max(abs(result.jac))), "converged": True}


def predict(head, x, base):
    z = (np.asarray(x) - head["mean"]) / head["scale"]
    return expit(base + z @ head["theta"][:-1] + head["theta"][-1])


def aligned(text, audio):
    for key in ("participant_ids", "labels"):
        if not np.array_equal(text[key], audio[key]):
            raise ValueError(f"Text/audio {key} mismatch")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text-export", type=Path, required=True)
    p.add_argument("--audio-branch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    # Require a completed parity-checked export and verify its content hashes.
    marker = json.loads((a.text_export / "complete.json").read_text())
    cached_or_create(a.text_export, marker["signature"])
    audit = json.loads((a.text_export / "audit.json").read_text())
    if not audit["protocol"]["dev_parity_passed"] or not audit["protocol"]["same_inference_path"]:
        raise ValueError("A verified consistent text export is required")
    text, audio, inputs = {}, {}, {}
    for split, n in (("train", 107), ("dev", 34)):
        for modality, root, target in (("text", a.text_export, text), ("audio", a.audio_branch, audio)):
            path = root / f"{split}_{modality}_embeddings.npz"
            target[split] = load_embeddings(path, n); inputs[str(path)] = sha(path)
        aligned(text[split], audio[split])
    check_splits(text["train"], text["dev"])
    for modality in (text, audio):
        if modality["train"]["embedding"].shape[1] != modality["dev"]["embedding"].shape[1]:
            raise ValueError("TRAIN/DEV feature dimensions differ")
    signature = {"version": 1, "inputs": inputs, "text_export": marker,
                 "l2": 1.0, "class_weight": "balanced", "threshold": .5,
                 "modes": list(MODES), "code": {n: sha(Path(__file__).with_name(n))
                 for n in ("compact_fusion.py", "common.py")}}
    if cached_or_create(a.output, signature):
        print("Verified existing compact fusion run:", a.output); return
    # Finish fitting every preset before computing any DEV metrics.
    heads, optimizers = {}, {}
    for mode in MODES:
        x = features(mode, text["train"]["embedding"], audio["train"]["embedding"])
        heads[mode], optimizers[mode] = fit_head(x, text["train"]["labels"],
                                              offset(mode, text["train"]["probability"]))
        np.savez_compressed(a.output / f"{mode}_head.npz", **heads[mode])
    rows, reports = [], {}
    for mode in ("text_only", *MODES):
        reports[mode] = {}
        for split in ("train", "dev"):
            t, au = text[split], audio[split]
            if mode == "text_only": probability = t["probability"]
            else:
                x = features(mode, t["embedding"], au["embedding"])
                probability = predict(heads[mode], x, offset(mode, t["probability"]))
            report = scores(t["labels"], probability); reports[mode][split] = report
            rows.append({"mode": mode, "split": split, "head_parameters": 0 if mode == "text_only"
                         else int(heads[mode]["theta"].size),
                         **{k: v for k, v in report.items() if not isinstance(v, (dict, list))}})
            pd.DataFrame({"participant_id": t["participant_ids"], "label": t["labels"],
                          "probability": probability, "prediction": (probability >= .5).astype(int)
                          }).to_csv(a.output / f"{mode}_{split}_predictions.csv", index=False)
            pd.DataFrame(report["classification_report"]).T.to_csv(a.output / f"{mode}_{split}_classification_report.csv")
            pd.DataFrame(report["confusion_matrix"], index=["actual_0", "actual_1"],
                         columns=["predicted_0", "predicted_1"]).to_csv(a.output / f"{mode}_{split}_confusion_matrix.csv")
            print(f"\n{mode} | {split}\nCM [0,1]: {report['confusion_matrix']}")
            print(pd.DataFrame(report["classification_report"]).T.round(4).to_string())
    pd.DataFrame(rows).to_csv(a.output / "metrics.csv", index=False)
    write_json(a.output / "audit.json", {"protocol": {"reads_test_data": False, "oof": False,
        "scaler_fit": "TRAIN only", "head_fit": "TRAIN only", "dev_selection": False,
        "upstream_checkpoints_dev_selected": True, "threshold": .5,
        "l2": 1., "intercept_penalized": True, "class_weight": "balanced",
        "note": "Exploratory DEV comparison; upstream branches were already DEV selected. No automatic winner."},
        "optimizers": optimizers, "metrics": reports,
        "embedding_shift": {m: shift_summary(d["train"]["embedding"], d["dev"]["embedding"])
                            for m, d in (("text", text), ("audio", audio))}})
    complete(a.output, signature)
    print("\nSaved compact fusion comparison:", a.output)


if __name__ == "__main__": main()
