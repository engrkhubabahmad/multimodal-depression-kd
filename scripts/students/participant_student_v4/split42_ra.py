"""Seed-42 split student: fresh TRAIN-only proxy targets, KD, and corruption audits."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from .common import cached_or_create, complete, fit_scaler, scores, sha, write_json
from .compact_fusion import predict
from .export_text import document, transcript_paths
from .split42_baseline import audio_paths, audio_summary


def teacher_targets(text_head, audio_head, text, audio, ids, labels, temperature=2.):
    tp = predict(text_head, text, np.zeros(len(ids)))
    ap = predict(audio_head, audio, np.zeros(len(ids)))
    tz = logit(np.clip(tp, 1e-8, 1 - 1e-8))
    az = logit(np.clip(ap, 1e-8, 1 - 1e-8))
    st = float(np.median(abs(tz))); sa = float(np.median(abs(az)))
    if min(st, sa) < 1e-10: raise ValueError("Degenerate teacher logit scale")
    ut, ua = tz / st, az / sa
    qt, qa = expit(ut / temperature), expit(ua / temperature)
    ct, ca = -np.expm1(-abs(ut)), -np.expm1(-abs(ua))
    wt = np.divide(ct, ct + ca, out=np.full_like(ct, .5), where=(ct + ca) > 1e-12)
    wa = 1 - wt
    frame = pd.DataFrame({"participant_id": ids, "label": labels,
                          "text_logit": tz, "text_probability": tp,
                          "audio_logit": az, "audio_probability": ap,
                          "text_reliability": ct, "audio_reliability": ca,
                          "text_weight": wt, "audio_weight": wa,
                          "standard_soft_target": (qt + qa) / 2,
                          "ra_soft_target": wt * qt + wa * qa})
    return frame, {"text_median_abs_train_logit": st,
                   "audio_median_abs_train_logit": sa, "temperature": temperature}


def fit_student(x, y, soft=None, temperature=2., kd_weight=.5, l2=1.):
    """Fixed convex objective: balanced hard BCE + T²-scaled soft BCE + L2."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mean, scale = fit_scaler(x)
    d = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    hard_weight = np.where(y == 1, .5 / sum(y == 1), .5 / sum(y == 0))
    if soft is not None:
        soft = np.asarray(soft, float)
        if soft.shape != y.shape or not np.isfinite(soft).all() or ((soft <= 0) | (soft >= 1)).any():
            raise ValueError("Invalid teacher TRAIN soft targets")
    def objective(theta):
        z = d @ theta; hard = np.sum(hard_weight * (np.logaddexp(0, z) - y * z))
        grad = (1 - (kd_weight if soft is not None else 0)) * (d.T @ (hard_weight * (expit(z) - y)))
        loss = (1 - (kd_weight if soft is not None else 0)) * hard
        if soft is not None:
            zt = z / temperature
            loss += kd_weight * temperature ** 2 * np.mean(np.logaddexp(0, zt) - soft * zt)
            grad += kd_weight * temperature * d.T @ (expit(zt) - soft) / len(y)
        return loss + .5 * l2 * (theta @ theta), grad + l2 * theta
    result = minimize(objective, np.zeros(d.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12})
    if not result.success: raise RuntimeError(result.message)
    return {"mean": mean, "scale": scale, "theta": result.x}


def perturb(x, text_dim, scenario, severity, rng):
    """Missing = TRAIN-mean replacement; noise SD in TRAIN-standardized units."""
    z = x.copy()
    if scenario in ("missing_text", "missing_both"): z[:, :text_dim] = 0
    if scenario in ("missing_audio", "missing_both"): z[:, text_dim:] = 0
    if scenario in ("noise_text", "noise_both"):
        z[:, :text_dim] += rng.normal(0, severity, z[:, :text_dim].shape)
    if scenario in ("noise_audio", "noise_both"):
        z[:, text_dim:] += rng.normal(0, severity, z[:, text_dim:].shape)
    return z


def run(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--split-baseline", type=Path, required=True)
    p.add_argument("--local-audio-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    baseline = a.split_baseline
    marker = json.loads((baseline / "complete.json").read_text())
    cached_or_create(baseline, marker["signature"])
    audit = json.loads((baseline / "audit.json").read_text())["protocol"]
    if audit["seed"] != 42 or audit["internal_holdout_scored"] or audit["previous_checkpoints_used"]:
        raise ValueError("Expected the unscored fresh seed-42 baseline")
    split = pd.read_csv(baseline / "split_manifest.csv")
    if split.experimental_split.value_counts().to_dict() != {"train": 99, "dev": 21, "holdout": 21}:
        raise ValueError("Unexpected split manifest")
    pathmap = transcript_paths(a.daic_root, split.participant_id)
    amap = audio_paths(a.features, split, a.local_audio_root)
    source = {str(baseline / name): sha(baseline / name) for name in
              ("complete.json", "split_manifest.csv", "train_vectorizer.pkl",
               "text_head.npz", "audio_head.npz")}
    for path in (*pathmap.values(), *amap.values()): source[str(path)] = sha(path)
    signature = {"version": 1, "method": "seed42_proxy_teacher_ra_student",
                 "inputs": source, "temperature": 2., "kd_weight": .5,
                 "l2": 1., "noise_sd": [0.5, 1.], "repeats": 20,
                 "code": {n: sha(Path(__file__).with_name(n)) for n in
                          ("split42_ra.py", "split42_baseline.py", "compact_fusion.py", "common.py")}}
    if cached_or_create(a.output, signature):
        print("Verified existing seed-42 RA student run:", a.output); return
    with (baseline / "train_vectorizer.pkl").open("rb") as f: v = pickle.load(f)
    with np.load(baseline / "text_head.npz") as z: th = {key: z[key] for key in z.files}
    with np.load(baseline / "audio_head.npz") as z: ah = {key: z[key] for key in z.files}
    docs = {int(pid): document(pathmap[int(pid)]) for pid in split.participant_id}
    vectors = {int(pid): audio_summary(amap[int(pid)]) for pid in split.participant_id}
    def data(name):
        frame = split.loc[split.experimental_split.eq(name)].sort_values("participant_id")
        ids = frame.participant_id.to_numpy(int)
        return ids, frame.label.to_numpy(int), v.transform([docs[int(i)] for i in ids]).toarray(), np.stack([vectors[int(i)] for i in ids])
    tid, ty, tt, ta = data("train")
    did, dy, dt, da = data("dev")
    targets, scales = teacher_targets(th, ah, tt, ta, tid, ty)
    targets.to_csv(a.output / "train_teacher_logits_probabilities.csv", index=False)
    if set(tid) & set(did) or len(tid) != 99 or len(did) != 21: raise ValueError("TRAIN/DEV overlap")
    tx = np.column_stack([tt, ta]); dx = np.column_stack([dt, da]); text_dim = tt.shape[1]
    heads = {}; reports = {}; rows = []
    for mode, soft in (("no_kd", None), ("standard_kd", targets.standard_soft_target),
                       ("ra_kd", targets.ra_soft_target)):
        head = fit_student(tx, ty, soft)
        heads[mode] = head; np.savez_compressed(a.output / f"{mode}_student.npz", **head)
    for mode, head in heads.items():
        reports[mode] = {}
        for name, ids, y, x in (("train", tid, ty, tx), ("dev", did, dy, dx)):
            probability = predict(head, x, np.zeros(len(x)))
            m = scores(y, probability); reports[mode][name] = m
            rows.append({"mode": mode, "split": name, "parameters": len(head["theta"]),
                         **{k: val for k, val in m.items() if not isinstance(val, (dict, list))}})
            pd.DataFrame({"participant_id": ids, "label": y, "probability": probability,
                          "prediction": (probability >= .5).astype(int)}).to_csv(
                          a.output / f"{mode}_{name}_predictions.csv", index=False)
            print(f"{mode} {name}: macroF1={m['macro_f1']:.4f} depressedF1={m['depressed_f1']:.4f} CM={m['confusion_matrix']}")
    pd.DataFrame(rows).to_csv(a.output / "metrics.csv", index=False)
    corruption = []
    for mode, head in heads.items():
        standard = (dx - head["mean"]) / head["scale"]
        for scenario in ("missing_text", "missing_audio", "missing_both",
                         "noise_text", "noise_audio", "noise_both"):
            is_noise = scenario.startswith("noise")
            for severity in ((.5, 1.) if is_noise else (0.,)):
                for repeat in range(20 if is_noise else 1):
                    # The same seed/scenario/repeat is applied to all modes.
                    rng = np.random.default_rng(np.random.SeedSequence([42,
                        ("missing_text", "missing_audio", "missing_both", "noise_text", "noise_audio", "noise_both").index(scenario),
                        int(severity * 10), repeat]))
                    z = perturb(standard, text_dim, scenario, severity, rng)
                    probability = expit(z @ head["theta"][:-1] + head["theta"][-1])
                    m = scores(dy, probability)
                    corruption.append({"mode": mode, "scenario": scenario, "noise_sd": severity,
                                       "repeat": repeat, "macro_f1": m["macro_f1"],
                                       "depressed_f1": m["depressed_f1"], "auroc": m["auroc"],
                                       "confusion_matrix": json.dumps(m["confusion_matrix"])})
    pd.DataFrame(corruption).to_csv(a.output / "dev_missing_noise.csv", index=False)
    write_json(a.output / "audit.json", {"protocol": {"split": "seed42_99_21_21",
        "official_test_used": False, "internal_student_test_scored": False,
        "teacher_targets": "in-sample TRAIN-99 text/audio shallow proxy heads, not published teachers",
        "teacher_target_participants": 99, "oof": False,
        "student_fit": "TRAIN-99 only", "student_seed": 42,
        "temperature": 2., "kd_weight": .5, "l2": 1., "threshold": .5,
        "missing": "replace modality with its TRAIN mean", "noise": "Gaussian in TRAIN-standardized units; 20 repeats per severity",
        "dev_use": "exploratory comparison; do not tune on the internal student test"},
        "teacher_scales": scales, "metrics": reports})
    complete(a.output, signature)
    print("Saved split-aware proxy teacher targets and RA robustness diagnostics:", a.output)


if __name__ == "__main__": run()
