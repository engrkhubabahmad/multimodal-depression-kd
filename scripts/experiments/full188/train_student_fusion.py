"""Matched no-KD, standard KD and RA-KD fusion on independent student branches."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit
from scripts.students.participant_student_v3.fusion_model import FrozenBranchFusion
from .features import verified_split
from .split import digest
from .train_text_teacher import metric


def load(branch, name, modality, expected):
    with np.load(branch / f"{name}_{modality}_embeddings.npz", allow_pickle=False) as z:
        ids = z["participant_ids"].astype(int)
        labels = z["labels"].astype(int)
        embedding = z["embedding"].astype(np.float32)
    if not np.array_equal(ids, expected.participant_id.to_numpy(int)) or not np.array_equal(labels, expected.label.to_numpy(int)):
        raise ValueError(f"Student {name} {modality} embedding IDs/labels differ from frozen split")
    return embedding


def targets(path, expected, modality):
    d = pd.read_csv(path).sort_values("participant_id")
    if len(d) != len(expected) or not np.array_equal(d.participant_id, expected.participant_id) or not np.array_equal(d.label, expected.label):
        raise ValueError(f"Teacher {modality} TRAIN targets differ from student TRAIN")
    p = d[f"{modality}_probability"].to_numpy(float)
    z = d[f"{modality}_logit"].to_numpy(float)
    if not np.isfinite(z).all() or not np.isfinite(p).all() or not np.all((p > 0) & (p < 1)):
        raise ValueError("Invalid teacher targets")
    if not np.allclose(expit(z), p, atol=1e-6, rtol=0):
        raise ValueError(f"Teacher {modality} logit/probability mismatch")
    return z


def distillation_targets(text_z, audio_z, temperature=2.):
    st = float(np.median(abs(text_z))); sa = float(np.median(abs(audio_z)))
    if min(st, sa) <= 1e-10: raise ValueError("Degenerate teacher TRAIN logits")
    t, a = text_z / st, audio_z / sa
    qt, qa = expit(t / temperature), expit(a / temperature)
    ct, ca = -np.expm1(-abs(t)), -np.expm1(-abs(a))
    wt = np.divide(ct, ct + ca, out=np.full_like(ct, .5), where=ct + ca > 1e-12)
    return (qt + qa) / 2, wt * qt + (1 - wt) * qa, {"text_scale": st, "audio_scale": sa,
             "mean_text_reliability_weight": float(wt.mean())}


def predict(model, audio, text, device):
    model.eval(); result = []
    with torch.inference_mode():
        for start in range(0, len(audio), 64):
            z = model(torch.from_numpy(audio[start:start + 64]).to(device),
                      torch.from_numpy(text[start:start + 64]).to(device))
            result.extend(torch.sigmoid(z).cpu().numpy())
    return np.asarray(result)


def perturb(a, t, scenario, severity, rng):
    a, t = a.copy(), t.copy()
    if scenario in ("missing_audio", "missing_both"): a[:] = 0
    if scenario in ("missing_text", "missing_both"): t[:] = 0
    if scenario in ("noise_audio", "noise_both"): a += rng.normal(0, severity, a.shape)
    if scenario in ("noise_text", "noise_both"): t += rng.normal(0, severity, t.shape)
    return a.astype(np.float32), t.astype(np.float32)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--student-branches", type=Path, required=True)
    p.add_argument("--text-teacher", type=Path, required=True)
    p.add_argument("--audio-teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    a = p.parse_args(argv)
    full = verified_split(a.split_dir)
    tr = full.loc[full.split.eq("train")].sort_values("participant_id")
    va = full.loc[full.split.eq("val")].sort_values("participant_id")
    expected_sha = digest(a.split_dir / "manifest.csv")
    for folder in (a.student_branches, a.text_teacher, a.audio_teacher):
        audit = json.loads((folder / "audit.json").read_text())
        signature = audit["signature"]
        if signature.get("split_sha256") != expected_sha:
            raise ValueError(f"Checkpoint trained on another split: {folder}")
    audio_tr = load(a.student_branches, "train", "audio", tr)
    audio_va = load(a.student_branches, "val", "audio", va)
    text_tr = load(a.student_branches, "train", "text", tr)
    text_va = load(a.student_branches, "val", "text", va)
    if (audio_tr.shape[1], text_tr.shape[1]) != (256, 64):
        raise ValueError("Expected independent compact student audio256/text64 embeddings")
    tz = targets(a.text_teacher / "train_text_targets.csv", tr, "text")
    az = targets(a.audio_teacher / "train_audio_targets.csv", tr, "audio")
    standard, ra, scales = distillation_targets(tz, az)
    source = {str(path): digest(path) for path in (
        a.student_branches / "train_audio_embeddings.npz", a.student_branches / "val_audio_embeddings.npz",
        a.student_branches / "train_text_embeddings.npz", a.student_branches / "val_text_embeddings.npz",
        a.text_teacher / "train_text_targets.csv", a.audio_teacher / "train_audio_targets.csv")}
    signature = {"split_sha256": expected_sha, "source_sha256": source,
                 "epochs": a.epochs, "patience": a.patience, "seed": 42,
                 "temperature": 2., "kd_weight": .5, "threshold": .5}
    if a.output.exists() and any(a.output.iterdir()):
        audit = a.output / "audit.json"
        if audit.is_file() and json.loads(audit.read_text()).get("signature") == signature:
            print("Verified existing student fusion:", a.output); return
        raise ValueError("Use a fresh fusion output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    amean, astd = audio_tr.mean(0), np.where(audio_tr.std(0) < 1e-6, 1., audio_tr.std(0))
    tmean, tstd = text_tr.mean(0), np.where(text_tr.std(0) < 1e-6, 1., text_tr.std(0))
    np.savez_compressed(a.output / "train_only_scalers.npz", audio_mean=amean, audio_std=astd,
                        text_mean=tmean, text_std=tstd)
    A, V = ((audio_tr - amean) / astd).astype(np.float32), ((audio_va - amean) / astd).astype(np.float32)
    T, W = ((text_tr - tmean) / tstd).astype(np.float32), ((text_va - tmean) / tstd).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {}; robustness = []
    for mode, soft in (("no_kd", None), ("standard_kd", standard), ("ra_kd", ra)):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
        model = FrozenBranchFusion().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-4)
        pos = torch.tensor(float(sum(tr.label == 0) / sum(tr.label == 1)), device=device)
        dataset = TensorDataset(torch.from_numpy(A), torch.from_numpy(T),
                                torch.from_numpy(tr.label.to_numpy(np.float32)),
                                torch.from_numpy(np.asarray(soft if soft is not None else np.zeros(len(tr)), np.float32)))
        loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(42))
        best = (-1., -1., -1.); stale = 0
        for epoch in range(1, a.epochs + 1):
            model.train()
            for ab, tb, yb, qb in loader:
                ab, tb, yb, qb = [x.to(device) for x in (ab, tb, yb, qb)]
                z = model(ab, tb)
                hard = torch.nn.functional.binary_cross_entropy_with_logits(z, yb, pos_weight=pos)
                if soft is None: loss = hard
                else:
                    kd = torch.nn.functional.binary_cross_entropy_with_logits(z / 2., qb) * 4.
                    loss = .5 * hard + .5 * kd
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            valp = predict(model, V, W, device); m = metric(va.label.to_numpy(int), valp)
            key = (m["macro_f1"], m["depressed_f1"], m["auroc"])
            print(f"{mode} epoch {epoch}/{a.epochs}: val macroF1={m['macro_f1']:.4f}")
            if key > best:
                best = key; stale = 0
                torch.save({"weights": model.state_dict(), "best_epoch": epoch,
                            "signature": signature}, a.output / f"{mode}_best.pt")
            else: stale += 1
            if stale >= a.patience: break
        state = torch.load(a.output / f"{mode}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["weights"])
        trainp = predict(model, A, T, device); valp = predict(model, V, W, device)
        result[mode] = {"train": metric(tr.label.to_numpy(int), trainp),
                        "val": metric(va.label.to_numpy(int), valp),
                        "best_epoch": state["best_epoch"]}
        for name, frame, prob in (("train", tr, trainp), ("val", va, valp)):
            pd.DataFrame({"participant_id": frame.participant_id.to_numpy(int),
                          "label": frame.label.to_numpy(int), "probability": prob,
                          "prediction": (prob >= .5).astype(int)}).to_csv(a.output / f"{mode}_{name}_predictions.csv", index=False)
        for si, scenario in enumerate(("missing_audio", "missing_text", "missing_both",
                                        "noise_audio", "noise_text", "noise_both")):
            is_noise = scenario.startswith("noise")
            for severity in ((.5, 1.) if is_noise else (0.,)):
                for repeat in range(20 if is_noise else 1):
                    rng = np.random.default_rng(np.random.SeedSequence([42, si, int(severity * 10), repeat]))
                    av, tv = perturb(V, W, scenario, severity, rng)
                    m = metric(va.label.to_numpy(int), predict(model, av, tv, device))
                    robustness.append({"mode": mode, "scenario": scenario, "noise_sd": severity,
                                       "repeat": repeat, **{k: m[k] for k in ("macro_f1", "depressed_f1", "auroc")}})
    pd.DataFrame(robustness).to_csv(a.output / "val_missing_noise.csv", index=False)
    (a.output / "audit.json").write_text(json.dumps({"signature": signature,
      "student_test_opened": False, "oof": False, "student_backbone_independent_of_teacher": True,
      "target_scales": scales, "results": result}, indent=2) + "\n")
    print("Saved matched student fusion comparison; student test closed:", a.output)


if __name__ == "__main__": main()
