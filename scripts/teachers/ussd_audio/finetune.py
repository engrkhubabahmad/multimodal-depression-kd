from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import AUTHOR_COMMIT, AUTHOR_TRAIN_CROP_FRAMES
from .common import (
    aggregate_segments, infer_segments, load_author_model, load_author_stats,
    load_feature_manifest, metric_dict, normalise_segments, save_json, segment_feature, sha256,
)
from .dataset import CroppedSegmentDataset


def evaluate(model, frame, mean, std, device, batch_size=64):
    participants = []
    for row in frame.itertuples(index=False):
        feat = np.load(row.feature_path).astype(np.float32)
        seg = normalise_segments(segment_feature(feat), mean, std)
        pred = infer_segments(model, seg, device, batch_size)
        participants.append(aggregate_segments(pred, row.participant_id, row.label))
    return pd.DataFrame(participants)


def train_one_epoch(model, train, mean, std, device, optimizer, epoch, seed, batch_size):
    model.train(); dataset = CroppedSegmentDataset(train, mean, std, seed, epoch)
    generator = torch.Generator().manual_seed(seed + epoch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda", generator=generator)
    counts = train.label.value_counts().to_dict(); total = len(train)
    class_weight = torch.tensor([total / (2.0 * counts[0]), total / (2.0 * counts[1])], dtype=torch.float32, device=device)
    losses = []
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True); logits, _, _ = model.forward_logits(x)
        raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (raw * class_weight[y.long()]).mean(); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)), len(dataset)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True); p.add_argument("--author-root", default="/content/solo_teacher_sources/USSD-depression"); p.add_argument("--output", required=True); p.add_argument("--audit-json", required=True)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--patience", type=int, default=10); p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--train-batch-size", type=int, default=20); p.add_argument("--eval-batch-size", type=int, default=64)
    a = p.parse_args(argv); features, author, out = Path(a.features), Path(a.author_root), Path(a.output); out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    audit = json.loads(Path(a.audit_json).read_text(encoding="utf-8"))
    local = audit.get("local_dev34", {})
    if audit.get("test_opened") is not False or audit.get("participant_440_excluded") is not True or local.get("n") != 34:
        raise AssertionError("Frozen DEV-34 audit is missing or does not satisfy the leakage guard")
    train_prep_path = features / "preprocessing_train.json"
    if not train_prep_path.exists(): raise FileNotFoundError(f"Missing TRAIN preprocessing provenance: {train_prep_path}")
    train_prep = json.loads(train_prep_path.read_text(encoding="utf-8"))
    if train_prep.get("test_opened") is not False or train_prep.get("splits") != ["train"]: raise AssertionError("TRAIN preprocessing provenance is not test-closed")
    if train_prep.get("compare16_config_sha256") != audit.get("compare16_config_sha256"): raise AssertionError("TRAIN and audited DEV used different ComParE16 configurations")
    train = load_feature_manifest(features, "train"); dev = load_feature_manifest(features, "dev")
    if set(train.participant_id) & set(dev.participant_id): raise AssertionError("TRAIN/DEV participant overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, _, source_ckpt = load_author_model(author, device); mean, std, stats_path = load_author_stats(author)
    if audit.get("checkpoint_sha256") != sha256(source_ckpt): raise AssertionError("Audit checkpoint hash does not match the fine-tuning source checkpoint")
    if audit.get("normalization_sha256") != sha256(stats_path): raise AssertionError("Audit normalization hash does not match the fine-tuning artifact")
    # Epoch 0 is the verified released checkpoint and must remain eligible.
    baseline_pred = evaluate(model, dev, mean, std, device, a.eval_batch_size); baseline = metric_dict(baseline_pred)
    audited_f1 = float(local["macro_f1"])
    if abs(baseline["macro_f1"] - audited_f1) > 1e-10:
        raise AssertionError(f"Epoch-0 DEV mismatch: current={baseline['macro_f1']}, audited={audited_f1}")
    best = {**baseline, "epoch": 0, "train_loss": None}
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "train_loss": None, "train_segments": 0,
                "dev_macro_f1": baseline["macro_f1"], "dev_accuracy": baseline["accuracy"]}]
    print(history[0])
    stalled = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)

    for epoch in range(1, a.epochs + 1):
        train_loss, train_segments = train_one_epoch(model, train, mean, std, device, optimizer, epoch, a.seed, a.train_batch_size)
        dev_pred = evaluate(model, dev, mean, std, device, a.eval_batch_size); dev_metrics = metric_dict(dev_pred)
        row = {"epoch": epoch, "train_loss": train_loss, "train_segments": train_segments, "dev_macro_f1": dev_metrics["macro_f1"], "dev_accuracy": dev_metrics["accuracy"]}; history.append(row); print(row)
        if dev_metrics["macro_f1"] > best["macro_f1"] + 1e-12:
            best = {**dev_metrics, "epoch": epoch, "train_loss": train_loss}; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stalled = 0
        else: stalled += 1
        if stalled >= a.patience: break

    model.load_state_dict(best_state); model.to(device).eval()
    train_pred = evaluate(model, train, mean, std, device, a.eval_batch_size); dev_pred = evaluate(model, dev, mean, std, device, a.eval_batch_size)
    train_pred.to_csv(out / "train_kd_targets.csv", index=False); dev_pred.to_csv(out / "dev_predictions.csv", index=False); pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    source_hash, stats_hash = sha256(source_ckpt), sha256(stats_path)
    torch.save({
        "model_state_dict": model.state_dict(), "source": "Ravi et al. USSD run #4 md_35_epochs.pth", "author_commit": AUTHOR_COMMIT,
        "source_checkpoint_sha256": source_hash, "normalization_sha256": stats_hash, "selection": "DEV-34 participant macro-F1 only; fixed author vote aggregation; earliest epoch wins ties",
        "best": best, "learning_rate": a.learning_rate, "weight_decay": a.weight_decay, "seed": a.seed,
    }, out / "frozen_ussd_audio_teacher.pt")
    summary = {
        "protocol": "USSD run #4 with optional TRAIN-107 depression-only adaptation; verified frozen epoch 0 retained unless DEV-34 improves", "author_commit": AUTHOR_COMMIT,
        "source_checkpoint": source_ckpt.name, "source_checkpoint_sha256": source_hash, "author_normalization_reused": True, "normalization_sha256": stats_hash, "compare16_config_sha256": audit.get("compare16_config_sha256"), "train_preprocessing_manifest": str(train_prep_path),
        "training_crop": {"random_train_only": True, "frames": AUTHOR_TRAIN_CROP_FRAMES, "segment_frames": 384, "batch_size": a.train_batch_size, "all_cropped_segments_seen_once_per_epoch": True},
        "fit_participants": 107, "selection_participants": 34, "participant_440_excluded": True, "test_opened": False,
        "selection": "Epoch 0 frozen checkpoint plus optional adapted epochs; DEV-34 participant macro-F1 only; no threshold search; earliest epoch wins ties", "best": best,
        "train_full_stream": metric_dict(train_pred), "dev_full_stream": metric_dict(dev_pred),
        "kd_export": "train_kd_targets.csv uses mean segment probability and logit(mean probability); author hard-vote fields retained separately",
        "label_policy": "Local AVEC split labels are used unchanged during adaptation; author-side participant 409 relabel is not silently applied.",
        "adaptation_batches": "Segment batches are shuffled across TRAIN participants; class weighting is computed from TRAIN participant counts only.",
        "caveat": "Any epoch >0 uses depression loss only. The unreleased external speaker-embedding dependency is not reconstructed, so adapted epochs are not a reproduction of the full auxiliary speaker-disentanglement objective. Epoch 0 is the released verified checkpoint.",
    }
    save_json(out / "metrics.json", summary); print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__": main()
