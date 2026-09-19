from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from . import AUTHOR_COMMIT, AUTHOR_TRAIN_CROP_FRAMES
from .common import (
    aggregate_segments, crop_author_train, infer_segments, load_author_model, load_author_stats,
    load_feature_manifest, metric_dict, normalise_segments, save_json, segment_feature, sha256,
)


def evaluate(model, frame, mean, std, device, batch_size=64):
    participants = []
    for row in frame.itertuples(index=False):
        feat = np.load(row.feature_path).astype(np.float32)
        seg = normalise_segments(segment_feature(feat), mean, std)
        pred = infer_segments(model, seg, device, batch_size)
        participants.append(aggregate_segments(pred, row.participant_id, row.label))
    return pd.DataFrame(participants)


def train_one_epoch(model, train, mean, std, device, optimizer, epoch, seed):
    model.train(); order = list(train.itertuples(index=False)); random.Random(seed + epoch).shuffle(order)
    counts = train.label.value_counts().to_dict(); total = len(train); class_weight = {int(k): total / (2.0 * int(v)) for k, v in counts.items()}
    losses = []
    for row in tqdm(order, desc=f"USSD fine-tune epoch {epoch:02d}", colour="green"):
        feat = np.load(row.feature_path).astype(np.float32)
        rng = random.Random(seed + epoch * 100003 + int(row.participant_id))
        seg = normalise_segments(segment_feature(crop_author_train(feat, rng)), mean, std)
        x = torch.from_numpy(seg).float().to(device); y = torch.full((len(seg),), float(row.label), device=device)
        optimizer.zero_grad(set_to_none=True); logits, _, _ = model.forward_logits(x)
        loss = F.binary_cross_entropy_with_logits(logits, y) * class_weight[int(row.label)]
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True); p.add_argument("--author-root", default="/content/solo_teacher_sources/USSD-depression"); p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=40); p.add_argument("--patience", type=int, default=10); p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--eval-batch-size", type=int, default=64)
    a = p.parse_args(argv); features, author, out = Path(a.features), Path(a.author_root), Path(a.output); out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    train = load_feature_manifest(features, "train"); dev = load_feature_manifest(features, "dev")
    if set(train.participant_id) & set(dev.participant_id): raise AssertionError("TRAIN/DEV participant overlap")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, source_state, source_ckpt = load_author_model(author, device); mean, std, stats_path = load_author_stats(author)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)

    best_state, best, history, stalled = None, None, [], 0
    for epoch in range(1, a.epochs + 1):
        train_loss = train_one_epoch(model, train, mean, std, device, optimizer, epoch, a.seed)
        dev_pred = evaluate(model, dev, mean, std, device, a.eval_batch_size); dev_metrics = metric_dict(dev_pred)
        row = {"epoch": epoch, "train_loss": train_loss, "dev_macro_f1": dev_metrics["macro_f1"], "dev_accuracy": dev_metrics["accuracy"]}; history.append(row); print(row)
        if best is None or dev_metrics["macro_f1"] > best["macro_f1"] + 1e-12:
            best = {**dev_metrics, "epoch": epoch, "train_loss": train_loss}; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stalled = 0
        else: stalled += 1
        if stalled >= a.patience: break

    if best_state is None: raise RuntimeError("No checkpoint selected")
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
        "protocol": "TRAIN-107 depression-only adaptation from frozen USSD run #4", "author_commit": AUTHOR_COMMIT,
        "source_checkpoint": source_ckpt.name, "source_checkpoint_sha256": source_hash, "author_normalization_reused": True, "normalization_sha256": stats_hash,
        "training_crop": {"random_train_only": True, "frames": AUTHOR_TRAIN_CROP_FRAMES, "segment_frames": 384},
        "fit_participants": 107, "selection_participants": 34, "participant_440_excluded": True, "test_opened": False,
        "selection": "DEV-34 participant macro-F1 only; no threshold search; earliest epoch wins ties", "best": best,
        "train_full_stream": metric_dict(train_pred), "dev_full_stream": metric_dict(dev_pred),
        "kd_export": "train_kd_targets.csv uses mean segment probability and logit(mean probability); author hard-vote fields retained separately",
        "caveat": "Fine-tuning uses depression loss only. The unreleased external speaker-embedding dependency is not reconstructed, so this is adaptation of the released USSD depression network, not a reproduction of the full auxiliary speaker-disentanglement training objective.",
    }
    save_json(out / "metrics.json", summary); print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__": main()
