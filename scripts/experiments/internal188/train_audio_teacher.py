"""Fine-tune USSD ComParE16+LSTM on canonical TRAIN; select on DEV."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from scripts.teachers.ussd_audio.common import CustomComparE16, AUTHOR_RUN4_CHECKPOINT_SHA256, load_author_stats
from .train_text_teacher import verified_split
from .split import digest
from .train_text_teacher import metric

FRAMES = 384


def rows_for(features, split):
    cached = pd.read_csv(Path(features) / "participant_manifest.csv")
    selected = split.loc[split.split.isin(['train', 'val'])]
    if cached.participant_id.duplicated().any() or set(cached.participant_id) != set(selected.participant_id):
        raise ValueError("TRAIN/VAL audio feature coverage mismatch")
    indexed = cached.set_index("participant_id")
    for r in selected.itertuples(index=False):
        f = indexed.loc[int(r.participant_id)]
        if int(f.label) != int(r.label) or str(f.split) != r.split:
            raise ValueError(f"Audio feature/split mismatch: {r.participant_id}")
    return indexed


def train_stats(frame, indexed):
    n = 0; total = np.zeros((130, 1), float); square = np.zeros((130, 1), float)
    for pid in frame.participant_id:
        x = np.load(indexed.loc[int(pid), "feature_path"], mmap_mode="r", allow_pickle=False)
        if x.ndim != 2 or x.shape[0] != 130 or not x.shape[1]: raise ValueError(f"Invalid feature shape: {pid}")
        for start in range(0, x.shape[1], 4096):
            z = np.asarray(x[:, start:start + 4096], float)
            if not np.isfinite(z).all(): raise ValueError(f"Non-finite audio: {pid}")
            n += z.shape[1]; total += z.sum(1, keepdims=True); square += (z * z).sum(1, keepdims=True)
    mean = total / n; std = np.sqrt(np.maximum(square / n - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class Segments(Dataset):
    def __init__(self, frame, indexed, mean, std, samples=4, seed=42):
        self.rows = frame.sort_values("participant_id").reset_index(drop=True)
        self.indexed = indexed; self.mean = mean; self.std = std
        self.samples = samples; self.seed = seed; self.epoch = 0

    def __len__(self): return len(self.rows) * self.samples

    def __getitem__(self, index):
        row = self.rows.iloc[index // self.samples]
        pid = int(row.participant_id)
        x = np.load(self.indexed.loc[pid, "feature_path"], mmap_mode="r", allow_pickle=False)
        nseg = math.ceil(x.shape[1] / FRAMES)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, pid, index % self.samples]))
        segment = int(rng.integers(nseg)); start = segment * FRAMES
        block = np.zeros((130, FRAMES), np.float32)
        portion = np.asarray(x[:, start:start + FRAMES], np.float32)
        block[:, :portion.shape[1]] = portion
        return torch.from_numpy(((block - self.mean) / self.std).astype(np.float32)), torch.tensor(float(row.label))


class ParticipantBags(Segments):
    """One participant per item; sample segments across the entire recording."""
    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]; pid = int(row.participant_id)
        x = np.load(self.indexed.loc[pid, "feature_path"], mmap_mode="r", allow_pickle=False)
        nseg = math.ceil(x.shape[1] / FRAMES)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, pid]))
        blocks = np.zeros((self.samples, 130, FRAMES), np.float32)
        for j in range(self.samples):
            segment = min(int((j + rng.random()) * nseg / self.samples), nseg - 1)
            part = np.asarray(x[:, segment * FRAMES:(segment + 1) * FRAMES], np.float32)
            blocks[j, :, :part.shape[1]] = part
        return torch.from_numpy(((blocks - self.mean[None]) / self.std[None]).astype(np.float32)), torch.tensor(float(row.label))


class BalancedPairs:
    """Every batch contains one positive and one negative TRAIN participant."""
    def __init__(self, rows, seed):
        self.positive = np.flatnonzero(rows.label.to_numpy(int) == 1)
        self.negative = np.flatnonzero(rows.label.to_numpy(int) == 0)
        if not len(self.positive) or not len(self.negative): raise ValueError('Both classes required')
        self.seed = seed; self.epoch = 0

    def __len__(self): return len(self.negative)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        neg = rng.permutation(self.negative)
        pos = np.concatenate([rng.permutation(self.positive) for _ in
                              range(math.ceil(len(neg) / len(self.positive)))])[:len(neg)]
        for positive, negative in zip(pos, neg):
            pair = [int(positive), int(negative)]
            rng.shuffle(pair)
            yield pair


def evaluate(model, frame, indexed, mean, std, device, batch=16):
    model.eval(); records = []; features = []
    with torch.inference_mode():
        for row in frame.sort_values("participant_id").itertuples(index=False):
            x = np.load(indexed.loc[int(row.participant_id), "feature_path"], mmap_mode="r", allow_pickle=False)
            probs = []; embeddings = []
            for start in range(0, x.shape[1], FRAMES * batch):
                blocks = []
                for a in range(start, min(start + FRAMES * batch, x.shape[1]), FRAMES):
                    block = np.zeros((130, FRAMES), np.float32)
                    chunk = np.asarray(x[:, a:a + FRAMES], np.float32)
                    block[:, :chunk.shape[1]] = chunk
                    blocks.append(((block - mean) / std).astype(np.float32))
                batch_x = torch.from_numpy(np.stack(blocks)).to(device)
                if hasattr(model, "forward_logits"):
                    _, p, h = model.forward_logits(batch_x)
                else:
                    z, h = model(batch_x); p = torch.sigmoid(z)
                probs.extend(p.cpu().numpy()); embeddings.extend(h.cpu().numpy())
            q = float(np.mean(probs)); e = np.asarray(embeddings, np.float32)
            records.append({"participant_id": int(row.participant_id), "label": int(row.label),
                            "audio_probability": float(np.clip(q, 1e-6, 1 - 1e-6)),
                            "audio_logit": float(np.log(np.clip(q, 1e-6, 1 - 1e-6) /
                                                         (1 - np.clip(q, 1e-6, 1 - 1e-6))))})
            features.append(np.concatenate([e.mean(0), e.std(0)]).astype(np.float32))
    return pd.DataFrame(records), np.stack(features)


def probability_summary(frame):
    return {str(label): {'min': float(group.audio_probability.min()),
                         'median': float(group.audio_probability.median()),
                         'max': float(group.audio_probability.max())}
            for label, group in frame.groupby('label')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=103)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument('--published-checkpoint', type=Path)
    p.add_argument('--author-root', type=Path,
                   help='USSD author checkout for pinned run-4 normalization artifact')
    p.add_argument('--balanced-pairs', action='store_true',
                   help='One depressed and one non-depressed TRAIN participant per batch; fresh model only')
    a = p.parse_args(argv)
    if a.seed not in (42, 103) or min(a.epochs, a.patience, a.batch_size) < 1: raise ValueError("Invalid training settings")
    if a.balanced_pairs and (a.published_checkpoint or a.batch_size != 16):
        raise ValueError('Balanced pairs require fresh weights and batch-size 16 (8 segments per participant)')
    full = verified_split(a.split_dir, a.coverage_dir); indexed = rows_for(a.features, full)
    tr = full.loc[full.split.eq("train")]; va = full.loc[full.split.eq("val")]
    if bool(a.published_checkpoint) != bool(a.author_root):
        raise ValueError('Published checkpoint requires --author-root for matched normalization')
    if a.published_checkpoint:
        mean, std, normalizer_path = load_author_stats(a.author_root)
        normalization_sha256 = digest(normalizer_path)
    else:
        mean, std = train_stats(tr, indexed)
        normalization_sha256 = None
    signature = {"split_sha256": digest(a.split_dir / "manifest.csv"),
                 "feature_manifest_sha256": digest(a.features / "participant_manifest.csv"),
                 "seed": a.seed, "epochs": a.epochs, "patience": a.patience,
                 "batch_size": a.batch_size, "lr": a.lr, "architecture": "USSD CustomComparE16",
                 "initialization": 'published' if a.published_checkpoint else 'fresh',
                 "published_checkpoint_sha256": digest(a.published_checkpoint) if a.published_checkpoint else None,
                 "normalization_sha256": normalization_sha256,
                 "balanced_pairs": a.balanced_pairs,
                 "training_unit": "participant_mean_probability", "segments_per_participant": 8}
    if a.published_checkpoint and signature['published_checkpoint_sha256'] != AUTHOR_RUN4_CHECKPOINT_SHA256:
        raise ValueError('Published USSD run #4 checkpoint SHA-256 differs from pinned original')
    if a.output.exists() and any(a.output.iterdir()):
        audit = a.output / "audit.json"
        if audit.is_file() and json.loads(audit.read_text()).get("signature") == signature:
            print("Verified existing audio teacher:", a.output); return
        raise ValueError("Use a fresh audio-teacher output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    np.savez_compressed(a.output / "normalization.npz", mean=mean, std=std)
    if a.batch_size % 8: raise ValueError("batch-size must be a multiple of 8 segments")
    dataset = ParticipantBags(tr, indexed, mean, std, samples=8, seed=a.seed)
    if a.balanced_pairs:
        pairs = BalancedPairs(dataset.rows, a.seed)
        loader = DataLoader(dataset, batch_sampler=pairs, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    else:
        generator = torch.Generator().manual_seed(a.seed)
        loader = DataLoader(dataset, batch_size=a.batch_size // 8, shuffle=True, generator=generator,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomComparE16().to(device)
    if a.published_checkpoint:
        saved = torch.load(a.published_checkpoint, map_location='cpu', weights_only=False)
        weights = saved.get('state_dict', saved.get('model_state_dict', saved))
        if all(k.startswith('module.') for k in weights):
            weights = {k[7:]: v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True)
        print('Initialized USSD architecture from pinned run #4 checkpoint:', a.published_checkpoint)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    pos_weight = torch.tensor(float(sum(tr.label == 0) / sum(tr.label == 1)), device=device)
    best = (-1., -1., -1.); stale = 0; history = []; epoch_zero = None
    if a.published_checkpoint:
        initial_train, _ = evaluate(model, tr, indexed, mean, std, device, a.batch_size)
        initial_val, _ = evaluate(model, va, indexed, mean, std, device, a.batch_size)
        initial_train_metric = metric(initial_train.label.to_numpy(int), initial_train.audio_probability.to_numpy(float))
        initial_val_metric = metric(initial_val.label.to_numpy(int), initial_val.audio_probability.to_numpy(float))
        epoch_zero = {'train': initial_train_metric, 'val': initial_val_metric,
                      'train_probability_by_label': probability_summary(initial_train),
                      'val_probability_by_label': probability_summary(initial_val),
                      'normalization_sha256': normalization_sha256}
        best = (initial_val_metric['macro_f1'], initial_val_metric['depressed_f1'], initial_val_metric['auroc'])
        torch.save({'model_state_dict': model.state_dict(), 'best_epoch': 0,
                    'signature': signature}, a.output / 'best.pt')
        print('USSD published epoch 0 TRAIN macroF1=', initial_train_metric['macro_f1'],
              'VAL macroF1=', initial_val_metric['macro_f1'],
              'VAL CM=', initial_val_metric['confusion_matrix'], flush=True)
    progress = tqdm(range(1, a.epochs + 1), desc="Audio teacher epochs", unit="epoch")
    for epoch in progress:
        dataset.epoch = epoch; model.train(); losses = []
        if a.balanced_pairs: pairs.epoch = epoch
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, probabilities, _ = model.forward_logits(x.flatten(0, 1))
            participant_prob = probabilities.view(len(y), dataset.samples).mean(1).clamp(1e-5, 1 - 1e-5)
            loss = torch.nn.functional.binary_cross_entropy(
                participant_prob, y,
                weight=None if a.balanced_pairs else torch.where(y > 0, pos_weight, 1.))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); losses.append(float(loss.detach().cpu()))
        train_result, _ = evaluate(model, tr, indexed, mean, std, device, a.batch_size)
        result, _ = evaluate(model, va, indexed, mean, std, device, a.batch_size)
        tm = metric(train_result.label.to_numpy(int), train_result.audio_probability.to_numpy(float))
        m = metric(result.label.to_numpy(int), result.audio_probability.to_numpy(float))
        key = (m["macro_f1"], m["depressed_f1"], m["auroc"])
        progress.set_postfix(train_f1=f"{tm['macro_f1']:.3f}", val_f1=f"{m['macro_f1']:.3f}")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "train_macro_f1": tm["macro_f1"], "train_depressed_f1": tm["depressed_f1"],
                        "val_macro_f1": m["macro_f1"], "val_depressed_f1": m["depressed_f1"],
                        "val_auroc": m["auroc"]})
        print(f"Audio epoch {epoch}/{a.epochs} train macroF1={tm['macro_f1']:.4f} val macroF1={m['macro_f1']:.4f} val depressedF1={m['depressed_f1']:.4f}")
        if key > best:
            best = key; stale = 0
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": epoch,
                        "signature": signature}, a.output / "best.pt")
        else: stale += 1
        if stale >= a.patience: break
    pd.DataFrame(history).to_csv(a.output / "history.csv", index=False)
    state = torch.load(a.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    scores_out = {}
    for name, frame in (("train", tr), ("val", va)):
        result, embedding = evaluate(model, frame, indexed, mean, std, device, a.batch_size)
        result.to_csv(a.output / f"{name}_audio_targets.csv", index=False)
        np.savez_compressed(a.output / f"{name}_audio_embeddings.npz",
                            participant_ids=result.participant_id.to_numpy(int),
                            labels=result.label.to_numpy(int), embedding=embedding,
                            probability=result.audio_probability.to_numpy(float))
        scores_out[name] = metric(result.label.to_numpy(int), result.audio_probability.to_numpy(float))
    (a.output / "audit.json").write_text(json.dumps({"signature": signature,
      "architecture": "USSD CustomComparE16", "previous_checkpoint_loaded": bool(a.published_checkpoint),
      "normalization_source": 'pinned_author_run4' if a.published_checkpoint else 'new_train_only',
      "normalization_sha256": normalization_sha256, "epoch_zero": epoch_zero,
      "balanced_pairs": a.balanced_pairs,
      "published_checkpoint_prior_exposure": {
          "original_train_ids_in_val": int(sum(va.source_split.eq('canonical_train'))) if a.published_checkpoint else 0,
          "original_train_ids_in_internal_test": int(sum(full.loc[full.split.eq('student_test'),
                                                     'source_split'].eq('canonical_train'))) if a.published_checkpoint else 0,
          "clean_holdout_claim_valid": False if a.published_checkpoint else True},
      "train_only_normalization": not bool(a.published_checkpoint), "training_unit": "participant", "train_participants": int(len(tr)), "val_participants": int(len(va)),
      "student_test_opened": False, "oof": False, "best_epoch": state["best_epoch"], **scores_out}, indent=2) + "\n")
    print("Saved fresh audio teacher:", a.output)


if __name__ == "__main__": main()
