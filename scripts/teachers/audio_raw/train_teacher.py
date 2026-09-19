"""Participant-level audio teacher using Ping et al. raw audio preprocessing.

The feature contract follows the released DepressionEstimation repository:
Participant speech only; concatenated speech; 80-bin log-Mel; n_fft=2048;
hop=533; 60-second (1800-frame) windows with 10-second (1500-frame) hop;
row L2 normalization.  Every clip remains within its participant split.

The classifier uses the released ConvLSTM_Audio backbone with a binary head.
Clip training is participant-balanced; all reporting and model selection are
participant-level after mean-logit aggregation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import wave
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


def one(root: Path, pattern: str) -> Path:
    files = list(root.rglob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r}; found {len(files)}")
    return files[0]


def read_split(path: Path, excluded: set[int]) -> list[tuple[int, int]]:
    frame = pd.read_csv(path)
    cols = {c.casefold(): c for c in frame.columns}
    return [(int(pid), int(label)) for pid, label in zip(frame[cols["participant_id"]], frame[cols["phq8_binary"]]) if int(pid) not in excluded]


def source_audio(root: Path, pid: int) -> tuple[np.ndarray, int, pd.DataFrame]:
    wav_path = one(root, f"{pid}_AUDIO.wav")
    transcript_path = one(root, f"{pid}_TRANSCRIPT.csv")
    with wave.open(str(wav_path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise ValueError(f"{wav_path}: only PCM16 WAV is supported")
        signal = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
        rate = handle.getframerate()
    transcript = pd.read_csv(transcript_path, sep="\t").fillna("")
    return signal, rate, transcript


def participant_speech(signal: np.ndarray, sample_rate: int, transcript: pd.DataFrame) -> np.ndarray:
    required = {"start_time", "stop_time", "speaker", "value"}
    if missing := required - set(transcript.columns):
        raise ValueError(f"Transcript missing {sorted(missing)}")
    frame = transcript.sort_values(["start_time", "stop_time"], kind="stable")
    speaker = frame.speaker.astype(str).str.strip().str.casefold()
    value = frame.value.astype(str)
    keep = (speaker == "participant") & ~value.str.contains("scrubbed_entry", case=False, regex=False)
    clips = []
    for row in frame[keep].itertuples():
        start = max(0, int(float(row.start_time) * sample_rate))
        stop = min(len(signal), int(float(row.stop_time) * sample_rate))
        if stop > start:
            clips.append(signal[start:stop])
    if not clips:
        raise ValueError("No usable Participant speech")
    return np.hstack(clips)


def mel_windows(signal: np.ndarray, sample_rate: int):
    mel = librosa.power_to_db(librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_fft=2048, hop_length=533, n_mels=80
    ))
    mel = normalize(mel).astype(np.float32)
    frame, hop = 1800, 1500
    count = max(1, (mel.shape[1] - frame) // hop + 2)
    for index in range(count):
        start = index * hop
        item = np.zeros((80, frame), dtype=np.float32)
        available = mel[:, start:start + frame]
        item[:, :available.shape[1]] = available
        yield item


def make_cache(root: Path, records: list[tuple[int, int]], split: str, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, label in tqdm(records, desc=f"Audio {split} preprocessing", colour="green"):
        existing = sorted(cache.glob(f"{pid}_*.npy"))
        if not existing:
            signal, rate, transcript = source_audio(root, pid)
            speech = participant_speech(signal, rate, transcript)
            existing = []
            for clip, feature in enumerate(mel_windows(speech, rate)):
                path = cache / f"{pid}_{clip:04d}.npy"
                np.save(path, feature)
                existing.append(path)
        for path in existing:
            rows.append({"participant_id": pid, "label": label, "split": split, "feature_path": str(path)})
    return pd.DataFrame(rows)


class ClipDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame):
        self.rows = manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return {
            "x": torch.from_numpy(np.load(row.feature_path)).float(),
            "y": torch.tensor(float(row.label)),
            "participant_id": torch.tensor(int(row.participant_id)),
        }


def participant_metrics(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    grouped = frame.groupby(["participant_id", "label"], as_index=False).agg(logit=("logit", "mean"), clips=("logit", "size"))
    grouped["prob_depressed"] = 1.0 / (1.0 + np.exp(-grouped.logit))
    grouped["prediction"] = (grouped.prob_depressed >= .5).astype(int)
    y, p = grouped.label.to_numpy(), grouped.prob_depressed.to_numpy()
    result = {
        "n": int(len(grouped)),
        "accuracy": float(accuracy_score(y, grouped.prediction)),
        "macro_f1": float(classification_report(y, grouped.prediction, output_dict=True, zero_division=0)["macro avg"]["f1-score"]),
        "auroc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, grouped.prediction, labels=[0, 1]).tolist(),
        "threshold": .5,
    }
    return result, grouped


def load_backbone(source_root: Path):
    path = source_root / "models/Audio_ConvLSTM/models/convlstm.py"
    spec = importlib.util.spec_from_file_location("ping_convlstm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConvLSTM_Audio


class AudioTeacher(nn.Module):
    def __init__(self, backbone_cls):
        super().__init__()
        self.backbone = backbone_cls(
            input_dim=80, output_dim=256, conv_hidden=256, lstm_hidden=256,
            num_layers=4, activation="relu", norm="bn", dropout=.5
        )
        self.binary_head = nn.Linear(256, 1)

    def forward(self, x):
        return self.binary_head(self.backbone(x)).squeeze(1)


def infer(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Audio participant inference", colour="green"):
            logits = model(batch["x"].to(device)).detach().cpu().numpy()
            for pid, y, logit in zip(batch["participant_id"].numpy(), batch["y"].numpy(), logits):
                rows.append({"participant_id": int(pid), "label": int(y), "logit": float(logit)})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--daic-root", required=True)
    p.add_argument("--source-root", default="/content/solo_teacher_sources/DepressionEstimation")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=103)
    p.add_argument("--exclude", type=int, nargs="*", default=[440])
    a = p.parse_args(argv)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    root, source, out = Path(a.daic_root), Path(a.source_root), Path(a.output)
    train_csv, dev_csv = one(root, "train_split_Depression_AVEC2017.csv"), one(root, "dev_split_Depression_AVEC2017.csv")
    excluded = set(a.exclude)
    train_ids, dev_ids = read_split(train_csv, excluded), read_split(dev_csv, excluded)
    if len(train_ids) != 107 or len(dev_ids) != 34:
        raise AssertionError(f"Expected TRAIN=107 DEV=34, got {len(train_ids)} DEV={len(dev_ids)}")
    if {pid for pid, _ in train_ids} & {pid for pid, _ in dev_ids}:
        raise AssertionError("Participant overlap")
    if not (source / "models/Audio_ConvLSTM/models/convlstm.py").exists():
        raise FileNotFoundError("Run scripts.teachers.solo_published.bootstrap first")

    train_manifest = make_cache(root, train_ids, "train", out / "cache/train")
    dev_manifest = make_cache(root, dev_ids, "dev", out / "cache/dev")
    train_manifest.to_csv(out / "train_clip_manifest.csv", index=False)
    dev_manifest.to_csv(out / "dev_clip_manifest.csv", index=False)
    print({"train_clips": len(train_manifest), "dev_clips": len(dev_manifest), "train_participants": 107, "dev_participants": 34, "test_opened": False})

    train_set, dev_set = ClipDataset(train_manifest), ClipDataset(dev_manifest)
    clip_counts = Counter(train_manifest.participant_id.tolist())
    class_counts = Counter(train_manifest.drop_duplicates("participant_id").label.tolist())
    weights = train_manifest.apply(lambda row: 1.0 / (clip_counts[row.participant_id] * class_counts[row.label]), axis=1).to_numpy()
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=a.batch_size, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    eval_train = DataLoader(train_set, batch_size=a.batch_size, shuffle=False, num_workers=0)
    dev_loader = DataLoader(dev_set, batch_size=a.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioTeacher(load_backbone(source)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state, best_dev, stalled = None, None, 0

    for epoch in range(1, a.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Audio epoch {epoch:02d}/{a.epochs}", colour="green"):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(batch["x"].to(device)), batch["y"].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        dev_clips = infer(model, dev_loader, device)
        dev_metrics, _ = participant_metrics(dev_clips)
        print({"epoch": epoch, "loss": float(np.mean(losses)), "dev": dev_metrics})
        if best_dev is None or dev_metrics["macro_f1"] > best_dev["macro_f1"]:
            best_dev = {**dev_metrics, "epoch": epoch, "loss": float(np.mean(losses))}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        if stalled >= a.patience:
            break

    model.load_state_dict(best_state)
    train_clips, dev_clips = infer(model, eval_train, device), infer(model, dev_loader, device)
    train_metrics, train_participants = participant_metrics(train_clips)
    dev_metrics, dev_participants = participant_metrics(dev_clips)
    out.mkdir(parents=True, exist_ok=True)
    train_participants.to_csv(out / "train_predictions.csv", index=False)
    dev_participants.to_csv(out / "dev_predictions.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "architecture": "Ping ConvLSTM_Audio binary head", "selected": best_dev}, out / "frozen_audio_teacher.pt")
    metrics = {
        "protocol": "participant-level Ping-preprocessing ConvLSTM audio teacher",
        "preprocessing": {"participant_speech": True, "n_mels": 80, "n_fft": 2048, "hop_length": 533, "window_frames": 1800, "hop_frames": 1500, "row_l2_normalization": True},
        "train": train_metrics, "dev": dev_metrics, "selected": best_dev,
        "train_participants": 107, "dev_participants": 34, "test_opened": False, "excluded_ids": sorted(excluded)
    }
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
