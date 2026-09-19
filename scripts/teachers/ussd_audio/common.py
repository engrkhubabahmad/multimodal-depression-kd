from __future__ import annotations

import hashlib
import json
import pickle
import random
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from . import AUTHOR_TRAIN_CROP_FRAMES, CHECKPOINT_NAME, EXCLUDED_DEV_IDS, FREQ_BINS, RUN4_REL, SEGMENT_FRAMES

INTERRUPT = {373: [395, 428], 444: [286, 387]}
MISALIGNED = {318: 34.319917, 321: 3.8379167, 341: 6.1892, 362: 16.8582}
SYNC = {"[sync]", "[syncing]"}


def one(root: Path, pattern: str) -> Path:
    hits = list(root.rglob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r}; found {len(hits)}")
    return hits[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def read_split(path: Path, exclude: set[int] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.casefold(): c for c in df.columns}
    if "participant_id" not in cols or "phq8_binary" not in cols:
        raise ValueError(f"{path} needs Participant_ID and PHQ8_Binary")
    out = pd.DataFrame({
        "participant_id": df[cols["participant_id"]].astype(int),
        "label": df[cols["phq8_binary"]].astype(int),
    })
    if exclude:
        out = out[~out.participant_id.isin(exclude)]
    return out.sort_values("participant_id").reset_index(drop=True)


def resolve_protocol_splits(root: Path, exclude: set[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclude = EXCLUDED_DEV_IDS if exclude is None else exclude
    train = read_split(one(root, "train_split_Depression_AVEC2017.csv"), set())
    dev = read_split(one(root, "dev_split_Depression_AVEC2017.csv"), exclude)
    if len(train) != 107 or len(dev) != 34:
        raise AssertionError(f"Expected TRAIN=107 DEV=34; got TRAIN={len(train)} DEV={len(dev)}")
    if set(train.participant_id) & set(dev.participant_id):
        raise AssertionError("Participant overlap between TRAIN and DEV")
    if 440 in set(dev.participant_id):
        raise AssertionError("Participant 440 must be excluded from DEV")
    return train, dev


def author_intervals(transcript_path: Path) -> list[list[float]]:
    """Behavioral port of the pinned author's transcript_file_processing()."""
    parent_token = transcript_path.parent.name.split("_")[0]\n    file_token = transcript_path.name.split("_")[0]\n    trial = int(parent_token) if parent_token.isdigit() else int(file_token)
    data = transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    inter: list[list[float]] = []
    holding_start: float | None = None
    for j, values in enumerate(data):
        if j == 0:
            continue
        parts = values.split()
        temp = parts[0:3]
        if not temp:
            continue
        offset = MISALIGNED.get(trial, 0.0)
        start, stop = float(temp[0]) + offset, float(temp[1]) + offset
        speaker = temp[-1]
        sync = parts[-1] in SYNC if parts else False
        if speaker != "Participant" or sync:
            continue
        if trial in INTERRUPT:
            cut0, cut1 = INTERRUPT[trial]
            if start < cut0 < stop:
                inter.append([start, cut0 - 0.01])
            elif start < cut1 < stop:
                inter.append([cut1 + 0.01, stop])
            elif cut0 < start < cut1 or cut0 < stop < cut1:
                continue
            elif stop < cut0 or start > cut1:
                inter.append([start, stop])
            continue
        prev = data[j - 1].split()[0:3] if j > 0 else ["", "", "Ellie"]
        if not prev and j - 2 > 0:
            prev = data[j - 2].split()[0:3]
        if j + 1 < len(data):
            nxt = data[j + 1].split()[0:3]
            if not nxt and j + 2 < len(data):
                nxt = data[j + 2].split()[0:3]
        else:
            nxt = ["", "", "Ellie"]
        if not prev or prev[-1] != "Participant" or holding_start is None:
            holding_start = start
        if nxt and nxt[-1] == "Participant":
            continue
        inter.append([holding_start, stop]); holding_start = None
    if not inter:
        raise ValueError(f"{trial}: no Participant intervals")
    return inter


def extract_participant_wav(raw_wav: Path, transcript_path: Path, output_wav: Path) -> dict:
    """Create the ComParE input WAV using the author's exact time-to-sample slicing.

    The author repository does not release the helper that serialised its
    concatenated Participant array to *_P_audio_data.wav. DAIC-WOZ PCM16
    samples are therefore copied losslessly instead of decoding/re-encoding.
    """
    with wave.open(str(raw_wav), "rb") as src:
        channels, width, sr = src.getnchannels(), src.getsampwidth(), src.getframerate()
        if channels != 1 or width != 2:
            raise ValueError(f"{raw_wav}: expected mono PCM16 DAIC-WOZ WAV; got channels={channels}, sample_width={width}")
        signal = np.frombuffer(src.readframes(src.getnframes()), dtype="<i2")
    chunks = []
    for start, stop in author_intervals(transcript_path):
        a, b = int(start * sr), int(stop * sr)
        a, b = max(a, 0), min(b, len(signal))
        if b > a:
            chunks.append(signal[a:b])
    if not chunks:
        raise ValueError(f"{raw_wav}: no usable Participant audio")
    patient = np.hstack(chunks).astype("<i2", copy=False)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as dst:
        dst.setnchannels(1); dst.setsampwidth(2); dst.setframerate(sr); dst.writeframes(patient.tobytes())
    return {"sample_rate": int(sr), "samples": int(patient.size), "seconds": float(patient.size / sr), "wav_bridge": "lossless PCM16 sample copy using author time indices"}

def compare16_matrix(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, sep=";")
    if "name" not in df.columns or "frameTime" not in df.columns:
        raise ValueError(f"{csv_path}: expected openSMILE columns name + frameTime")
    mat = df.drop(columns=["name", "frameTime"]).to_numpy(dtype=np.float32)
    if mat.shape[1] != FREQ_BINS:
        raise ValueError(f"{csv_path}: expected {FREQ_BINS} ComParE16 columns, got {mat.shape[1]}")
    # Exact author database path: flatten time x 130 in C order, then loader reshapes to 130 x time.
    return mat.reshape(-1).reshape(FREQ_BINS, -1)


def segment_feature(x: np.ndarray, frames: int = SEGMENT_FRAMES) -> np.ndarray:
    n = (x.shape[1] + frames - 1) // frames
    out = np.zeros((n, x.shape[0], frames), dtype=np.float32)
    for i in range(n):
        q = x[:, i * frames:(i + 1) * frames]
        out[i, :, :q.shape[1]] = q
    return out


def crop_author_train(x: np.ndarray, rng: random.Random, frames: int = AUTHOR_TRAIN_CROP_FRAMES) -> np.ndarray:
    if x.shape[1] <= frames:
        return x
    start = rng.randint(0, x.shape[1] - frames)
    return x[:, start:start + frames]


def load_author_stats(author_root: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    path = author_root / RUN4_REL / "data_saver.pickle"
    if not path.exists():
        raise FileNotFoundError(f"Missing author normalization artifact: {path}")
    with path.open("rb") as f:
        saved = pickle.load(f)
    if "mean" not in saved or "std" not in saved:
        raise KeyError("data_saver.pickle has no mean/std")
    mean, std = np.asarray(saved["mean"], dtype=np.float32), np.asarray(saved["std"], dtype=np.float32)
    mean, std = mean.reshape(FREQ_BINS, 1), std.reshape(FREQ_BINS, 1)
    if np.any(std == 0):
        raise ValueError("Author std contains zeros")
    return mean, std, path


def normalise_segments(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :, :]) / std[None, :, :]).astype(np.float32)


class ConvBlock1d(nn.Module):
    def __init__(self):
        super().__init__(); self.conv1 = nn.Conv1d(130, 256, 3, 1, 1); self.bn1 = nn.BatchNorm1d(256); self.relu = nn.ReLU()
    def forward(self, x): return self.relu(self.bn1(self.conv1(x)))


class FullyConnected(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(256, 1); self.act = nn.Sigmoid()
    def forward(self, x): return self.act(self.fc(x))


class CustomComparE16(nn.Module):
    """State-dict-compatible implementation of the author's released model."""
    def __init__(self):
        super().__init__(); self.conv = ConvBlock1d(); self.pool = nn.MaxPool1d(3, 3); self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, bidirectional=False); self.fc1 = FullyConnected()

    def forward_logits(self, x):
        x = self.drop(self.pool(self.conv(x))); x = torch.transpose(x, 1, 2); x, hidden = self.lstm(x)
        logit = self.fc1.fc(x[:, -1, :]); prob = torch.sigmoid(logit)
        return logit.squeeze(1), prob.squeeze(1), hidden[0][-1]

    def forward(self, x):
        _, prob, emb = self.forward_logits(x); return prob.view(-1, 1), emb


def load_author_model(author_root: Path, device: torch.device) -> tuple[CustomComparE16, dict, Path]:
    ckpt_path = author_root / RUN4_REL / CHECKPOINT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing released run #4 checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state.get("model_state_dict", state))
    if not isinstance(sd, dict):
        raise TypeError("Author checkpoint does not contain a state_dict")
    if sd and all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model = CustomComparE16()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    return model, state, ckpt_path


def infer_segments(model: CustomComparE16, segments: np.ndarray, device: torch.device, batch_size: int = 64) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(segments), batch_size):
            x = torch.from_numpy(segments[start:start + batch_size]).float().to(device)
            logit, prob, _ = model.forward_logits(x)
            l, p = logit.cpu().numpy(), prob.cpu().numpy()
            if not np.allclose(1.0 / (1.0 + np.exp(-l)), p, atol=1e-6):
                raise AssertionError("Probability != sigmoid(logit)")
            for j, (z, q) in enumerate(zip(l, p), start=start):
                rows.append({"segment": j, "logit": float(z), "prob": float(q), "segment_hard": int(np.rint(q))})
    return pd.DataFrame(rows)


def aggregate_segments(segment_rows: pd.DataFrame, participant_id: int, label: int) -> dict:
    probs = segment_rows.prob.to_numpy(dtype=float)
    hard = segment_rows.segment_hard.to_numpy(dtype=int)
    vote_fraction = float(hard.mean())
    author_pred = int(np.rint(vote_fraction))
    kd_prob = float(probs.mean())
    eps = 1e-6; q = float(np.clip(kd_prob, eps, 1 - eps)); kd_logit = float(np.log(q / (1 - q)))
    return {
        "participant_id": int(participant_id), "label": int(label), "n_segments": int(len(segment_rows)),
        "author_vote_fraction": vote_fraction, "author_prediction": author_pred,
        "kd_probability": kd_prob, "kd_logit": kd_logit,
    }


def metric_dict(frame: pd.DataFrame) -> dict:
    y = frame.label.to_numpy(dtype=int); pred = frame.author_prediction.to_numpy(dtype=int); prob = frame.kd_probability.to_numpy(dtype=float)
    out = {
        "n": int(len(frame)), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(classification_report(y, pred, output_dict=True, zero_division=0)["macro avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
        "segment_hardening": "numpy.rint(probability); author majority vote",
        "nominal_threshold": 0.5,
    }
    out["auroc_soft_mean_probability"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None
    return out


def load_feature_manifest(features: Path, split: str) -> pd.DataFrame:
    manifest = pd.read_csv(features / "participant_manifest.csv")
    if "split" not in manifest.columns:
        raise ValueError("participant_manifest.csv needs split")
    frame = manifest[manifest.split.str.lower() == split.lower()].copy().sort_values("participant_id")
    expected = 107 if split.lower() == "train" else 34 if split.lower() == "dev" else None
    if expected is None:
        raise ValueError("Only TRAIN and DEV are allowed in this teacher pipeline")
    if len(frame) != expected:
        raise AssertionError(f"Expected {split.upper()}={expected}; got {len(frame)}")
    if split.lower() == "dev" and 440 in set(frame.participant_id):
        raise AssertionError("DEV contains excluded participant 440")
    return frame
