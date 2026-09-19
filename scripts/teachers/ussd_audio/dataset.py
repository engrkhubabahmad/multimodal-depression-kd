from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import AUTHOR_TRAIN_CROP_FRAMES, SEGMENT_FRAMES


@lru_cache(maxsize=32)
def _mmap(path: str):
    return np.load(path, mmap_mode="r")


class CroppedSegmentDataset(Dataset):
    """All TRAIN participants, deterministic per-epoch crop, segment-level batches."""

    def __init__(self, frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray, seed: int, epoch: int):
        self.mean = mean.astype(np.float32); self.std = std.astype(np.float32); self.items = []
        rng = random.Random(seed + epoch)
        for row in frame.itertuples(index=False):
            path = str(Path(row.feature_path)); x = _mmap(path)
            if x.ndim != 2 or x.shape[0] != self.mean.shape[0]:
                raise ValueError(f"{path}: expected [130,time], got {x.shape}")
            width = int(x.shape[1]); crop = min(width, AUTHOR_TRAIN_CROP_FRAMES)
            start = 0 if width <= crop else rng.randint(0, width - crop)
            n = (crop + SEGMENT_FRAMES - 1) // SEGMENT_FRAMES
            for j in range(n):
                self.items.append((path, start + j * SEGMENT_FRAMES, min(SEGMENT_FRAMES, crop - j * SEGMENT_FRAMES), int(row.label), int(row.participant_id)))
        if not self.items:
            raise ValueError("No TRAIN segments were created")

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        path, start, valid, label, pid = self.items[index]; x = _mmap(path)
        segment = np.zeros((x.shape[0], SEGMENT_FRAMES), dtype=np.float32)
        if valid > 0: segment[:, :valid] = np.asarray(x[:, start:start + valid], dtype=np.float32)
        segment = (segment - self.mean) / self.std
        return torch.from_numpy(segment), torch.tensor(label, dtype=torch.float32), pid
