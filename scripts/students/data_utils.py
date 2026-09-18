"""Shared DAIC-WOZ segment/student feature utilities for RA-PDS-KD.

This module deliberately separates TRAIN/DEV preparation from final TEST use.
Student vocabulary and audio normalization are fit on TRAIN only.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json, math, os, re, unicodedata

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SEGMENTER_VERSION = 'aligned_turn_partition_v1'
PAD_ID = 0
UNK_ID = 1


def clean_text(value):
    value = unicodedata.normalize('NFC', str(value)).replace('\x00', ' ')
    return re.sub(r'\s+', ' ', value).strip()


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_split(data_root: Path, filename: str, require_labels: bool):
    path = data_root / 'metadata' / filename
    df = pd.read_csv(path); df.columns = df.columns.str.strip().str.lower()
    assert 'participant_id' in df, f'Missing participant_id in {filename}'
    ids = pd.to_numeric(df['participant_id'], errors='raise')
    assert ids.notna().all() and (ids % 1 == 0).all() and not ids.duplicated().any()
    out = pd.DataFrame({'participant_id': ids.astype(int)})
    if require_labels:
        assert 'phq8_binary' in df, f'{filename} has no phq8_binary labels'
        y = pd.to_numeric(df['phq8_binary'], errors='raise')
        assert y.isin([0, 1]).all(); out['label'] = y.astype(int)
    elif 'phq8_binary' in df:
        y = pd.to_numeric(df['phq8_binary'], errors='coerce')
        if y.notna().all() and y.isin([0, 1]).all(): out['label'] = y.astype(int)
    return out


def index_sources(data_root: Path):
    audio_index, text_index = {}, {}
    for base, dirs, files in os.walk(data_root):
        dirs[:] = [d for d in dirs if d not in {'experiments', 'processed', '.git', '__pycache__'}]
        for name in files:
            a = re.fullmatch(r'(\d+)_AUDIO\.wav', name, re.I)
            t = re.fullmatch(r'(\d+)_TRANSCRIPT\.(?:csv|txt)', name, re.I)
            if a: audio_index.setdefault(int(a.group(1)), []).append(Path(base) / name)
            if t: text_index.setdefault(int(t.group(1)), []).append(Path(base) / name)
    return audio_index, text_index


def read_turns(path):
    df = pd.read_csv(path, sep='\t'); df.columns = df.columns.str.strip().str.lower()
    if not {'start_time', 'stop_time', 'speaker', 'value'} <= set(df.columns):
        df = pd.read_csv(path, sep=None, engine='python'); df.columns = df.columns.str.strip().str.lower()
    required = {'start_time', 'stop_time', 'speaker', 'value'}
    assert required <= set(df.columns), f'Unexpected transcript columns: {list(df.columns)}'
    df = df.loc[df.speaker.astype(str).str.strip().str.lower().eq('participant')].copy()
    df['value'] = df.value.fillna('').map(clean_text)
    df = df.loc[df.value.ne('') & ~df.value.str.contains('scrubbed|redacted', case=False, regex=True)]
    for c in ['start_time', 'stop_time']: df[c] = pd.to_numeric(df[c], errors='raise')
    df = df.loc[(df.start_time >= 0) & (df.stop_time > df.start_time)].sort_values('start_time').reset_index(drop=True)
    assert len(df), f'No usable participant turns: {path}'
    return df


def load_teacher_protocol(data_root: Path, seed=103):
    path = data_root / 'experiments' / 'teachers' / 'segment_level_v1' / f'seed_{seed}' / 'protocol.json'
    assert path.exists(), f'Run segment teachers first: {path}'
    protocol = json.loads(path.read_text())
    cache_root = Path(protocol['cache_root'])
    cfg = protocol['config']
    cache_cfg = json.loads((cache_root / 'config.json').read_text())
    assert cfg['segmenter'] == SEGMENTER_VERSION
    return protocol, cache_root, cfg, cache_cfg


def make_aligned_segments(pid, turns, tokenizer, cfg):
    rows = []
    for ti, row in enumerate(turns.itertuples()):
        token_ids = tokenizer.encode(row.value, add_special_tokens=False)
        if not token_ids: continue
        duration = float(row.stop_time - row.start_time)
        n = max(1, math.ceil(duration / cfg['max_audio_seconds']), math.ceil(len(token_ids) / cfg['max_text_tokens']))
        for k in range(n):
            start = float(row.start_time) + duration * k / n
            stop = float(row.start_time) + duration * (k + 1) / n
            if stop - start < cfg['min_audio_seconds']: continue
            a = round(len(token_ids) * k / n); b = round(len(token_ids) * (k + 1) / n)
            piece = (token_ids[a:b] if b > a else token_ids)[:cfg['max_text_tokens']]
            text = tokenizer.decode(piece, skip_special_tokens=True).strip() or row.value
            rows.append({'segment_id': f'{int(pid)}_{ti:04d}_{k:02d}', 'start': start, 'stop': stop,
                         'text': text, 'source_turn': int(ti), 'part': int(k), 'parts': int(n)})
    assert rows, f'{pid}: no aligned segments'
    if len(rows) > cfg['max_segments_per_participant']:
        idx = np.linspace(0, len(rows) - 1, cfg['max_segments_per_participant'], dtype=int)
        rows = [rows[i] for i in idx]
    return rows


def simple_tokens(text):
    return re.findall(r"[A-Za-z0-9']+|[^\w\s]", clean_text(text).lower())


def build_vocab(texts, max_vocab=10000, min_freq=2):
    counts = Counter()
    for text in texts: counts.update(simple_tokens(text))
    items = sorted(((tok, n) for tok, n in counts.items() if n >= min_freq), key=lambda x: (-x[1], x[0]))
    vocab = {'<PAD>': PAD_ID, '<UNK>': UNK_ID}
    for tok, _ in items[:max(0, max_vocab - len(vocab))]: vocab[tok] = len(vocab)
    return vocab


def encode_text(text, vocab, max_tokens=64):
    ids = [vocab.get(tok, UNK_ID) for tok in simple_tokens(text)][:max_tokens]
    out = np.full(max_tokens, PAD_ID, dtype=np.int64)
    if ids: out[:len(ids)] = ids
    return out


def corrupt_text_ids(ids, vocab_size, rate, rng):
    x = np.asarray(ids, dtype=np.int64).copy()
    pos = np.flatnonzero(x != PAD_ID)
    if not len(pos) or rate <= 0: return x
    n = max(1, int(round(len(pos) * rate))); chosen = rng.choice(pos, size=min(n, len(pos)), replace=False)
    delete = []
    for p in chosen:
        op = int(rng.integers(0, 3))
        if op == 0: x[p] = UNK_ID
        elif op == 1: delete.append(int(p))
        else: x[p] = int(rng.integers(2, max(3, vocab_size)))
    if delete:
        kept = [int(v) for i, v in enumerate(x) if i not in set(delete) and v != PAD_ID]
        x[:] = PAD_ID; x[:min(len(x), len(kept))] = kept[:len(x)]
    return x


def load_audio_segment(path, start, stop, target_sr=16000):
    with sf.SoundFile(path) as wav:
        sr = int(wav.samplerate); wav.seek(min(round(start * sr), len(wav)))
        x = wav.read(max(1, round((stop - start) * sr)), dtype='float32', always_2d=True).mean(axis=1)
    assert len(x) and np.isfinite(x).all()
    if sr != target_sr:
        g = math.gcd(sr, target_sr); x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return x.astype(np.float32)


def add_gaussian_noise_snr(x, snr_db, rng):
    x = np.asarray(x, dtype=np.float32)
    signal_power = float(np.mean(x * x))
    if signal_power <= 1e-12: return x.copy()
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=len(x)).astype(np.float32)
    return (x + noise).astype(np.float32)


def logmel_summary(x, sr=16000, n_mels=64):
    import librosa
    mel = librosa.feature.melspectrogram(y=np.asarray(x, dtype=np.float32), sr=sr, n_mels=n_mels,
                                         n_fft=512, hop_length=160, win_length=400, power=2.0)
    logmel = librosa.power_to_db(np.maximum(mel, 1e-10), ref=np.max)
    feat = np.concatenate([logmel.mean(axis=1), logmel.std(axis=1)]).astype(np.float32)
    assert feat.shape == (2 * n_mels,) and np.isfinite(feat).all()
    return feat


def fit_standardizer(x):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0).astype(np.float32); std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(x, mean, std):
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def quality_from_snr(snr_db):
    return float(np.clip(float(snr_db) / 20.0, 0.05, 1.0))


def quality_from_text_noise(rate):
    return float(np.clip(1.0 - float(rate), 0.05, 1.0))
