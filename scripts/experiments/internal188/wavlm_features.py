"""Freeze externally pretrained WavLM Base+; cache TRAIN/VAL participant speech vectors."""
import argparse
import json
import math
import os
import shutil
import tempfile
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from huggingface_hub import model_info

from scripts.teachers.ussd_audio.common import extract_participant_wav, one
from .split import digest
from .train_text_teacher import verified_split


MODEL = 'microsoft/wavlm-base-plus'
SAMPLE_RATE = 16000
SECONDS = 6
CHUNKS = 8


def participant_audio(root, pid, transcript):
    with tempfile.TemporaryDirectory(prefix=f'daic_wavlm_{pid}_', dir='/content') as temp:
        wav = Path(temp) / 'participant.wav'
        raw = one(root, f'{pid}_AUDIO.wav')
        extract_participant_wav(raw, transcript, wav)
        with wave.open(str(wav), 'rb') as src:
            sr = src.getframerate()
            if src.getnchannels() != 1 or src.getsampwidth() != 2:
                raise ValueError(f'{pid}: participant WAV must be mono PCM16')
            audio = np.frombuffer(src.readframes(src.getnframes()), dtype='<i2').astype(np.float32) / 32768.
        if sr != SAMPLE_RATE:
            divisor = math.gcd(sr, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
        return audio, digest(raw)


def windows(audio):
    size = SAMPLE_RATE * SECONDS
    if len(audio) < size: audio = np.pad(audio, (0, size - len(audio)))
    limit = len(audio) - size
    starts = [round(limit * (j + .5) / CHUNKS) for j in range(CHUNKS)]
    return np.stack([audio[start:start + size] for start in starts])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daic-root', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--reuse-from', type=Path,
                   help='Completed earlier WavLM cache; copy matching TRAIN/VAL embeddings only')
    a = p.parse_args(argv)
    if a.batch_size < 1: raise ValueError('Batch size must be positive')
    manifest = verified_split(a.split_dir, a.coverage_dir)
    selected = manifest.loc[manifest.split.isin(['train', 'val'])].sort_values('participant_id')
    coverage = pd.read_csv(a.coverage_dir / 'participant_manifest.csv').set_index('participant_id')
    previous = None
    if a.reuse_from:
        previous_audit = json.loads((a.reuse_from / 'audit.json').read_text())
        if (previous_audit['model'] != MODEL or previous_audit['chunks_per_participant'] != CHUNKS
                or previous_audit['seconds_per_chunk'] != SECONDS or previous_audit['sample_rate'] != SAMPLE_RATE
                or previous_audit['embedding_dim'] != 1536):
            raise ValueError('Prior WavLM cache uses another extraction protocol')
        revision = previous_audit['revision']
        previous = pd.read_csv(a.reuse_from / 'participant_manifest.csv').set_index('participant_id')
        if previous.index.duplicated().any(): raise ValueError('Prior WavLM cache contains duplicate IDs')
    else:
        revision = model_info(MODEL).sha
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = model = None
    a.output.mkdir(parents=True, exist_ok=True)
    cache = a.output / 'participants'; cache.mkdir(exist_ok=True)
    rows = []
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc='WavLM TRAIN/VAL', colour='green'):
        pid = int(row.participant_id)
        transcript = Path(coverage.loc[pid, 'transcript_path'])
        transcript_sha = digest(transcript)
        path = cache / f'{pid}.npz'
        if not path.is_file() and previous is not None and pid in previous.index:
            prior = previous.loc[pid]
            source = Path(prior.feature_path)
            with np.load(source, allow_pickle=False) as z:
                if (str(z['revision']) != revision or str(z['transcript_sha256']) != transcript_sha
                        or z['embedding'].shape != (1536,) or not np.isfinite(z['embedding']).all()):
                    raise ValueError(f'{pid}: prior WavLM embedding differs')
            shutil.copy2(source, path)
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                if (str(z['revision']) != revision or str(z['transcript_sha256']) != transcript_sha
                        or z['embedding'].shape != (1536,) or not np.isfinite(z['embedding']).all()):
                    raise ValueError(f'{pid}: cached WavLM embedding or transcript changed')
                raw_sha = str(z['raw_audio_sha256'])
        else:
            if model is None:
                extractor = AutoFeatureExtractor.from_pretrained(MODEL, revision=revision)
                model = AutoModel.from_pretrained(MODEL, revision=revision).to(device).eval()
                model.requires_grad_(False)
                if int(model.config.hidden_size) != 768: raise ValueError('Unexpected WavLM Base+ embedding dimension')
            audio, raw_sha = participant_audio(a.daic_root, pid, transcript)
            chunks = windows(audio)
            vectors = []
            with torch.inference_mode():
                for start in range(0, CHUNKS, a.batch_size):
                    batch = extractor([x for x in chunks[start:start + a.batch_size]],
                                      sampling_rate=SAMPLE_RATE, return_tensors='pt', padding=True)
                    hidden = model(batch['input_values'].to(device)).last_hidden_state
                    vectors.append(hidden.mean(1).cpu().numpy())
            vectors = np.concatenate(vectors, axis=0)
            embedding = np.concatenate([vectors.mean(0), vectors.std(0)]).astype(np.float32)
            if embedding.shape != (1536,) or not np.isfinite(embedding).all():
                raise ValueError(f'{pid}: invalid embedding')
            with tempfile.NamedTemporaryFile(dir=cache, suffix='.npz', delete=False) as pending:
                pending_path = Path(pending.name)
            try:
                np.savez_compressed(pending_path, embedding=embedding, revision=revision,
                                    transcript_sha256=transcript_sha, raw_audio_sha256=raw_sha)
                os.replace(pending_path, path)
            finally:
                pending_path.unlink(missing_ok=True)
        rows.append({'participant_id': pid, 'label': int(row.label), 'split': row.split,
                     'feature_path': str(path), 'raw_audio_sha256': raw_sha,
                     'transcript_sha256': transcript_sha})
    data = pd.DataFrame(rows)
    target = a.output / 'participant_manifest.csv'
    if target.exists() and not pd.read_csv(target).equals(data):
        raise ValueError('Existing WavLM manifest differs; do not overwrite')
    data.to_csv(target, index=False)
    (a.output / 'audit.json').write_text(json.dumps({'model': MODEL, 'revision': revision,
        'split_sha256': digest(a.split_dir / 'manifest.csv'),
        'coverage_sha256': digest(a.coverage_dir / 'participant_manifest.csv'),
        'sample_rate': SAMPLE_RATE, 'seconds_per_chunk': SECONDS, 'chunks_per_participant': CHUNKS,
        'embedding_dim': 1536, 'train_count': 113, 'val_count': 37,
        'reused_from': str(a.reuse_from) if a.reuse_from else None,
        'internal_test_media_opened': False, 'daic_finetuning': False}, indent=2) + '\n')
    print('Cached 150 TRAIN/VAL participants; WavLM revision:', revision)


if __name__ == '__main__': main()
