"""Prepare short Step-Audio2 chunks for a bounded canonical TRAIN/DEV pilot."""
from __future__ import annotations
import argparse
import json
import os
import re
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
try:
    from tqdm.auto import tqdm
except ImportError:  # Keep script usable in minimal environments; Colab shows the progress bar.
    tqdm = lambda iterable, **_: iterable

from .split import digest

MODEL = 'stepfun-ai/Step-Audio-2-mini-Think'
ADAPTER = 'maimai11/woz'
PROMPT = ('Listen to this interview segment. Is the participant depressed? '
          'Answer Yes or No first, then briefly explain.')
SAMPLE_RATE = 16000


def audio_index(root: Path) -> dict[int, list[Path]]:
    """Walk the Drive tree once; repeated rglob per participant caused long setup."""
    found: dict[int, list[Path]] = {}
    for base, _, names in os.walk(root):
        for name in names:
            match = re.fullmatch(r'(\d+)_AUDIO\.wav', name, flags=re.IGNORECASE)
            if match:
                found.setdefault(int(match.group(1)), []).append(Path(base) / name)
    return found


def select_rows(manifest: pd.DataFrame, limit: int) -> pd.DataFrame:
    selected = manifest.loc[manifest.split.isin(['train', 'val'])].copy()
    if limit <= 0 or limit >= len(selected):
        return selected.sort_values(['split', 'participant_id']).reset_index(drop=True)
    # A small, deterministic smoke test: include one example per class per split.
    pilot = (selected.sort_values('participant_id').groupby(['split', 'label'], sort=True)
             .head(1).sort_values(['split', 'label', 'participant_id']))
    if len(pilot) > limit:
        pilot = pilot.head(limit)
    if len(pilot) < limit:
        remaining = selected.loc[~selected.participant_id.isin(pilot.participant_id)]
        pilot = pd.concat([pilot, remaining.sort_values('participant_id').head(limit-len(pilot))])
    return pilot.sort_values(['split', 'participant_id']).reset_index(drop=True)


def chunk_audio(source: Path, cache: Path, pid: int, seconds: int) -> list[tuple[Path, float]]:
    rate, header_audio = wavfile.read(source, mmap=True)
    signature = {'source_size': source.stat().st_size, 'source_frames': len(header_audio),
                 'source_rate': rate, 'source_mtime_ns': source.stat().st_mtime_ns,
                 'seconds': seconds, 'target_rate': SAMPLE_RATE}
    participant_cache = cache / str(pid)
    marker = participant_cache / 'complete.json'
    if marker.is_file():
        try:
            old = json.loads(marker.read_text())
            paths = [participant_cache / x['file'] for x in old['chunks']]
            if old['signature'] == signature and all(p.is_file() for p in paths):
                return [(p, float(x['duration_seconds'])) for p, x in zip(paths, old['chunks'])]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    rate, audio = wavfile.read(source)
    if audio.dtype.kind == 'u':
        midpoint = (np.iinfo(audio.dtype).max + 1) / 2
        audio = (audio.astype('float32') - midpoint) / midpoint
    elif audio.dtype.kind == 'i':
        limits = np.iinfo(audio.dtype)
        audio = audio.astype('float32') / max(abs(limits.min), limits.max)
    else:
        audio = audio.astype('float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        common = gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // common, rate // common).astype('float32')
    participant_cache.mkdir(parents=True, exist_ok=True)
    chunk_frames = seconds * SAMPLE_RATE
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_frames)):
        clip = audio[start:start + chunk_frames]
        if not len(clip):
            continue
        path = participant_cache / f'{pid}_{i:04d}.wav'
        pcm16 = (np.clip(clip, -1.0, 1.0) * 32767).astype('int16')
        wavfile.write(path, SAMPLE_RATE, pcm16)
        duration = len(clip) / SAMPLE_RATE
        chunks.append({'file': path.name, 'duration_seconds': duration})
    if not chunks:
        raise ValueError(f'Empty audio: {source}')
    marker.write_text(json.dumps({'signature': signature, 'chunks': chunks}, indent=2) + '\n')
    return [(participant_cache / x['file'], float(x['duration_seconds'])) for x in chunks]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daic-root', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--adapter-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk-cache', type=Path, default=Path('/content/stepaudio2_chunks_20s'))
    p.add_argument('--chunk-seconds', type=int, default=20)
    p.add_argument('--max-participants', type=int, default=4,
                   help='Pilot count; use 0 to prepare all 141 TRAIN/DEV participants.')
    a = p.parse_args(argv)
    if a.chunk_seconds < 1:
        raise ValueError('--chunk-seconds must be positive')
    marker = json.loads((a.split_dir / 'complete.json').read_text())
    manifest_path = a.split_dir / 'manifest.csv'
    if digest(manifest_path) != marker['manifest_sha256']:
        raise ValueError('Split manifest changed')
    manifest = pd.read_csv(manifest_path)
    counts = manifest.split.value_counts().to_dict()
    if counts != {'train': 107, 'val': 34, 'student_test': 47}:
        raise ValueError(f'Expected canonical TRAIN/DEV/TEST=107/34/47, got {counts}')
    audit = json.loads(a.adapter_audit.read_text())
    if audit.get('adapter') != ADAPTER or audit.get('split_sha256') != digest(manifest_path):
        raise ValueError('Adapter audit/split mismatch')

    selected = select_rows(manifest, a.max_participants)
    paths_by_id = audio_index(a.daic_root)
    missing = [int(pid) for pid in selected.participant_id if len(paths_by_id.get(int(pid), [])) != 1]
    if missing:
        raise ValueError('Audio coverage must be exactly one WAV per selected participant; problematic IDs: '
                         + ', '.join(map(str, missing[:20])))

    records, index = [], []
    for row in tqdm(selected.itertuples(index=False), total=len(selected),
                    desc='Chunking Step-Audio2 pilot', colour='green'):
        pid = int(row.participant_id)
        source = paths_by_id[pid][0]
        for chunk_id, (chunk_path, duration) in enumerate(
                chunk_audio(source, a.chunk_cache, pid, a.chunk_seconds)):
            records.append({'messages': [{'role': 'user', 'content': '<audio>' + PROMPT}],
                            'audios': [str(chunk_path)]})
            index.append({'participant_id': pid, 'label': int(row.label), 'split': row.split,
                          'source_split': row.source_split, 'chunk_id': chunk_id,
                          'duration_seconds': duration, 'chunk_path': str(chunk_path),
                          'source_audio': str(source)})

    a.output.mkdir(parents=True, exist_ok=True)
    data = a.output / 'chunks.jsonl'
    idx = a.output / 'chunk_index.csv'
    text = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records)
    frame = pd.DataFrame(index)
    if data.exists() and data.read_text() != text:
        raise ValueError(f'Existing dataset differs: {data}; use a fresh output folder')
    if idx.exists() and not pd.read_csv(idx).equals(frame):
        raise ValueError(f'Existing chunk map differs: {idx}; use a fresh output folder')
    data.write_text(text)
    frame.to_csv(idx, index=False)
    prov = {'adapter': ADAPTER, 'adapter_revision': audit['revision'],
            'split_sha256': digest(manifest_path), 'participants_selected': len(selected),
            'chunks': len(frame), 'chunk_seconds_max': a.chunk_seconds,
            'audio_handling': 'mono, resampled to 16 kHz when needed, non-overlapping WAV chunks',
            'prompt': PROMPT, 'author_config_match': {'max_length': 8000,
                'response_prefix': None, 'max_new_tokens': 64, 'temperature': 0.0},
            'author_data_preprocessing_source': 'not published; chunking is an explicit exploratory approximation',
            'pilot': bool(a.max_participants > 0 and len(selected) < 141),
            'full_run_hint': 'After reviewing pilot generations, rerun with --max-participants 0 and a fresh output directory.',
            'test_opened': False}
    (a.output / 'provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    command = ['swift', 'infer', '--model', MODEL, '--model_type', 'step_audio2_mini',
        '--adapters', ADAPTER, '--use_hf', 'true', '--load_args', 'false',
        '--val_dataset', str(data), '--result_path', str(a.output / 'chunk_predictions.jsonl'),
        '--infer_backend', 'transformers', '--max_batch_size', '1', '--temperature', '0',
        '--max_new_tokens', '64', '--max_length', '8000', '--add_non_thinking_prefix', 'true']
    print(f'Prepared {len(frame)} chunks for {len(selected)} TRAIN/DEV participants; TEST untouched.')
    print(f'Chunk WAV cache: {a.chunk_cache}')
    print('This is an exploratory approximation; author preprocessing files are not public.')
    return command


if __name__ == '__main__':
    main()
