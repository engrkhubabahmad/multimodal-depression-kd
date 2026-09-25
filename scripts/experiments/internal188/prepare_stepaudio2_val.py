"""Prepare validation-only Step-Audio2 inference in bounded speech windows."""
import argparse
import json
import math
import shlex
import wave
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from scripts.teachers.ussd_audio.common import extract_participant_wav, one
from .split import digest
from .train_text_teacher import verified_split

MODEL = 'stepfun-ai/Step-Audio-2-mini-Think'
ADAPTER = 'maimai11/woz'
PROMPT = 'Listen only to the participant speech. Is the participant depressed? Answer Yes or No.'
DEFAULT_CHUNK_SECONDS = 20


def make_audio_chunks(source: Path, output_dir: Path, participant_id: int, seconds: int):
    """Write contiguous, lossless PCM16 windows and return their timing metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with wave.open(str(source), 'rb') as wav:
        channels, width, rate, total = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        if channels != 1 or width != 2:
            raise ValueError(f'{source}: expected mono PCM16 WAV; got channels={channels}, width={width}')
        window = rate * seconds
        for chunk_id, start in enumerate(range(0, total, window)):
            count = min(window, total - start)
            path = output_dir / f'{participant_id}_chunk_{chunk_id:04d}.wav'
            wav.setpos(start)
            frames = wav.readframes(count)
            if len(frames) != count * channels * width:
                raise IOError(f'{source}: incomplete audio read at frame {start}')
            if path.exists():
                with wave.open(str(path), 'rb') as cached:
                    valid = (cached.getnchannels() == channels and cached.getsampwidth() == width
                             and cached.getframerate() == rate and cached.getnframes() == count
                             and cached.readframes(count) == frames)
                if not valid:
                    raise ValueError(f'Cached audio window differs from source: {path}')
            else:
                with wave.open(str(path), 'wb') as out:
                    out.setnchannels(channels); out.setsampwidth(width); out.setframerate(rate); out.writeframes(frames)
            rows.append({'participant_id': participant_id, 'chunk_id': chunk_id,
                         'chunk_start_seconds': start / rate, 'duration_seconds': count / rate,
                         'audio_path': str(path), 'audio_sha256': digest(path)})
    if not rows:
        raise ValueError(f'{source}: no audio frames')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daic-root', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path, required=True)
    p.add_argument('--adapter-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot-only', action='store_true', help='Prepare one VAL participant to check runtime')
    p.add_argument('--chunk-seconds', type=int, default=DEFAULT_CHUNK_SECONDS)
    a = p.parse_args(argv)
    if a.chunk_seconds < 1 or a.chunk_seconds > 25:
        raise ValueError('chunk-seconds must be between 1 and 25; use 20 on a T4')
    manifest = verified_split(a.split_dir, a.coverage_dir)
    val = manifest.loc[manifest.split.eq('val')].sort_values('participant_id')
    if len(val) != 37 or val.participant_id.nunique() != 37:
        raise ValueError(f'Expected the frozen 37-person VAL split; found {len(val)} rows')
    audit = json.loads(a.adapter_audit.read_text())
    if audit['split_sha256'] != digest(a.split_dir / 'manifest.csv') or audit['adapter'] != ADAPTER:
        raise ValueError('Adapter audit belongs to another split or model')
    if a.pilot_only:
        val = val.head(1)
    coverage = pd.read_csv(a.coverage_dir / 'participant_manifest.csv').set_index('participant_id')
    a.output.mkdir(parents=True, exist_ok=True)
    audio_dir = a.output / 'participant_audio'; audio_dir.mkdir(exist_ok=True)
    chunk_dir = a.output / f'audio_chunks_{a.chunk_seconds}s'; chunk_dir.mkdir(exist_ok=True)
    dataset = a.output / f'val_chunks_{a.chunk_seconds}s.jsonl'
    index = a.output / f'val_chunks_{a.chunk_seconds}s_index.csv'
    records = []; references = []
    for row in tqdm(val.itertuples(index=False), total=len(val), desc='Step-Audio2 VAL windows', colour='green'):
        pid = int(row.participant_id)
        transcript = Path(coverage.loc[pid, 'transcript_path'])
        if not transcript.is_file():
            raise FileNotFoundError(transcript)
        raw = one(a.daic_root, f'{pid}_AUDIO.wav')
        wav = audio_dir / f'{pid}.wav'
        if not wav.exists():
            extract_participant_wav(raw, transcript, wav)
        if not wav.is_file() or wav.stat().st_size < 44:
            raise ValueError(f'Empty speech WAV: {pid}')
        transcript_sha = digest(transcript)
        raw_audio_sha = digest(raw)
        pieces = make_audio_chunks(wav, chunk_dir, pid, a.chunk_seconds)
        for piece in pieces:
            records.append({'messages': [{'role': 'user', 'content': '<audio>' + PROMPT}],
                            'audios': [piece['audio_path']]})
            references.append({**piece, 'label': int(row.label), 'split': 'val',
                               'transcript_sha256': transcript_sha, 'raw_audio_sha256': raw_audio_sha})
    content = ''.join(json.dumps(r) + '\n' for r in records)
    reference = pd.DataFrame(references).sort_values(['participant_id', 'chunk_id']).reset_index(drop=True)
    if dataset.exists() and dataset.read_text() != content:
        raise ValueError(f'Dataset differs; refusing overwrite: {dataset}')
    if index.exists() and not pd.read_csv(index).equals(reference):
        raise ValueError(f'Index differs; refusing overwrite: {index}')
    dataset.write_text(content); reference.to_csv(index, index=False)
    result = a.output / f'val_chunks_{a.chunk_seconds}s_direct_predictions.jsonl'
    command = ['swift', 'infer', '--model', MODEL, '--model_type', 'step_audio2_mini',
               '--adapters', ADAPTER, '--use_hf', 'true', '--load_args', 'false',
               '--val_dataset', str(dataset), '--result_path', str(result),
               '--infer_backend', 'transformers', '--max_batch_size', '1',
               '--temperature', '0', '--max_new_tokens', '8', '--response_prefix', '<英语>']
    print(f'Prepared {len(val)} VAL participants as {len(reference)} non-overlapping windows of at most {a.chunk_seconds}s; internal TEST unopened.')
    print('Run with a Step-Audio2 compatible GPU and ms-swift installed:')
    print(shlex.join(command))
    print('Window map:', index)
    print('Direct-answer prefix: <英语>; generation is limited to 8 tokens, then window Yes/No answers are aggregated by duration.')
    return command


if __name__ == '__main__':
    main()
