"""Prepare validation-only participant speech for exploratory Step-Audio2 inference."""
import argparse
import json
import shlex
from pathlib import Path

import pandas as pd

from scripts.teachers.ussd_audio.common import extract_participant_wav, one
from .split import digest
from .train_text_teacher import verified_split

MODEL = 'stepfun-ai/Step-Audio-2-mini-Think'
ADAPTER = 'maimai11/woz'
PROMPT = 'Listen only to the participant speech. Is the participant depressed? Answer Yes or No.'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daic-root', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path, required=True)
    p.add_argument('--adapter-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pilot-only', action='store_true', help='Prepare one VAL example to check runtime first')
    a = p.parse_args(argv)
    manifest = verified_split(a.split_dir, a.coverage_dir)
    val = manifest.loc[manifest.split.eq('val')].sort_values('participant_id')
    audit = json.loads(a.adapter_audit.read_text())
    if audit['split_sha256'] != digest(a.split_dir / 'manifest.csv') or audit['adapter'] != ADAPTER:
        raise ValueError('Adapter audit belongs to another split or model')
    if a.pilot_only: val = val.head(1)
    coverage = pd.read_csv(a.coverage_dir / 'participant_manifest.csv').set_index('participant_id')
    a.output.mkdir(parents=True, exist_ok=True)
    audio_dir = a.output / 'participant_audio'; audio_dir.mkdir(exist_ok=True)
    dataset = a.output / ('pilot.jsonl' if a.pilot_only else 'val.jsonl')
    index = a.output / ('pilot_index.csv' if a.pilot_only else 'val_index.csv')
    records = []; references = []
    for row in val.itertuples(index=False):
        pid = int(row.participant_id)
        transcript = Path(coverage.loc[pid, 'transcript_path'])
        if not transcript.is_file(): raise FileNotFoundError(transcript)
        raw = one(a.daic_root, f'{pid}_AUDIO.wav')
        wav = audio_dir / f'{pid}.wav'
        if not wav.exists(): extract_participant_wav(raw, transcript, wav)
        if not wav.is_file() or wav.stat().st_size < 44: raise ValueError(f'Empty speech WAV: {pid}')
        records.append({'messages': [{'role': 'user', 'content': '<audio>' + PROMPT}], 'audios': [str(wav)]})
        references.append({'participant_id': pid, 'label': int(row.label), 'split': 'val',
                           'audio_path': str(wav), 'transcript_sha256': digest(transcript),
                           'raw_audio_sha256': digest(raw)})
    content = ''.join(json.dumps(r) + '\n' for r in records)
    reference = pd.DataFrame(references)
    if dataset.exists() and dataset.read_text() != content: raise ValueError('Dataset differs; refusing overwrite')
    if index.exists() and not pd.read_csv(index).equals(reference): raise ValueError('Index differs; refusing overwrite')
    dataset.write_text(content); reference.to_csv(index, index=False)
    result = a.output / ('pilot_predictions.jsonl' if a.pilot_only else 'val_predictions.jsonl')
    command = ['swift', 'infer', '--model', MODEL, '--model_type', 'step_audio2_mini',
               '--adapters', ADAPTER, '--use_hf', 'true',
               '--load_args', 'false', '--val_dataset', str(dataset), '--result_path', str(result),
               '--infer_backend', 'transformers', '--max_batch_size', '1',
               '--temperature', '0', '--max_new_tokens', '16']
    print('Prepared', len(reference), 'VAL participant(s); internal TEST unopened.')
    print('Run with a Step-Audio2 compatible GPU and ms-swift installed:')
    print(shlex.join(command))
    print('Order map:', index)
    print('Exploratory only: prompt/aggregation may differ from adapter author training; no calibrated logits.')


if __name__ == '__main__': main()
