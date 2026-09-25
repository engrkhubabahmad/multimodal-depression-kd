"""Prepare exploratory SFT from the released Step-Audio2 DAIC LoRA on TRAIN/VAL."""
import argparse
import json
import shlex
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from scripts.teachers.ussd_audio.common import extract_participant_wav, one
from .split import digest
from .train_text_teacher import verified_split
from .prepare_stepaudio2_val import ADAPTER, MODEL, PROMPT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daic-root', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path, required=True)
    p.add_argument('--adapter-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    manifest = verified_split(a.split_dir, a.coverage_dir)
    audit = json.loads(a.adapter_audit.read_text())
    if audit['adapter'] != ADAPTER or audit['split_sha256'] != digest(a.split_dir / 'manifest.csv'):
        raise ValueError('Adapter audit belongs to another split or model')
    covered = pd.read_csv(a.coverage_dir / 'participant_manifest.csv').set_index('participant_id')
    a.output.mkdir(parents=True, exist_ok=True)
    wav_dir = a.output / 'participant_audio'; wav_dir.mkdir(exist_ok=True)
    for split, count in [('train', 113), ('val', 37)]:
        subset = manifest.loc[manifest.split.eq(split)].sort_values('participant_id')
        if len(subset) != count: raise ValueError(f'Unexpected {split} count')
        lines = []; provenance = []
        for row in tqdm(subset.itertuples(index=False), total=count,
                        desc=f'Step-Audio2 {split} audio', colour='green'):
            pid = int(row.participant_id)
            transcript = Path(covered.loc[pid, 'transcript_path'])
            raw = one(a.daic_root, f'{pid}_AUDIO.wav')
            wav = wav_dir / f'{pid}.wav'
            if not wav.is_file(): extract_participant_wav(raw, transcript, wav)
            if wav.stat().st_size <= 44: raise ValueError(f'Empty WAV: {pid}')
            answer = 'Yes' if int(row.label) == 1 else 'No'
            lines.append({'messages': [{'role': 'user', 'content': '<audio>' + PROMPT},
                                       {'role': 'assistant', 'content': answer}],
                          'audios': [str(wav)]})
            provenance.append({'participant_id': pid, 'label': int(row.label), 'split': split,
                               'transcript_sha256': digest(transcript), 'raw_audio_sha256': digest(raw),
                               'audio_path': str(wav)})
        dest = a.output / f'{split}.jsonl'; content = ''.join(json.dumps(x) + '\n' for x in lines)
        index = a.output / f'{split}_index.csv'; frame = pd.DataFrame(provenance)
        if dest.exists() and dest.read_text() != content: raise ValueError(f'Existing {dest} differs')
        if index.exists() and not pd.read_csv(index).equals(frame): raise ValueError(f'Existing {index} differs')
        dest.write_text(content); frame.to_csv(index, index=False)
    command = ['swift', 'sft', '--model', MODEL, '--model_type', 'step_audio2_mini',
               '--adapters', ADAPTER, '--use_hf', 'true', '--tuner_type', 'lora',
               '--dataset', str(a.output / 'train.jsonl'), '--val_dataset', str(a.output / 'val.jsonl'),
               '--output_dir', str(a.output / 'checkpoints'), '--seed', '103',
               '--num_train_epochs', '3', '--learning_rate', '1e-5',
               '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '8',
               '--eval_strategy', 'epoch', '--save_strategy', 'epoch',
               '--save_total_limit', '3', '--max_length', '8000']
    print('Prepared 113 TRAIN and 37 VAL participants. Internal TEST unopened.')
    print('Run only after the single-VAL inference pilot succeeds on your GPU:')
    print(shlex.join(command))
    print('This initializes from DAIC-trained adapter weights; validation is exploratory, and VAL loss is not participant F1.')
    return command


if __name__ == '__main__': main()
