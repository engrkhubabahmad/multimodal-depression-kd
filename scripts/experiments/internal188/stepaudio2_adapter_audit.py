"""Audit the public Step-Audio2 DAIC LoRA adapter before exploratory use."""
import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

from .split import digest

ADAPTER = 'maimai11/woz'
EXPECTED_BASE = 'stepfun-ai/Step-Audio-2-mini-Think'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download-weights', action='store_true')
    a = p.parse_args(argv)
    marker = json.loads((a.split_dir / 'complete.json').read_text())
    split_file = a.split_dir / 'manifest.csv'
    if digest(split_file) != marker['manifest_sha256']:
        raise ValueError('Split manifest changed')
    manifest = pd.read_csv(split_file)
    if manifest.split.value_counts().to_dict() != {'train': 113, 'val': 37, 'student_test': 38}:
        raise ValueError('Unexpected participant counts')
    info = HfApi().model_info(ADAPTER, files_metadata=True)
    filenames = {f.rfilename for f in info.siblings}
    required = {'adapter_config.json', 'adapter_model.safetensors', 'dev_metrics.json'}
    if not required <= filenames: raise ValueError(f'Missing adapter artifacts: {required - filenames}')
    allow = list(required if a.download_weights else required - {'adapter_model.safetensors'})
    directory = Path(snapshot_download(ADAPTER, revision=info.sha, allow_patterns=allow))
    adapter_config = json.loads((directory / 'adapter_config.json').read_text())
    base = adapter_config.get('base_model_name_or_path', '')
    # The published adapter stores a machine-local path with "Audio2", while
    # the public StepFun repository spells the same model "Audio-2".
    allowed_names = {'step-audio2-mini-think', 'step-audio-2-mini-think'}
    if not base or base.rstrip('/').split('/')[-1].lower() not in allowed_names:
        raise ValueError(f'Adapter base differs from expected Step-Audio2 mini Think: {base}')
    declared = json.loads((directory / 'dev_metrics.json').read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    report = {'adapter': ADAPTER, 'revision': info.sha, 'base_declared_in_config': base,
              'expected_base': EXPECTED_BASE, 'adapter_config': adapter_config,
              'model_card_dev_metrics': declared, 'split_sha256': digest(split_file),
              'split_counts': manifest.split.value_counts().to_dict(),
              'original_canonical_train_in_val': int(((manifest.source_split == 'canonical_train') & (manifest.split == 'val')).sum()),
              'original_canonical_train_in_internal_test': int(((manifest.source_split == 'canonical_train') & (manifest.split == 'student_test')).sum()),
              'author_training_participant_ids_verified': False,
              'clean_holdout_claim_valid': False,
              'adapter_weights_downloaded': a.download_weights,
              'status': 'QUARANTINED: do not use for primary KD or clean validation/test claims without author training roster'}
    target = a.output / 'audit.json'
    if target.exists() and json.loads(target.read_text()) != report:
        raise ValueError(f'Existing adapter audit differs: {target}')
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('adapter', 'revision', 'base_declared_in_config',
        'original_canonical_train_in_val', 'original_canonical_train_in_internal_test',
        'author_training_participant_ids_verified', 'status')}, indent=2))
    print('Audit:', target)


if __name__ == '__main__': main()
