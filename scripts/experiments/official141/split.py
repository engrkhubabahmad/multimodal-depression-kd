"""Stratified 113/14/14 split of original TRAIN+DEV only; no TEST reads."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    raw = pd.read_csv(path); raw.columns = raw.columns.str.strip().str.lower()
    frame = raw[['participant_id', 'phq8_binary']].rename(columns={'phq8_binary': 'label'}).copy()
    if frame.isna().any().any(): raise ValueError(f'Missing participant ID or label in {path}')
    frame = frame.astype(int)
    if frame.participant_id.duplicated().any() or not frame.label.isin([0, 1]).all():
        raise ValueError(f'Invalid participant labels in {path}')
    return frame


def build(root, seed=103):
    metadata = Path(root) / 'metadata'
    train_path = metadata / 'train_split_Depression_AVEC2017.csv'
    dev_path = metadata / 'dev_split_Depression_AVEC2017.csv'
    train = source(train_path); dev = source(dev_path)
    if (len(train), len(dev)) != (107, 35) or 440 not in set(dev.participant_id):
        raise ValueError('Expected original TRAIN-107 and DEV-35 including corrupt participant 440')
    dev = dev.loc[dev.participant_id.ne(440)].copy()
    if set(train.participant_id) & set(dev.participant_id): raise ValueError('TRAIN/DEV overlap')
    pool = pd.concat([train.assign(source_split='original_train'),
                      dev.assign(source_split='original_dev')], ignore_index=True)
    if len(pool) != 141 or pool.participant_id.duplicated().any(): raise ValueError('Pool is not 141 distinct participants')
    fit, held = train_test_split(pool, train_size=113, stratify=pool.label, random_state=seed)
    val, internal_test = train_test_split(held, train_size=14, stratify=held.label, random_state=seed)
    manifest = pd.concat([fit.assign(split='train'), val.assign(split='val'),
                          internal_test.assign(split='internal_test')]).sort_values('participant_id').reset_index(drop=True)
    assert manifest.split.value_counts().to_dict() == {'train': 113, 'val': 14, 'internal_test': 14}
    return manifest, {str(p): sha(p) for p in (train_path, dev_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daic-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=103)
    args = parser.parse_args(argv)
    manifest, sources = build(args.daic_root, args.seed)
    signature = {'seed': args.seed, 'source_sha256': sources, 'code_sha256': sha(__file__),
                 'counts': {'train': 113, 'val': 14, 'internal_test': 14}}
    marker = args.output / 'complete.json'; csv = args.output / 'manifest.csv'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get('signature') != signature or not csv.is_file() or previous.get('manifest_sha256') != sha(csv):
            raise ValueError('Saved split does not match current inputs/code; use another output directory')
        print('Verified existing split:', args.output); return
    if args.output.exists() and any(args.output.iterdir()): raise ValueError('Refusing to overwrite nonempty split')
    args.output.mkdir(parents=True, exist_ok=True); manifest.to_csv(csv, index=False)
    marker.write_text(json.dumps({'signature': signature, 'manifest_sha256': sha(csv),
        'official_test_accessed': False, 'participant_440_excluded': True,
        'class_counts': {name: group.label.value_counts().sort_index().to_dict()
                         for name, group in manifest.groupby('split')}}, indent=2) + '\n')
    print('Saved split:', args.output); print(pd.crosstab(manifest.split, manifest.label))


if __name__ == '__main__': main()
