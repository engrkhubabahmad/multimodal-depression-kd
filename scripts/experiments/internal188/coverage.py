"""Audit transcripts and cached audio for TRAIN/VAL; never open internal TEST media."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .split import digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--daic-root', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-cache', type=Path, action='append', required=True)
    a = p.parse_args(argv)
    marker = json.loads((a.split_dir / 'complete.json').read_text())
    manifest_path = a.split_dir / 'manifest.csv'
    if digest(manifest_path) != marker['manifest_sha256']: raise ValueError('Split manifest changed')
    manifest = pd.read_csv(manifest_path)
    if manifest.split.value_counts().to_dict() != {'train': 150, 'val': 19, 'student_test': 19}:
        raise ValueError('Unexpected split counts')
    selected = manifest.loc[manifest.split.isin(['train', 'val'])].sort_values('participant_id')
    if len(selected) != 169 or selected.participant_id.duplicated().any(): raise ValueError('TRAIN/VAL IDs invalid')
    wanted = set(selected.participant_id.astype(int)); transcripts = {}
    for path in a.daic_root.rglob('*_TRANSCRIPT.csv'):
        match = re.fullmatch(r'(\d+)_TRANSCRIPT\.csv', path.name, re.I)
        if match and int(match.group(1)) in wanted:
            transcripts.setdefault(int(match.group(1)), []).append(path)
    caches = []
    for folder in a.feature_cache:
        index = folder / 'participant_manifest.csv'
        if index.is_file():
            frame = pd.read_csv(index)
            if not {'participant_id', 'label', 'feature_path'} <= set(frame): raise ValueError(f'Invalid cache {index}')
            if frame.participant_id.duplicated().any(): raise ValueError(f'Duplicate IDs in {index}')
            caches.append((index, frame.set_index('participant_id')))
    rows = []; missing = []
    for row in selected.itertuples(index=False):
        pid = int(row.participant_id); matches = transcripts.get(pid, [])
        if len(matches) != 1:
            missing.append(f'{pid}: transcript matches={len(matches)}'); continue
        candidates = []
        for index, cache in caches:
            if pid in cache.index:
                found = cache.loc[pid]
                if int(found.label) != int(row.label): raise ValueError(f'{pid}: cache label mismatch in {index}')
                path = Path(found.feature_path)
                if path.is_file(): candidates.append((index, path))
        if not candidates:
            missing.append(f'{pid}: no cached audio'); continue
        index, path = candidates[0]
        audio = np.load(path, mmap_mode='r', allow_pickle=False)
        if audio.ndim != 2 or audio.shape[0] != 130 or audio.shape[1] < 1:
            raise ValueError(f'{pid}: bad ComParE16 shape {audio.shape}')
        rows.append({'participant_id': pid, 'label': int(row.label), 'split': row.split,
                     'transcript_path': str(matches[0]), 'feature_path': str(path),
                     'frames': int(audio.shape[1]), 'cache_manifest': str(index)})
    if missing: raise FileNotFoundError('TRAIN/VAL coverage incomplete: ' + '; '.join(missing[:30]))
    if a.output.exists() and any(a.output.iterdir()):
        prior = a.output / 'participant_manifest.csv'
        if not prior.is_file() or not pd.read_csv(prior).equals(pd.DataFrame(rows)):
            raise ValueError('Existing coverage differs; use a fresh output directory')
        print('Verified existing TRAIN/VAL coverage:', a.output); return
    a.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output / 'participant_manifest.csv', index=False)
    (a.output / 'audit.json').write_text(json.dumps({'split_manifest_sha256': digest(manifest_path),
        'train_count': 150, 'val_count': 19, 'internal_test_media_opened': False,
        'feature_cache_manifests': {str(index): digest(index) for index, _ in caches}}, indent=2) + '\n')
    print('TRAIN/VAL coverage complete: 169 transcripts and 169 audio arrays')
    print(pd.DataFrame(rows).groupby(['split', 'cache_manifest']).size())


if __name__ == '__main__': main()
