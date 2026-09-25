"""Aggregate frozen Step-Audio2 chunk answers into participant-level votes."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .split import digest


def explicit_yes_no(response):
    response = response or ''
    candidates = []
    match = re.match(r'^\s*(?:<[^>\r\n]{1,32}>\s*)*(yes|no)\b', response, re.I)
    if match:
        candidates.append(match.group(1).lower())
    candidates.extend(x.lower() for x in re.findall(r'<英语>\s*(yes|no)\b', response, re.I))
    return candidates[0] if candidates and len(set(candidates)) == 1 else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--generations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    manifest_path = a.split_dir / 'manifest.csv'
    marker = json.loads((a.split_dir / 'complete.json').read_text())
    if digest(manifest_path) != marker['manifest_sha256']:
        raise ValueError('Split manifest changed')
    manifest = pd.read_csv(manifest_path)
    expected = manifest.loc[manifest.split.isin(['train', 'val'])]
    index = pd.read_csv(a.index)
    required = {'participant_id', 'label', 'split', 'chunk_id', 'duration_seconds', 'chunk_path'}
    if not required.issubset(index.columns):
        raise ValueError(f'Chunk index missing columns: {sorted(required - set(index.columns))}')
    if index.duplicated(['participant_id', 'chunk_id']).any():
        raise ValueError('Duplicate participant/chunk entries in index')
    checked = index[['participant_id', 'label', 'split']].drop_duplicates()
    if checked.participant_id.duplicated().any():
        raise ValueError('Participant label/split changes between chunks')
    expected_map = expected.set_index('participant_id')[['label', 'split']]
    for r in checked.itertuples(index=False):
        pid = int(r.participant_id)
        if pid not in expected_map.index:
            raise ValueError(f'Participant {pid} is outside canonical TRAIN/DEV')
        want = expected_map.loc[pid]
        if int(r.label) != int(want.label) or r.split != want.split:
            raise ValueError(f'Participant {pid} label/split differs from canonical manifest')

    rows = [json.loads(line) for line in a.generations.read_text().splitlines() if line.strip()]
    if len(rows) != len(index):
        raise ValueError(f'Expected {len(index)} chunk generations; found {len(rows)}')
    expected_paths = dict(zip(index.chunk_path.astype(str), index.index))
    seen = {}
    for row in rows:
        audios = row.get('audios')
        if not isinstance(audios, list) or len(audios) != 1:
            raise ValueError('Generation lacks its single chunk audio path')
        path = str(audios[0])
        row_idx = expected_paths.get(path)
        if row_idx is None or row_idx in seen:
            raise ValueError(f'Unknown or duplicate generated chunk path: {path}')
        response = row.get('response') or ''
        seen[row_idx] = (explicit_yes_no(response), response)
    if len(seen) != len(index):
        raise ValueError(f'Generation coverage incomplete: {len(seen)}/{len(index)} chunks')

    chunks = index.copy()
    chunks['response'] = [seen[i][1] for i in chunks.index]
    chunks['answer'] = [seen[i][0] for i in chunks.index]
    chunks['parseable'] = chunks.answer.notna()
    chunks['prediction'] = chunks.answer.map({'no': 0, 'yes': 1})
    participants = []
    for pid, group in chunks.groupby('participant_id', sort=True):
        parsed = group.loc[group.parseable]
        yes_seconds = float(parsed.loc[parsed.prediction.eq(1), 'duration_seconds'].sum())
        no_seconds = float(parsed.loc[parsed.prediction.eq(0), 'duration_seconds'].sum())
        prediction = None if parsed.empty else int(yes_seconds > no_seconds)
        first = group.iloc[0]
        participants.append({'participant_id': int(pid), 'label': int(first.label),
            'split': first.split, 'n_chunks': int(len(group)), 'n_parseable_chunks': int(len(parsed)),
            'parsed_duration_seconds': float(parsed.duration_seconds.sum()),
            'yes_duration_seconds': yes_seconds, 'no_duration_seconds': no_seconds,
            'prediction': prediction, 'response_parseable': prediction is not None})
    result = pd.DataFrame(participants)
    metrics = {}
    for split in ('train', 'val'):
        group = result.loc[result.split.eq(split)]
        known = group.loc[group.prediction.notna()]
        y = known.label.to_numpy(int)
        pred = known.prediction.to_numpy(int)
        metrics[split] = {'n_total': int(len(group)), 'n_parseable': int(len(known)),
            'n_unparseable': int(len(group) - len(known)),
            'accuracy_parseable_only': float(accuracy_score(y, pred)) if len(known) else None,
            'macro_f1_parseable_only': float(f1_score(y, pred, average='macro', zero_division=0)) if len(known) else None,
            'depressed_f1_parseable_only': float(f1_score(y, pred, zero_division=0)) if len(known) else None,
            'confusion_matrix_parseable_only': confusion_matrix(y, pred, labels=[0, 1]).tolist() if len(known) else None,
            'classification_report_parseable_only': classification_report(y, pred, labels=[0, 1],
                target_names=['Non-depressed', 'Depressed'], output_dict=True, zero_division=0) if len(known) else None,
            'full_split_scored': bool(len(known) == len(group))}
    summary = {'adapter': 'maimai11/woz frozen; no training',
        'aggregation': 'duration-weighted majority of explicit per-chunk Yes/No answers; ties predict No',
        'participants_evaluated': int(len(result)), 'chunks_evaluated': int(len(chunks)),
        'unparseable_chunk_count': int((~chunks.parseable).sum()), 'results': metrics,
        'probabilities_available': False, 'auroc_available': False, 'test_opened': False,
        'status': 'EXPLORATORY; chunking/prompt are approximate because author preprocessing and training roster are unavailable.'}
    a.output.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(a.output / 'chunk_predictions.csv', index=False)
    result.to_csv(a.output / 'participant_predictions.csv', index=False)
    (a.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()
