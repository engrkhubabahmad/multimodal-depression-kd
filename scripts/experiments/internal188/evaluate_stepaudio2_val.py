"""Aggregate exploratory Step-Audio2 window generations to VAL participants."""
import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .split import digest


def explicit_yes_no(response):
    """Return a unique explicit Yes/No label, or None when the output is ambiguous."""
    response = response or ''
    candidates = []
    leading = re.match(r'^\s*(?:<[^>\r\n]{1,32}>\s*)*(yes|no)\b', response, flags=re.IGNORECASE)
    if leading:
        candidates.append(leading.group(1).lower())
    candidates.extend(x.lower() for x in re.findall(r'<英语>\s*(yes|no)\b', response, flags=re.IGNORECASE))
    return candidates[0] if candidates and len(set(candidates)) == 1 else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--generations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-undecided-windows', action='store_true',
                   help='Score only if every participant vote is fixed for all possible missing answers')
    a = p.parse_args(argv)
    marker = json.loads((a.split_dir / 'complete.json').read_text())
    if digest(a.split_dir / 'manifest.csv') != marker['manifest_sha256']:
        raise ValueError('Split manifest changed')
    val = pd.read_csv(a.split_dir / 'manifest.csv').query("split == 'val'").sort_values('participant_id')
    index = pd.read_csv(a.index).sort_values(['participant_id', 'chunk_id']).reset_index(drop=True)
    if len(val) != 37 or val.participant_id.nunique() != 37:
        raise ValueError('Expected the frozen 37-person VAL split')
    labels = val.set_index('participant_id').label.astype(int).to_dict()
    if set(index.participant_id.astype(int)) != set(labels):
        raise ValueError('Window index does not cover the frozen VAL participants')
    if any(int(row.label) != labels[int(row.participant_id)] or row.split != 'val'
           for row in index.itertuples(index=False)):
        raise ValueError('Window labels/split do not match the frozen VAL manifest')
    rows = [json.loads(line) for line in a.generations.read_text().splitlines() if line.strip()]
    if len(rows) != len(index):
        raise ValueError(f'Expected {len(index)} window generations, found {len(rows)}')
    path_to_window = {str(Path(r.audio_path).resolve()): r for r in index.itertuples(index=False)}
    answers = {}
    for row in rows:
        audios = row.get('audios')
        if not isinstance(audios, list) or len(audios) != 1:
            raise ValueError('Generation missing single window audio path')
        key = str(Path(audios[0]).resolve())
        ref = path_to_window.get(key)
        if ref is None or key in answers:
            raise ValueError(f'Unknown, changed, or duplicated audio window: {audios[0]}')
        response = row.get('response') or ''
        answer = explicit_yes_no(response)
        if answer is None and not a.allow_undecided_windows:
            raise ValueError(f'{ref.participant_id}/{ref.chunk_id}: no unique explicit Yes/No answer '
                             f'in {len(response)} response characters; raw response omitted')
        answers[key] = {'participant_id': int(ref.participant_id), 'chunk_id': int(ref.chunk_id),
                        'label': int(ref.label), 'prediction': None if answer is None else int(answer == 'yes'),
                        'duration_seconds': float(ref.duration_seconds), 'response': response}
    if set(answers) != set(path_to_window):
        raise ValueError('Incomplete window generation coverage')
    chunks = pd.DataFrame(answers.values()).sort_values(['participant_id', 'chunk_id']).reset_index(drop=True)
    participant_rows = []
    for pid, group in chunks.groupby('participant_id', sort=True):
        yes = float(group.loc[group.prediction == 1, 'duration_seconds'].sum())
        no = float(group.loc[group.prediction == 0, 'duration_seconds'].sum())
        undecided = float(group.loc[group.prediction.isna(), 'duration_seconds'].sum())
        total = yes + no + undecided
        lower_fraction, upper_fraction = yes / total, (yes + undecided) / total
        lower_vote, upper_vote = int(lower_fraction > 0.5), int(upper_fraction > 0.5)
        if lower_vote != upper_vote:
            raise ValueError(f'Participant {pid} vote depends on {int(group.prediction.isna().sum())} '
                             'undecided window(s); no point metric is valid')
        participant_rows.append({'participant_id': int(pid), 'label': int(group.label.iloc[0]),
                                 'prediction': lower_vote, 'yes_duration_fraction_lower': lower_fraction,
                                 'yes_duration_fraction_upper': upper_fraction,
                                 'n_windows': int(len(group)), 'n_undecided_windows': int(group.prediction.isna().sum()),
                                 'audio_seconds': total,
                                 'tie_predicts_no': math.isclose(lower_fraction, 0.5, abs_tol=1e-12)})
    result = pd.DataFrame(participant_rows).sort_values('participant_id').reset_index(drop=True)
    if len(result) != 37 or set(result.participant_id) != set(labels):
        raise ValueError('Incomplete participant-level VAL coverage')
    y = result.label.to_numpy(); pred = result.prediction.to_numpy()
    metrics = {'n': 37, 'n_windows': int(len(chunks)),
               'n_explicit_windows': int(chunks.prediction.notna().sum()),
               'n_undecided_windows': int(chunks.prediction.isna().sum()),
               'participant_votes_invariant_to_undecided': True,
               'accuracy': float(accuracy_score(y, pred)),
               'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
               'depressed_f1': float(f1_score(y, pred, pos_label=1, zero_division=0)),
               'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
               'aggregation': 'duration-weighted window Yes fraction; predict Yes only when fraction > 0.5; ties predict No',
               'undecided_handling': 'Each undecided window may be Yes or No; report a point metric only when both extreme assignments give the same participant vote. No window answer is imputed.',
               'split_sha256': digest(a.split_dir / 'manifest.csv'),
               'generations_sha256': digest(a.generations),
               'probabilities_available': False, 'auroc_available': False,
               'status': 'EXPLORATORY; inherited adapter may have prior exposure to current VAL; windowing differs from any participant-level source evaluation'}
    a.output.mkdir(parents=True, exist_ok=True)
    chunk_path = a.output / 'window_predictions.csv'
    predictions = a.output / 'val_predictions.csv'
    summary = a.output / 'val_metrics.json'
    for path, frame in ((chunk_path, chunks), (predictions, result)):
        if path.exists() and path.read_text() != frame.to_csv(index=False):
            raise ValueError(f'Existing predictions differ: {path}')
    if summary.exists() and json.loads(summary.read_text()) != metrics:
        raise ValueError(f'Existing metrics differ: {summary}')
    chunks.to_csv(chunk_path, index=False); result.to_csv(predictions, index=False)
    summary.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    main()
