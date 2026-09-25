"""Score exploratory Step-Audio2 VAL Yes/No generations without using TEST."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .split import digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--generations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    marker = json.loads((a.split_dir / 'complete.json').read_text())
    if digest(a.split_dir / 'manifest.csv') != marker['manifest_sha256']:
        raise ValueError('Split manifest changed')
    val = pd.read_csv(a.split_dir / 'manifest.csv').query("split == 'val'").sort_values('participant_id')
    index = pd.read_csv(a.index).sort_values('participant_id')
    if len(val) != 37 or not val[['participant_id', 'label']].reset_index(drop=True).equals(
            index[['participant_id', 'label']].reset_index(drop=True)):
        raise ValueError('VAL index does not match frozen split')
    rows = [json.loads(line) for line in a.generations.read_text().splitlines() if line.strip()]
    if len(rows) != 37: raise ValueError(f'Expected 37 generations, found {len(rows)}')
    answers = {}
    for row in rows:
        audios = row.get('audios')
        if not isinstance(audios, list) or len(audios) != 1:
            raise ValueError('Generation missing single participant audio path')
        pid = int(Path(audios[0]).stem)
        expected = index.loc[index.participant_id.eq(pid)]
        if len(expected) != 1 or Path(audios[0]) != Path(expected.iloc[0].audio_path) or pid in answers:
            raise ValueError(f'Unknown, changed, or duplicated participant: {pid}')
        response = row.get('response')
        match = re.match(r'^\s*(yes|no)\b', response or '', flags=re.IGNORECASE)
        if not match: raise ValueError(f'{pid}: ambiguous first answer: {str(response)[:120]!r}')
        answers[pid] = {'participant_id': pid, 'label': int(expected.iloc[0].label),
                        'prediction': int(match.group(1).lower() == 'yes'), 'response': response}
    result = pd.DataFrame(answers.values()).sort_values('participant_id').reset_index(drop=True)
    if set(result.participant_id) != set(val.participant_id): raise ValueError('Incomplete VAL coverage')
    y = result.label.to_numpy(); pred = result.prediction.to_numpy()
    metrics = {'n': 37, 'accuracy': float(accuracy_score(y, pred)),
               'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
               'depressed_f1': float(f1_score(y, pred, pos_label=1, zero_division=0)),
               'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
               'split_sha256': digest(a.split_dir / 'manifest.csv'),
               'generations_sha256': digest(a.generations),
               'probabilities_available': False, 'auroc_available': False,
               'status': 'EXPLORATORY; inherited adapter may have prior exposure to current VAL'}
    a.output.mkdir(parents=True, exist_ok=True)
    predictions = a.output / 'val_predictions.csv'; summary = a.output / 'val_metrics.json'
    if predictions.exists() and not pd.read_csv(predictions).equals(result):
        raise ValueError(f'Existing predictions differ: {predictions}')
    if summary.exists() and json.loads(summary.read_text()) != metrics:
        raise ValueError(f'Existing metrics differ: {summary}')
    result.to_csv(predictions, index=False)
    summary.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__': main()
