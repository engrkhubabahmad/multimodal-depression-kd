"""Fit a regularized participant-level audio teacher to frozen WavLM embeddings."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .split import digest
from .train_text_teacher import verified_split


def scores(y, p):
    pred = (p >= .5).astype(int)
    return {'n': len(y), 'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'depressed_f1': float(f1_score(y, pred, zero_division=0)),
            'auroc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path, required=True)
    p.add_argument('--embeddings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=103)
    a = p.parse_args(argv)
    manifest = verified_split(a.split_dir, a.coverage_dir)
    embed_audit = json.loads((a.embeddings / 'audit.json').read_text())
    if embed_audit['split_sha256'] != digest(a.split_dir / 'manifest.csv'):
        raise ValueError('WavLM embeddings belong to another split')
    audio = pd.read_csv(a.embeddings / 'participant_manifest.csv').sort_values('participant_id')
    selected = manifest.loc[manifest.split.isin(['train', 'val'])].sort_values('participant_id')
    if not audio[['participant_id', 'label', 'split']].reset_index(drop=True).equals(
            selected[['participant_id', 'label', 'split']].reset_index(drop=True)):
        raise ValueError('Embedding IDs, labels or split differ')
    vectors = []
    for row in audio.itertuples(index=False):
        with np.load(row.feature_path, allow_pickle=False) as z:
            if str(z['revision']) != embed_audit['revision']:
                raise ValueError(f'{row.participant_id}: WavLM revision differs')
            vectors.append(z['embedding'].astype(np.float32))
    X = np.stack(vectors)
    if X.shape != (169, 1536) or not np.isfinite(X).all(): raise ValueError('Invalid WavLM embeddings')
    train = audio.split.eq('train').to_numpy(); val = audio.split.eq('val').to_numpy()
    if train.sum() != 150 or val.sum() != 19: raise ValueError('TRAIN/VAL counts changed')
    y = audio.label.to_numpy(int)
    # Predeclared C and 0.5 threshold: no search over the small validation group.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced',
                          solver='liblinear', max_iter=2000, random_state=a.seed))
    model.fit(X[train], y[train])
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()): raise ValueError('Use a fresh WavLM teacher output folder')
    results = {}
    for name, mask in [('train', train), ('val', val)]:
        prob = model.predict_proba(X[mask])[:, 1]
        safe = np.clip(prob, 1e-6, 1 - 1e-6)
        frame = audio.loc[mask].copy()
        pd.DataFrame({'participant_id': frame.participant_id.to_numpy(int),
                      'label': frame.label.to_numpy(int), 'audio_probability': safe,
                      'audio_logit': logit(safe)}).to_csv(a.output / f'{name}_audio_targets.csv', index=False)
        results[name] = scores(y[mask], prob)
    with (a.output / 'classifier.pkl').open('wb') as f: pickle.dump(model, f)
    (a.output / 'audit.json').write_text(json.dumps({'model': embed_audit['model'],
        'pretrained_revision': embed_audit['revision'], 'embedding_manifest_sha256': digest(a.embeddings / 'participant_manifest.csv'),
        'split_sha256': digest(a.split_dir / 'manifest.csv'), 'seed': a.seed,
        'classifier': 'StandardScaler + balanced L2 logistic regression', 'C': .1, 'threshold': .5,
        'normalization_fit': 'TRAIN-150 only', 'internal_test_opened': False,
        'pretrained_on_daic': False, 'train': results['train'], 'val': results['val']}, indent=2) + '\n')
    print('TRAIN:', results['train']); print('VAL:', results['val'])


if __name__ == '__main__': main()
