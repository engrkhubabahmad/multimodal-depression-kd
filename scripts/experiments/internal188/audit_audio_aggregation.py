"""Compare saved audio teacher mean probability with author's segment hard vote; TRAIN/VAL only."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from scripts.teachers.ussd_audio.common import CustomComparE16
from .split import digest
from .train_audio_teacher import FRAMES, rows_for
from .train_text_teacher import verified_split


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--coverage-dir', type=Path, required=True)
    p.add_argument('--teacher-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    manifest = verified_split(a.split_dir, a.coverage_dir)
    indexed = rows_for(a.coverage_dir, manifest)
    audit = json.loads((a.teacher_dir / 'audit.json').read_text())
    if audit['signature']['split_sha256'] != digest(a.split_dir / 'manifest.csv'):
        raise ValueError('Audio checkpoint belongs to another split')
    with np.load(a.teacher_dir / 'normalization.npz') as z:
        mean, std = z['mean'], z['std']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomComparE16().to(device)
    state = torch.load(a.teacher_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'], strict=True); model.eval()
    rows = []
    with torch.inference_mode():
        for r in manifest.loc[manifest.split.isin(['train', 'val'])].itertuples(index=False):
            x = np.load(indexed.loc[int(r.participant_id), 'feature_path'], mmap_mode='r', allow_pickle=False)
            probs = []
            for start in range(0, x.shape[1], FRAMES * 32):
                batch = []
                for frame in range(start, min(start + FRAMES * 32, x.shape[1]), FRAMES):
                    block = np.zeros((130, FRAMES), np.float32)
                    part = np.asarray(x[:, frame:frame + FRAMES], np.float32)
                    block[:, :part.shape[1]] = part
                    batch.append(((block - mean) / std).astype(np.float32))
                _, pred, _ = model.forward_logits(torch.from_numpy(np.stack(batch)).to(device))
                probs.extend(pred.cpu().numpy().tolist())
            values = np.asarray(probs, float)
            # Match the author's np.rint per segment, then np.rint vote fraction.
            vote_fraction = float(np.rint(values).mean())
            mean_prob = float(values.mean())
            rows.append({'participant_id': int(r.participant_id), 'split': r.split,
                         'label': int(r.label), 'segments': int(math.ceil(x.shape[1] / FRAMES)),
                         'mean_probability': mean_prob, 'mean_prediction': int(mean_prob >= .5),
                         'author_vote_fraction': vote_fraction,
                         'author_prediction': int(np.rint(vote_fraction))})
    data = pd.DataFrame(rows).sort_values('participant_id')
    for split in ('train', 'val'):
        saved = pd.read_csv(a.teacher_dir / f'{split}_audio_targets.csv').sort_values('participant_id')
        selected = data.loc[data.split.eq(split)].sort_values('participant_id')
        if (not np.array_equal(saved.participant_id.to_numpy(int), selected.participant_id.to_numpy(int))
                or not np.allclose(saved.audio_probability.to_numpy(float),
                                   selected.mean_probability.to_numpy(float), atol=1e-5, rtol=0)):
            raise ValueError(f'{split}: recalculated mean probabilities differ from saved teacher targets')
    result = {}
    for split, frame in data.groupby('split'):
        y = frame.label.to_numpy(int)
        result[split] = {'n': len(frame), 'auroc_mean_probability': float(roc_auc_score(y, frame.mean_probability))}
        for name, col in [('mean_probability', 'mean_prediction'), ('author_vote', 'author_prediction')]:
            pred = frame[col].to_numpy(int)
            result[split][name] = {'accuracy': float(accuracy_score(y, pred)),
                                   'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
                                   'depressed_f1': float(f1_score(y, pred, zero_division=0)),
                                   'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError('Refusing to overwrite existing audit directory')
    a.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(a.output / 'participant_predictions.csv', index=False)
    (a.output / 'summary.json').write_text(json.dumps({'teacher_checkpoint_sha256': digest(a.teacher_dir / 'best.pt'),
        'internal_test_opened': False, 'results': result}, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
