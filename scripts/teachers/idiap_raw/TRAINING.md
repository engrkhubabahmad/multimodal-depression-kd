# Raw Idiap-style teacher training

This is the legal raw-data text-teacher path.

```text
official TRAIN-107 local transcripts
→ one Participant document / participant
→ TRAIN-only English-stopword TF-IDF
→ TRAIN-only SelectKBest(f_classif), top 250 terms
→ author InducTGCN graph (PMI window=3, embedding=64, dropout=.5)
→ AdamW training with balanced class weights
→ DEV-34 macro-F1 selects one frozen checkpoint
→ TRAIN and DEV participant-level logits/probabilities
```

TEST is never read. Participant 440 is excluded by default.

## Colab

```python
!python -m scripts.teachers.solo_published.bootstrap

!python -m scripts.teachers.idiap_raw.train_teacher \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --source-root /content/solo_teacher_sources/bias_in_daic-woz \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/idiap_raw_trained/seed17 \
  --epochs 600 \
  --patience 80 \
  --learning-rates 1e-4 3e-4 1e-3
```

Outputs:

- `frozen_idiap_style_teacher.pt`
- `vectorizer.pkl`
- `train_predictions.csv`
- `dev_predictions.csv`
- `metrics.json`

The model is a raw-data reimplementation using Idiap's released architecture, not the unreleased author-prepared input checkpoint.