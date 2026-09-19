# Idiap raw participant rebuild

This is a clean, text-only diagnostic path:

```text
local DAIC-WOZ TRANSCRIPT.csv
  -> stable time order
  -> non-scrubbed Participant utterances only
  -> one concatenated document per participant
  -> published fitted TF-IDF vectorizer
  -> published InducT-GCN weights
  -> positive-class probability at threshold 0.5
```

It uses the official DEV participant IDs, excludes 440 by default, and never searches or reads a test split.

## Run in Colab

```python
!python -m scripts.teachers.solo_published.bootstrap
!python -m scripts.teachers.idiap_raw.run_dev \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --source-root /content/solo_teacher_sources/bias_in_daic-woz \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/idiap_raw/dev
```

The green `Idiap raw participant preprocessing` bar covers all 34 raw local DEV interviews. Outputs are `dev_raw_predictions.jsonl` and `dev_raw_metrics.json`.

## Interpretation

The public checkpoint can exactly replay its published score only by using its stored author-prepared DEV TF-IDF matrix. That matrix is not the same artifact as a raw local transcript rebuild, and the public author repository cannot distribute the original prepared corpus. Therefore `run_dev.py` is a reproducible rebuild of the published *public preprocessing contract*, not a claim of exact stored-feature reproduction. The existing `solo_published.run_text` is retained solely for the stored-feature benchmark.

This branch does not make KD targets. A frozen checkpoint without compatible TRAIN input features cannot produce valid TRAIN teacher logits.
