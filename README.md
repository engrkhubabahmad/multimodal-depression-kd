# reliability-aware-depression-kd

## Active protocol

This repository uses a strict participant-disjoint DAIC-WOZ protocol:

- TRAIN: 107 participants
- DEV: 34 participants
- participant 440 excluded from DEV
- TEST: 47 participants, closed until the final student is frozen

## Locked teachers

### Audio teacher

Ravi et al. USSD, ComParE16 + LSTM-only, released run #4:

- author commit: `c3e68649153004ed2174878a1c54c716ad26cfd7`
- checkpoint: `md_35_epochs.pth`
- author normalization reused
- frozen DEV-34 macro-F1: 0.7875
- TRAIN KD targets: author-style run-4 seed-1300, 6662-frame crop, participant-level mean probability/logit
- no fine-tuning is part of the active teacher path

Code: `scripts/teachers/ussd_audio/`

### Text teacher

Idiap participant-level InducT-GCN top-250:

- published checkpoint + vectorizer
- TRAIN KD targets: checkpoint training-graph document nodes
- DEV targets: saved `A_dev`, participant 440 excluded
- DEV-34 macro-F1: 0.8418604651
- participant-level probability and binary logit exported directly from the checkpoint
- TEST is never opened

Code: `scripts/teachers/idiap_text/`

## Active teacher outputs

Expected Drive artifacts:

```text
DAIC_WOZ/experiments/ussd_audio_teacher_final/
  train_run4_crop_kd_targets.csv
  train_run4_crop_audit.json
  dev_predictions.csv
  metrics.json

DAIC_WOZ/experiments/idiap_text_teacher/
  train_text_kd_targets.csv
  dev_text_predictions.csv
  text_target_audit.json
```

## Student status

The old segment-teacher pipeline has been removed. Existing student files are retained only as a backbone for migration. Do not start student training until the student pipeline is updated to consume the locked participant-level teachers and select/evaluate checkpoints at participant level.

TEST must remain closed.
