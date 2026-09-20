# reliability-aware-depression-kd

## Active protocol

Strict participant-disjoint DAIC-WOZ protocol:

- TRAIN: 107 participants
- DEV: 34 participants
- participant 440 excluded from DEV
- TEST: 47 participants, closed until the final student is frozen

## Locked teachers

### Audio teacher
Ravi et al. USSD, ComParE16 + LSTM-only, released run #4.

- author commit: `c3e68649153004ed2174878a1c54c716ad26cfd7`
- checkpoint: `md_35_epochs.pth`
- frozen DEV-34 macro-F1: 0.7875
- TRAIN KD targets: author-style run-4 seed-1300, 6662-frame crop
- no fine-tuning in the active path

Code: `scripts/teachers/ussd_audio/`

### Text teacher
Idiap participant-level InducT-GCN top-250.

- published checkpoint + vectorizer
- TRAIN KD targets: checkpoint training-graph document nodes
- DEV targets: saved `A_dev`, participant 440 excluded
- DEV-34 macro-F1: 0.8418604651
- TEST never opened

Code: `scripts/teachers/idiap_text/`

## Active teacher outputs

```text
DAIC_WOZ/experiments/ussd_frozen_audit/
  audit.json
  dev_participant_predictions.csv

DAIC_WOZ/experiments/ussd_audio_teacher_final/
  train_run4_crop_kd_targets.csv
  train_run4_crop_audit.json

DAIC_WOZ/experiments/idiap_text_teacher/
  train_text_kd_targets.csv
  dev_text_predictions.csv
  text_target_audit.json
```

## Active student

`ParticipantReLiMPNet` is the active student.

Per participant:
- up to 128 aligned Participant-only segments
- max 10 s audio/segment, min 0.5 s
- audio input: 64-bin log-Mel mean + std = 128-D/segment
- text input: TRAIN-only vocabulary, max 64 token IDs/segment
- audio MLP + one-layer 4-head text Transformer
- per-segment multimodal fusion
- masked attention pooling across participant segments
- one participant depression logit

No-KD training uses TRAIN-107 hard labels only. Checkpoint selection uses DEV-34 participant macro-F1 at a fixed 0.5 threshold. TEST is not prepared or opened.

Code: `scripts/students/participant_student/`

The training run automatically exports participant predictions, metrics, confusion/classification reports, parameter counts, checkpoint size, profiled GFLOPs, latency, throughput, and peak CUDA memory.

## Colab notebooks

Run in order:

1. `notebooks/01_text_teacher.ipynb`
2. `notebooks/02_audio_teacher.ipynb`
3. `notebooks/03_student_baseline.ipynb` — runnable now
4. `notebooks/04_kd.ipynb` — standard + reliability-aware KD, implemented after the no-KD baseline is accepted

The obsolete segment-level teacher/student code has been removed from `main`.

TEST must remain closed.
