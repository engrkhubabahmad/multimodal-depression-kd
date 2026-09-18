# reliability-aware-depression-kd

## Active methodology: RA-PDS-KD

**Reliability-Aware Participant-Disjoint Segment-Level Knowledge Distillation**

The active protocol splits DAIC-WOZ by participant **before** segmentation, then trains and evaluates the audio and text teachers at **segment level only**. Audio/text segments share an aligned `segment_id`, enabling per-segment reliability-aware KD without participant aggregation.

- Train: 107 participants
- Dev: 34 usable participants (participant 440 excluded because the original files are corrupted)
- Test: 47 participants, reserved until the final student is frozen
- Teacher evaluation: DEV segments only
- Final TEST evaluation: final frozen student only

See `docs/RA_PDS_KD_METHOD.md`.

## Active Colab workflow

1. `notebooks/01_daic_woz_setup.ipynb` — dataset/setup checks.
2. `notebooks/02_train_segment_teachers.ipynb` — aligned segmentation, dual feature caches, audio/text segment teachers, DEV segment evaluation.
3. `notebooks/03_build_reliability_kd_targets.ipynb` — build TRAIN/DEV reliability-aware dual-teacher targets. TEST remains closed.

Reusable implementation lives under `scripts/`; notebooks stay thin runners.

## Active scripts

- `scripts/teachers/train_segment_teachers.py`
  - official participant split first;
  - aligned audio/text segment IDs;
  - Wav2Vec2 audio cache + MiniLM text cache;
  - segment-level teacher training and TRAIN/DEV predictions;
  - no TEST feature extraction/evaluation.

- `scripts/kd/build_reliability_targets.py`
  - merges aligned audio/text teacher probabilities;
  - computes per-segment modality reliability;
  - saves TRAIN/DEV KD targets only.

- `scripts/kd/reliability_loss.py`
  - hard-label BCE + reliability-weighted dual-teacher binary KD loss for the segment-level student.

## Legacy baselines

The following are retained for reproducibility of earlier participant-level experiments and are **not** the active RA-PDS-KD protocol:

- `notebooks/02_train_frozen_teachers.ipynb`
- `notebooks/03_finetune_full_teachers.ipynb`
- `scripts/teachers/train_frozen_teachers.py`
- `scripts/teachers/train_full_teachers_10epochs.py`

Do not compare their participant-level metrics directly with the active segment-level metrics.
