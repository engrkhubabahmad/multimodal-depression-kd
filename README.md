# reliability-aware-depression-kd

## Active methodology: RA-PDS-KD

**Reliability-Aware Participant-Disjoint Segment-Level Knowledge Distillation**

The active protocol splits DAIC-WOZ by participant **before** segmentation, trains audio/text teachers at **segment level**, and compares three lightweight student variants. Robustness to missing/noisy modalities is trained/evaluated on TRAIN/DEV only. The held-out TEST split is opened only after the final student is frozen and is evaluated under the **clean condition only**.

- Train: 107 participants
- Dev: 34 usable participants (participant 440 excluded because the original files are corrupted)
- Test: 47 participants, closed until the final student is frozen
- Teacher evaluation: DEV segments only
- Student robustness evaluation: DEV only
- Final TEST: clean-only, frozen final student, fixed DEV threshold

See `docs/RA_PDS_KD_METHOD.md`.

## Active Colab workflow

1. `notebooks/01_daic_woz_setup.ipynb` — dataset/setup checks.
2. `notebooks/02_train_segment_teachers.ipynb` — aligned segmentation, dual caches, audio/text segment teachers, DEV teacher evaluation.
3. `notebooks/03_build_reliability_kd_targets.ipynb` — aligned TRAIN/DEV teacher probabilities + confidence priors. TEST remains closed.
4. `notebooks/04_prepare_segment_student_features.ipynb` — clean/noisy student features for TRAIN/DEV only; vocabulary/scaler fit on TRAIN only.
5. `notebooks/05_train_segment_students.ipynb` — train the 3 student variants and run DEV-only robustness evaluation.
6. `notebooks/06_final_clean_test.ipynb` — open TEST only after final selection; evaluate the frozen final student on clean TEST segments only.

Reusable implementation lives under `scripts/`; notebooks stay thin runners.

## Three student experiments

- **Student 1 — No KD:** hard labels, clean training input only.
- **Student 2 — Standard KD:** hard labels + equal-weight audio/text soft targets, clean training input only.
- **Student 3 — Reliability-Aware Robust KD:** hard labels + reliability-weighted audio/text soft targets, trained with clean, audio-missing, text-missing, audio-noisy, and text-noisy conditions.

All three students share the same lightweight segment-level ReLiMP-Net backbone. All three are evaluated on DEV under the same five conditions. Checkpoint/threshold selection uses **clean DEV only**. Missing/noisy results are robustness analyses and never touch TEST.

## Active scripts

- `scripts/teachers/train_segment_teachers.py` — participant-disjoint aligned segment teachers and modality-specific Wav2Vec2/MiniLM caches.
- `scripts/kd/build_reliability_targets.py` — aligned TRAIN/DEV dual-teacher probabilities and entropy-confidence priors; no TEST targets.
- `scripts/kd/reliability_loss.py` — standard KD and availability/quality-aware KD losses.
- `scripts/students/data_utils.py` — shared segmentation and student feature utilities.
- `scripts/students/prepare_segment_features.py` — TRAIN/DEV ReLiMP-Net features only; includes configurable synthetic audio/text noise variants.
- `scripts/students/relimpnet_segment.py` — lightweight segment-level ReLiMP-Net.
- `scripts/students/train_segment_students.py` — three student experiments + DEV-only robustness evaluation.
- `scripts/students/evaluate_final_clean_test.py` — one-time clean TEST evaluation, no teacher query and no threshold search.

## Reliability rule

For Student 3, KD reliability is explicitly tied to modality **availability and quality**:

`r_audio = audio_quality × audio_teacher_confidence`

`r_text = text_quality × text_teacher_confidence`

Missing modality quality is `0`. Noisy modality quality is reduced according to the configured corruption severity. Clean quality is `1`. The normalized reliability values weight the two teacher soft targets during KD. At student inference, modality-quality weights are used for reliability-aware feature fusion; no teacher output is required.

## Legacy baselines

The following participant-level experiments are retained only for reproducibility and are not the active RA-PDS-KD protocol:

- `notebooks/02_train_frozen_teachers.ipynb`
- `notebooks/03_finetune_full_teachers.ipynb`
- `scripts/teachers/train_frozen_teachers.py`
- `scripts/teachers/train_full_teachers_10epochs.py`

Do not compare their participant-level metrics directly with active segment-level metrics.
