# reliability-aware-depression-kd

> **USSD audio-teacher branch:** This branch contains the locked Ravi et al. USSD ComParE16 + LSTM-only run #4 frozen-checkpoint audit under `scripts/teachers/ussd_audio/`. For this branch, do **not** use the older Wav2Vec2 segment-teacher path below as the audio teacher. First reproduce `md_35_epochs.pth` on DEV-34 (440 excluded); only after that audit is accepted may TRAIN-107 adaptation run. TEST-47 remains closed.

## Active methodology: RA-PDS-KD

**Reliability-Aware Participant-Disjoint Segment-Level Knowledge Distillation**

This repository now has one active Colab workflow:

`notebooks/RA_PDS_KD_pipeline.ipynb`

Run it top-to-bottom in one Colab runtime. Reusable implementation stays under `scripts/`; experiment data/caches/checkpoints stay in Google Drive.

### Data protocol

- TRAIN: 107 participants
- DEV: 34 usable participants (participant 440 excluded because its original files are corrupted)
- TEST: 47 participants, closed until the final student is frozen
- Participant split happens **before** segmentation
- Audio/text use aligned segment IDs
- Teacher and student evaluation are **segment-level only**
- No participant-level prediction aggregation
- Missing/noisy robustness experiments use TRAIN/DEV only
- Final TEST is **clean-only** using the frozen final student and fixed DEV threshold

### Pipeline

1. Segment-level audio/text teachers
   - aligned TRAIN/DEV segments
   - modality-specific caches under `experiments/features/rapdskd_segments_<hash>/`
   - Wav2Vec2 audio embeddings + MiniLM text embeddings
   - DEV teacher evaluation only
2. Aligned TRAIN/DEV teacher probabilities and confidence priors
3. Lightweight ReLiMP-Net TRAIN/DEV segment features
4. Three student experiments:
   - No KD
   - Standard KD
   - Reliability-Aware Robust KD
5. DEV robustness evaluation:
   - Clean
   - Audio missing
   - Text missing
   - Audio noisy
   - Text noisy
6. Freeze final student/checkpoint/DEV threshold
7. Open TEST once and run **clean TEST only**

### Reliability rule

For Student 3:

`r_audio = audio_quality × audio_teacher_confidence`

`r_text = text_quality × text_teacher_confidence`

Missing modality quality is 0. Clean quality is 1. Noisy quality is reduced according to corruption severity. Normalized reliabilities weight the two teacher targets during KD. Reliability-aware feature fusion at student inference uses modality quality and does not require teacher inference.

### Active code

- `scripts/teachers/train_segment_teachers.py`
- `scripts/kd/build_reliability_targets.py`
- `scripts/kd/reliability_loss.py`
- `scripts/students/data_utils.py`
- `scripts/students/relimpnet_segment.py`
- `scripts/students/prepare_segment_features.py`
- `scripts/students/train_segment_students.py`
- `scripts/students/evaluate_final_clean_test.py`

The old participant-level frozen/full Facebook/Wav2Vec2 teacher runners were removed to prevent accidental use of the obsolete `teacher_v2_*` cache path. The active segment teacher still uses the pretrained Wav2Vec2 encoder for **segment embeddings**, but it is a different participant-disjoint segment-level pipeline and writes only `rapdskd_segments_*` caches.
