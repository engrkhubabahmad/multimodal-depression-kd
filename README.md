# reliability-aware-depression-kd

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

1. SBT-Net unimodal audio/text teachers
   - audio teacher: Wav2Vec2 transfer learning; text teacher: ALBERT-large transfer learning
   - teacher-training chunks may differ from KD segments
   - frozen teachers export logits/probabilities for the existing aligned TRAIN/DEV segment IDs
   - participant-balanced weighted sampling and staged encoder fine-tuning
   - DEV checkpoint selection only; valid checkpoints are reused automatically
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

- `scripts/teachers/sbt_unimodal/train.py`
- `scripts/teachers/sbt_unimodal/{data.py,models.py,README.md}`
- `scripts/kd/build_reliability_targets.py`
- `scripts/kd/reliability_loss.py`
- `scripts/students/data_utils.py`
- `scripts/students/relimpnet_segment.py`
- `scripts/students/prepare_segment_features.py`
- `scripts/students/train_segment_students.py`
- `scripts/students/evaluate_final_clean_test.py`

The active teachers are independent SBT-Net-inspired unimodal models. The released SBT-Net repository does not contain the exact published unimodal heads or depression-trained checkpoints, so this project records that provenance explicitly and retrains modality-specific teachers from the published/released backbones before exporting aligned KD targets.
