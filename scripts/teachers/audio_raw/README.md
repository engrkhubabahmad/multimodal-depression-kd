# Participant-level audio teacher

Selected text teacher is locked in `configs/selected_text_teacher.json`:
`original_non_pagerank`, DEV macro-F1 0.7700, threshold 0.5.

Audio training uses the published Ping et al. preprocessing:

```text
TRAIN/DEV raw audio + transcripts
→ Participant speech intervals only
→ concatenate speech within participant
→ 80-bin log-Mel: n_fft=2048, hop=533
→ 60 s windows: 1800 frames; 10 s overlap: 1500 frames
→ row L2 normalization
→ released ConvLSTM_Audio backbone + binary head
→ participant-balanced clip sampling
→ mean-logit participant aggregation
→ DEV macro-F1 selection and freeze
```

TEST is not read. Participant 440 is excluded. Audio clips are cache files only; teacher labels, evaluation, and KD outputs are one row per participant.

## Colab

```python
!python -m scripts.teachers.solo_published.bootstrap

!python -m scripts.teachers.audio_raw.train_teacher \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --source-root /content/solo_teacher_sources/DepressionEstimation \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/audio_raw_teacher/seed103 \
  --epochs 80 --patience 15 --batch-size 8 --learning-rate 1e-4
```

Outputs are a disk feature cache, one participant prediction CSV for TRAIN and DEV, `frozen_audio_teacher.pt`, and `metrics.json`.