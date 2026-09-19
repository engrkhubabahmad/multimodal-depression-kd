# USSD ComParE16 audio teacher

This module implements the locked audio-teacher decision:

- Ravi et al. USSD, audio only
- ComParE16 + LSTM-only `CustomComparE16`
- released run **#4**
- released checkpoint **md_35_epochs.pth**
- pinned author repository commit `c3e68649153004ed2174878a1c54c716ad26cfd7`

The paper/repository majority-vote result (~0.776) is a five-run ensemble result and is **not** assigned to this checkpoint. The author's saved run #4 canonical DEV-35 macro-F1 is **0.8011** and is kept only as a reference. Our authoritative frozen result is DEV-34 with participant 440 excluded.

## Leakage contract

TRAIN = 107 participants. DEV = 34 participants. Participant 440 is excluded. TEST is never discovered or loaded by these scripts. Frozen author inference happens before any fine-tuning. Fine-tuning uses TRAIN-107 only; checkpoint selection is DEV-34 participant macro-F1 only at the fixed author hard aggregation. Ties keep the earliest epoch.

## Exact preprocessing contract

Raw DAIC-WOZ -> author-style Participant timing extraction (including published interruption/misalignment corrections) -> concatenated patient-only PCM16 WAV -> author-style `SMILExtract -C ComParE_2016.conf` -> drop `name` and `frameTime` -> preserve the author's flatten/reshape behavior -> 130 x 384 non-overlapping segments with zero-padding -> released run #4 `data_saver.pickle` mean/std normalization.

There is deliberately **no Python openSMILE fallback**. If `SMILExtract`, the ComParE16 config, checkpoint, or saved normalization artifact is missing, the run stops.

The author hard participant prediction is reproduced separately from KD soft targets:

- author metric path: `np.rint(segment_probability)` -> participant vote fraction -> `np.rint(vote_fraction)`
- KD soft path: mean segment probability -> `logit(mean_probability)`

## Colab sequence

```bash
# 1) Pin author code + released run #4 assets
python -m scripts.teachers.ussd_audio.bootstrap \
  --root /content/solo_teacher_sources/USSD-depression

# 2) Prepare DEV only first. Use the actual SMILExtract binary + ComParE_2016.conf.
python -m scripts.teachers.ussd_audio.prepare_compare16 \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --split dev \
  --smile-extract /path/to/SMILExtract \
  --compare16-config /path/to/ComParE_2016.conf

# 3) Frozen checkpoint audit. No fitting.
python -m scripts.teachers.ussd_audio.audit_frozen \
  --features /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --author-root /content/solo_teacher_sources/USSD-depression \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_frozen_audit
```

Only after the frozen audit is accepted:

```bash
# 4) Add TRAIN features to the same cache
python -m scripts.teachers.ussd_audio.prepare_compare16 \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --split train \
  --smile-extract /path/to/SMILExtract \
  --compare16-config /path/to/ComParE_2016.conf

# 5) TRAIN-only adaptation, DEV-only model selection, no TEST
python -m scripts.teachers.ussd_audio.finetune \
  --features /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --author-root /content/solo_teacher_sources/USSD-depression \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_finetuned
```

`train_kd_targets.csv` contains participant-level `kd_probability` and `kd_logit` for the student, while retaining the author's hard-vote fields for reproducibility.

### Fine-tuning caveat

The released frozen checkpoint is fully usable for inference. The original full USSD training objective also referenced an external speaker-embedding artifact that is not released in the repository. Therefore this implementation fine-tunes the released depression network with depression loss only and labels it **TRAIN-only adaptation**, not an exact reproduction of the auxiliary speaker-disentanglement training objective.
