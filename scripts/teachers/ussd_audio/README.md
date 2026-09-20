# USSD ComParE16 audio teacher

Locked audio teacher:

- Ravi et al. USSD
- ComParE16 + LSTM-only
- released run #4
- checkpoint `md_35_epochs.pth`
- pinned author commit `c3e68649153004ed2174878a1c54c716ad26cfd7`

The five-run majority-vote paper result is not assigned to this checkpoint. The independently reproduced authoritative result is DEV-34 macro-F1 **0.7875**, with participant 440 excluded.

## Active path

```bash
python -m scripts.teachers.ussd_audio.bootstrap --root /content/drive/MyDrive/tools/solo_teacher_sources/USSD-depression

python -m scripts.teachers.ussd_audio.prepare_compare16 \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --split dev \
  --smile-extract /path/to/SMILExtract \
  --compare16-config /path/to/ComParE_2016.conf

python -m scripts.teachers.ussd_audio.audit_frozen \
  --features /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --author-root /content/drive/MyDrive/tools/solo_teacher_sources/USSD-depression \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_frozen_audit

python -m scripts.teachers.ussd_audio.prepare_compare16 \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --split train \
  --smile-extract /path/to/SMILExtract \
  --compare16-config /path/to/ComParE_2016.conf

python -m scripts.teachers.ussd_audio.audit_train_crop \
  --features /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_compare16 \
  --author-root /content/drive/MyDrive/tools/solo_teacher_sources/USSD-depression \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/ussd_audio_teacher_final
```

TRAIN export uses the author-style run-4 seed 1300 contiguous 6662-frame crop, then 384-frame segmentation. The final KD target is participant-level mean segment probability plus its binary logit.

Fine-tuning is intentionally absent from the active implementation because the tested depression-only adaptation did not improve DEV-34. TEST is never opened.
