# Idiap participant InducT-GCN text teacher

Locked text teacher:

- participant-only InducT-GCN
- top-250 features
- published checkpoint `model/Participant/model_inductgcn[250].pkl`
- published vectorizer `model/Participant/vtzer_inductgcn[250].pkl`

`export_targets.py` exports:

- TRAIN-107: checkpoint training-graph document-node probability and binary logit
- DEV-34: checkpoint saved `A_dev`, excluding participant 440
- no TEST output

Verified DEV-34 macro-F1: **0.8418604651**.

```bash
python -m scripts.teachers.idiap_text.export_targets \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --source-root /content/drive/MyDrive/tools/solo_teacher_sources/bias_in_daic-woz \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/idiap_text_teacher
```

For binary KD, `text_logit = log(p/(1-p))`, where `p` is the checkpoint probability of the positive/depressed class.
