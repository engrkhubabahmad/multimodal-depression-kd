# Locked teacher implementations

Only two teacher implementations are active.

## Audio
`ussd_audio/` reproduces the frozen Ravi et al. USSD run #4 ComParE16 + LSTM checkpoint. The accepted model is the released `md_35_epochs.pth`; optional depression-only adaptation was evaluated separately and rejected because it degraded DEV-34 performance.

Active sequence:
1. `bootstrap.py`
2. `prepare_compare16.py --split dev`
3. `audit_frozen.py`
4. `prepare_compare16.py --split train`
5. `audit_train_crop.py`

The final TRAIN KD file is `train_run4_crop_kd_targets.csv`.

## Text
`idiap_text/` loads the published Idiap participant InducT-GCN top-250 checkpoint and vectorizer. `export_targets.py` exports TRAIN-107 training-graph document-node probabilities/logits and DEV-34 saved-`A_dev` probabilities/logits.

No teacher script accesses TEST.
