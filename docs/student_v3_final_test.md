# v3 one-time final TEST-47 evaluation

Run only after `freeze_dev_selection.py` reports `DEV FREEZE: PASS`.

The final TEST evaluates **only the frozen selected Standard-KD student**. The frozen text and audio branches are internal feature encoders required by the student architecture; they do not receive TEST metrics.

## Stage 1: blind student inference

`prepare_final_test_blind.py` reads only `test_split_Depression_AVEC2017.csv` for the 47 participant identifiers. It does not load depression labels.

It:
- reconstructs TEST Participant-only documents
- applies the frozen v3 text encoder to obtain the student's 64-D text input
- extracts/reuses exact patient-only ComParE16 TEST features
- applies the frozen compact audio encoder to obtain the student's 256-D audio input
- applies the saved TRAIN-only embedding standardizers
- loads the frozen Standard-KD fusion checkpoint
- writes only the selected student's fixed-threshold probability and prediction
- hashes the frozen artifacts and records that TEST labels were not loaded

The unimodal branch probabilities are not saved for TEST evaluation. No teacher targets are computed on TEST.

## Stage 2: one-time student scoring

`score_final_test_once.py` verifies the blind prediction hash and DEV freeze, then opens `full_test_split.csv` and accepts `PHQ_Binary` or `PHQ8_Binary` as binary ground truth.

It computes metrics only for:
- frozen selected Standard-KD student

The threshold remains 0.5. There is no TEST calibration, checkpoint selection, architecture change, hyperparameter search, teacher evaluation, or unimodal branch evaluation.

After scoring it writes `FINAL_TEST_SCORED.json` and refuses repeat scoring.

No post-TEST model changes are permitted.
