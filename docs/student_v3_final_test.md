# v3 one-time final TEST-47 evaluation

Run only after `freeze_dev_selection.py` reports `DEV FREEZE: PASS`.

The final evaluation is deliberately split into two stages.

## Stage 1: blind inference

`prepare_final_test_blind.py` reads only `test_split_Depression_AVEC2017.csv` for the 47 participant identifiers. It does not load depression labels.

It then:
- reconstructs TEST Participant-only documents
- applies the frozen v3 text vectorizer and frozen text inference state
- extracts/reuses exact patient-only ComParE16 TEST features
- applies the frozen compact audio branch and author TRAIN normalization
- applies the saved TRAIN-only embedding standardizers
- loads the frozen Standard-KD fusion checkpoint
- writes fixed-threshold blind predictions
- hashes all frozen artifacts and records that labels were not loaded

No teacher targets are computed on TEST.

## Stage 2: one-time scoring

`score_final_test_once.py` verifies the blind prediction hash and DEV freeze, then opens `full_test_split.csv` and accepts `PHQ_Binary` (or `PHQ8_Binary` if present) as the binary ground truth.

It scores, simultaneously and without selection:
- frozen selected Standard-KD multimodal student
- frozen text-only branch
- frozen audio branch

The selected Standard-KD threshold remains 0.5. There is no TEST calibration, checkpoint selection, architecture change, or hyperparameter search.

After scoring it writes `FINAL_TEST_SCORED.json` and refuses repeat scoring.

No post-TEST model changes are permitted.
