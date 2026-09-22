# v3 one-time final TEST-47 evaluation

The final TEST evaluates **only the frozen selected Standard-KD student**.

The text and audio branches are internal encoders required to construct the student's frozen input representation. They are not TEST evaluation targets and do not receive TEST metrics.

## Stage 1: blind inference

Blind inference may have been produced by either the original script, which also saved diagnostic unimodal probabilities, or the revised student-only script. In either case TEST labels remain unopened.

## Stage 1b: freeze student-only predictions

Before scoring, run `freeze_student_only_test_predictions.py`.

This utility:
- reads the already-frozen blind TEST prediction CSV
- copies only the selected Standard-KD student's probability and fixed-threshold prediction
- writes `blind_student_only_predictions.csv`
- records SHA-256 hashes of the source and sanitized files
- never reads a TEST label file
- preserves the original blind file unchanged

This avoids repeating expensive ComParE16 extraction when a legacy blind inference file exists.

## Stage 2: one-time student scoring

`score_final_test_once.py` refuses to score the legacy diagnostic file. It accepts only:
- `blind_student_only_predictions.csv`
- `blind_student_only_manifest.json`

Only after verifying those frozen hashes and the DEV freeze does it open `full_test_split.csv`.

It computes metrics only for the selected frozen Standard-KD student at threshold 0.5. It does not compute teacher, text-branch, or audio-branch TEST metrics.

After scoring, no model, threshold, architecture, reliability rule, checkpoint, or hyperparameter changes are permitted.
