# Exact v3 reproduction result: run_20260926_022900_UTC

Status: **PASS**

Protocol: TRAIN-107 / DEV-34, participant 440 excluded. TEST was not read or scored.

Core reproduction commit: `0d6ac621be7da1db7c1d769d075d142d0a740fd8`

## Result table

| Model | N | Accuracy | Balanced Accuracy | Macro-F1 | Depressed F1 | AUROC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| IDIAP text teacher - epoch 0 original | 34 | 0.8529 | 0.8676 | 0.8419 | 0.8000 | 0.7826 | 19 | 4 | 1 | 10 |
| USSD audio teacher - epoch 0 original | 34 | 0.8235 | 0.7747 | 0.7875 | 0.7000 | 0.6364 | 21 | 2 | 4 | 7 |
| v3 text branch | 34 | 0.7941 | 0.8241 | 0.7850 | 0.7407 | 0.8063 | 17 | 6 | 1 | 10 |
| v3 audio branch | 34 | 0.7059 | 0.6166 | 0.6222 | 0.4444 | 0.5257 | 20 | 3 | 7 | 4 |
| No-KD multimodal | 34 | 0.7353 | 0.7095 | 0.7043 | 0.6087 | 0.7470 | 18 | 5 | 4 | 7 |
| Standard KD | 34 | 0.7353 | 0.7569 | 0.7236 | 0.6667 | 0.7668 | 16 | 7 | 2 | 9 |


## Included artifacts

- `comparative_table.csv` and `comparative_table.md`
- `predictions/*_dev34.csv`: participant-level DEV predictions used to recompute every metric
- `confusion_matrices/*.csv`, `*.png` (600 DPI), and `*.pdf` (600 DPI)
- `classification_reports/*.csv`
- `confusion_matrices.json`
- `classification_reports.json`
- `source_metrics/*.json`: original teacher/branch/fusion metric files
- `01_text_teacher/teacher_selection.json` and `02_audio_teacher/teacher_selection.json` remain in the Drive run and record that the selected teachers are the original epoch-0 checkpoints
- `provenance/run_manifest.json`
- `provenance/reproduction_summary.json`

Large checkpoints, extracted ComParE16 features, intermediate arrays and training histories remain in the corresponding Drive run folder and are intentionally not committed to Git.
