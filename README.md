# Exact v3 reproduction

This branch is deliberately narrow. The default `main` pipeline reproduces the frozen v3 TRAIN-107 / DEV-34 experiment only.

## Goal

Reproduce, from a fresh run folder, the exact v3 path:

1. IDIAP InducT-GCN text teacher targets from its pinned published checkpoint.
2. USSD ComParE16 + LSTM run-4 audio teacher from raw DAIC-WOZ audio/transcripts using openSMILE 3.0.2.
3. Independent v3 text and audio student branches.
4. No-KD multimodal baseline.
5. Standard KD multimodal student.
6. Verification against the frozen DEV-34 results.

RA-KD, final TEST scoring, full188/internal188 experiments, StepAudio2, NUSD, WavLM and other exploratory code are intentionally absent from `main`.

## Preserved history

The previous main branch is preserved intact at:

`archive/pre-exact-repro-main-20260926`

Existing experiment branches and all Google Drive experiment folders are left untouched.

## Frozen protocol

- TRAIN: 107 participants
- DEV: 34 participants
- Participant 440 excluded from DEV
- Threshold: 0.5
- TEST is never read or scored by this pipeline
- Standard KD: temperature 2.0, KD weight 0.5
- Fusion seed: 103

Expected DEV-34 results:

| Model | Macro-F1 | Depressed F1 | AUROC |
|---|---:|---:|---:|
| v3 text branch | 0.7850 | 0.7407 | 0.8063 |
| No-KD multimodal | 0.7043 | 0.6087 | 0.7470 |
| Standard KD | 0.7236 | 0.6667 | 0.7668 |

Teacher checks:

- IDIAP text teacher DEV-34 Macro-F1: 0.8418604651
- USSD audio teacher DEV-34 Macro-F1: 0.7875

## Drive safety

Every reproduction starts in a brand-new folder:

`/content/drive/MyDrive/DAIC_WOZ/experiments/exact_v3_reproduction/<run_id>/`

Historical experiment folders are read-only reference material and are never overwritten.

Run notebooks in numerical order.

## Git result packages

After notebooks 00–05 succeed, notebook 06 verifies the frozen targets and automatically publishes a compact, independently checkable result package to:

`results/<run_id>/`

Each result package includes the DEV-34 participant predictions, comparative metrics table, confusion matrices (CSV + PNG at 600 DPI + PDF at 600 DPI), classification reports (CSV + JSON), source metric/audit JSON, and run provenance. Large checkpoints/features remain only in the fresh Drive run folder.


## Teacher epoch-0 selection

For both external teachers, **epoch 0 means the original published/released checkpoint before any local fine-tuning**.

- IDIAP text teacher: original published InducT-GCN checkpoint.
- USSD audio teacher: original released run-4 `md_35_epochs.pth` checkpoint.

Selection always starts from epoch 0. A fine-tuned teacher may replace it only if it strictly improves the fixed DEV lexicographic key:

1. Macro-F1
2. Depressed-class F1
3. AUROC

If fine-tuning does not improve that key, the original epoch-0 teacher remains selected. The selected TRAIN KD targets are exposed through `selected_train_kd_targets.csv`, and Standard KD reads those selected aliases rather than assuming a fine-tuned model.

The compact v3 student branches are separate architectures. Their untrained state is saved only as `random_init.*` diagnostics and is never called the original teacher epoch 0.

For exact v3 audio-branch reproduction, the locked compact-student recipe remains batch size **20**, seed **1300**, and historical selected epoch **51**. Notebook 04 refuses to start fusion unless that frozen audio-branch result is reproduced.
