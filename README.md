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
