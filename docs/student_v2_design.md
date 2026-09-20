# Student v2 design audit

## Objective

The v0 student was intentionally lightweight but compressed each 10-second raw-audio segment to 64 log-Mel means plus 64 standard deviations before learning. On DEV-34 it reached macro-F1 0.6092 and the equal Standard-KD v0 run fell to 0.5729. The v2 redesign improves information quality while reducing trainable parameters.

## Locked teacher facts used to design v2

### USSD audio teacher

Active implementation: `scripts/teachers/ussd_audio/common.py::CustomComparE16`.

Input per segment: 130 ComParE16 channels × 384 frames.

Architecture:
- Conv1D 130 → 256, kernel 3
- BatchNorm + ReLU
- MaxPool 3
- 2-layer unidirectional LSTM, 256 hidden units
- 256 → 1 output

Exact trainable parameters from the implemented state-dict-compatible model: **1,153,537**.

The v2 student reuses only the already cached **input ComParE16 arrays**. It does not use teacher hidden states or predictions in the no-KD baseline.

Source: https://github.com/vijaysumaravi/USSD-depression

### Idiap participant text teacher

The frozen top-250 InducT-GCN uses:
- TF-IDF participant documents
- top-250 feature selection
- 250 → 64 graph projection
- 64 → 2 output

With bias disabled in both linear layers, this is **16,128 trainable parameters** (250×64 + 64×2).

The v2 student independently refits its own TRAIN-only TF-IDF + top-250 selection. It does not load the teacher vectorizer or checkpoint for the no-KD baseline.

Source: https://github.com/idiap/bias_in_daic-woz

## External architecture check

A 2025 lightweight multimodal depression-detection paper also preserves Mel-spectrogram time structure for its audio branch and uses a pretrained textual representation with cross-modal fusion rather than reducing audio to global mean/std before learning:

E. Lim et al., *A lightweight approach based on cross-modality for depression detection*, Computers in Biology and Medicine 186 (2025) 109618.
Code: https://github.com/Sclab-Projects-2023/depression-detection

We do not copy its model or reported score. It is used only as an architecture sanity check.

## v2 model

Audio:
- cached 130×384 ComParE16 segments
- 130 → 96 temporal Conv1D, stride 2
- 96-channel depthwise-separable temporal Conv1D, stride 2
- one-layer GRU, hidden 64
- 96-D segment representation

Text:
- independent TRAIN-only top-250 TF-IDF
- top-250 TF-IDF + fixed TRAIN-only positive-PMI graph diffusion\n- concatenate original + graph-smoothed TF-IDF\n- 500 → 128 → 96 MLP

Fusion:
- text-conditioned acoustic-segment attention
- concatenate audio, text, elementwise product, absolute difference
- 384 → 128 → 32 → 1

Expected trainable parameter count from the implementation: about **260k**, approximately 23% of the audio teacher.

## Protocol safeguards

- TRAIN 107 only for fitting preprocessing/model parameters.
- DEV 34, participant 440 excluded, for checkpoint selection.
- threshold fixed at 0.5.
- TEST not prepared or opened.
- No teacher probability/logit/hidden-state use in the v2 no-KD baseline.
- Student audio normalization is fitted from TRAIN-107 only.
- Student text vocabulary/feature selection is fitted from TRAIN-107 only.
