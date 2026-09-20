# Rich compact participant student v2

This replaces the ultra-compressed v0 student as the active architecture candidate.

## Why v2 exists

v0 reduced each raw-audio segment to 64 log-Mel means + 64 standard deviations before learning. That discarded temporal dynamics and produced weak DEV-34 performance.

v2 uses richer, teacher-grade **inputs** without using teacher predictions or hidden states:

- Audio: existing patient-only ComParE16 sequence files already prepared for the frozen USSD audit.
- Text: an independently fitted TRAIN-107 top-250 TF-IDF document vector. The frozen Idiap vectorizer is not reused.

## Model

Per 384-frame ComParE16 segment:

1. 130 -> 96 temporal Conv1D, stride 2
2. depthwise-separable temporal Conv1D, stride 2
3. one-layer 64-unit GRU
4. 96-D segment projection

Participant text:

1. top-250 TF-IDF
2. 250 -> 128 -> 96 MLP

Fusion:

- text-conditioned attention over up to 32 acoustic segments
- concatenate audio, text, elementwise product, absolute difference
- 384 -> 128 -> 32 -> 1 classifier

Expected trainable size is about 260k parameters, far below the frozen USSD audio teacher (1,153,537 parameters).

No-KD training loads no teacher probabilities. TEST is not prepared or opened.
