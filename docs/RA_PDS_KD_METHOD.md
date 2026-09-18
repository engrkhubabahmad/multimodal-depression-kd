# RA-PDS-KD Methodology

**Name:** Reliability-Aware Participant-Disjoint Segment-Level Knowledge Distillation (RA-PDS-KD)

## Protocol

```text
DAIC-WOZ
   |
   v
PARTICIPANT-LEVEL SPLIT FIRST
   |
   +-- TRAIN: 107 participants
   +-- DEV:    34 participants (participant 440 excluded: corrupted original files)
   +-- TEST:   47 participants -> CLOSED until the final student is frozen
   |
   v
SEGMENT TRAIN/DEV INDEPENDENTLY
   |
   +-- aligned AUDIO segment (<=10 s participant speech)
   +-- aligned TEXT segment/chunk (<=254 tokens)
   |
   +-- audio cache -> Wav2Vec2 embedding
   +-- text cache  -> MiniLM embedding
   |
   v
SEGMENT-LEVEL AUDIO TEACHER + TEXT TEACHER
   |
   v
DEV SEGMENT EVALUATION / TEACHER SELECTION
   |
   v
TRAIN-SEGMENT TEACHER PROBABILITIES
   |
   +-- audio reliability
   +-- text reliability
   |
   v
RELIABILITY-AWARE FUSION
   |
   v
hard-label loss + reliability-weighted KD loss
   |
   v
ReLiMP-Net student (segment-level)
   |
   v
DEV SEGMENTS ONLY for checkpoint/hyperparameter selection
   |
   v
FREEZE FINAL STUDENT
   |
   v
TEST SEGMENTS -> ONE FINAL SEGMENT-LEVEL EVALUATION
```

## Leakage rule

The participant split is fixed before any segmentation. Every segment inherits the participant's split. No participant can contribute segments to more than one split.

## Segment alignment

The active segmenter creates a shared `segment_id` for audio and text. Long participant turns are partitioned into the minimum number of aligned pieces needed to satisfy both the audio-duration and text-token limits. This gives one audio teacher probability and one text teacher probability for the same segment, which is required for per-segment reliability-aware KD.

## Cache policy

Audio and text are cached separately under one cache configuration:

- Audio cache validity: waveform SHA256 + transcript SHA256 + exact aligned `segment_id` list.
- Text cache validity: transcript SHA256 + exact aligned `segment_id` list.
- Cache key includes segmentation settings, immutable Hugging Face model revisions, and package versions.
- Changing only the classifier reuses both feature caches.
- Changing a transcript invalidates both modalities because segment boundaries and IDs are transcript-defined.
- Changing only the waveform invalidates audio but preserves text.

## Evaluation policy

Teachers are evaluated at **segment level only** on DEV. There is no participant-level aggregation or participant-level metric.

The TEST split is not used for teacher training, teacher evaluation, teacher selection, reliability fitting, KD tuning, threshold tuning, or student checkpoint selection. TEST is evaluated only after the final student is frozen.

## Reliability baseline

The current transparent baseline is uncertainty-based per-segment reliability:

`r = 1 - H(p) / log(2)`

where `H(p)` is binary entropy. Audio/text reliabilities are normalized to produce per-segment KD weights. This module is isolated so a learned reliability estimator can replace it later without changing the split/cache/segment contract.
