# Participant-level ReLiMP-Net student

Active no-KD student path.

## Input
Per participant:
- aligned Participant-only audio/text segments
- max 128 segments
- max 10 s audio per segment, min 0.5 s
- audio: 64-bin log-Mel mean + std = 128-D/segment
- text: TRAIN-only vocabulary, max 64 token IDs/segment
- segment mask for variable interview length

## Architecture
Audio MLP + one-layer 4-head text Transformer -> per-segment fusion -> masked attention pooling across participant segments -> participant binary classifier.

## Protocol
TRAIN-107 only for fitting. DEV-34 (440 excluded) for checkpoint selection at fixed threshold 0.5. No teacher targets in the baseline. TEST is not prepared or opened.

Outputs include standardized participant predictions, metrics, confusion/classification reports, checkpoint, training history, and model-complexity profile.
