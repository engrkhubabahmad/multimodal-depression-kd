# SBT-Net unimodal teachers for RA-PDS-KD

This folder adapts the published SBT-Net unimodal evidence into two independent KD teachers:

- audio: `facebook/wav2vec2-base` + released-style 128-unit classifier head
- text: `albert-large-v2` + released-style 128-unit classifier head

## Why this is an adaptation, not a byte-for-byte reproduction

The SBT-Net paper reports DAIC-WOZ unimodal F1 values of 0.91 (audio) and 0.87 (text), and specifies wav2vec2.0 plus ALBERT-large. The public repository, however, only releases a multimodal demo model. That demo uses `facebook/wav2vec2-base` and `albert-base-v2`, imports two modules that are not present in the repository, and does not release the exact unimodal heads or depression-trained checkpoints.

For scientific traceability, the code therefore does not claim that the authors' 0.91/0.87 checkpoints are being reused. It uses the published/released backbone choices as transfer-learning initialization and trains modality-specific depression teachers on the RA-PDS-KD TRAIN participants.

## Protocol

Teacher training units are allowed to differ from the student/KD segments:

- audio teacher training: participant-only speech chunks up to 15 s
- text teacher training: participant-only transcript chunks up to 128 ALBERT tokens
- KD export: the existing aligned RA-PDS-KD segments (<=10 s and <=254 segmentation tokens)

The teachers are trained only on the 107 TRAIN participants, selected on the 34 usable DEV participants, then frozen. TEST remains closed. The frozen teachers export probabilities/logits for the identical aligned TRAIN/DEV `segment_id`s used by downstream KD.

The implementation follows the paper where reproducible: weighted sampling, staged encoder freezing, top-encoder fine-tuning at a lower learning rate, and DEV macro-F1 checkpoint selection. It preserves participant-balanced sampling so participants with many speech/text chunks do not dominate optimization.

## Run

The active Colab notebook calls `scripts/teachers/sbt_unimodal/train.py`.

Valid checkpoints are reused automatically. Pass `--force-retrain` only when a deliberate retraining is required.
