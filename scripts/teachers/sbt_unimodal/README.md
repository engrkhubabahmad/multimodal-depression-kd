# SBT released-loader unimodal adaptation

Source: https://github.com/ghy-yhg/SBT-Net (released dataset_loader.py and model.py).
This is NOT a reproduction of the published SBT-Net results. No author-trained
DAIC-WOZ checkpoint is supplied. Initial weights are generic pretrained
`facebook/wav2vec2-base` and `albert-base-v2`, pinned to Hub revisions.

## Files
- dataset_loader.py: librosa 16 kHz mono; first 15 seconds; no waveform z-score;
  ALBERT tokenizer truncation/padding to 128 tokens including special tokens.
- model.py: independent encoders and Linear(hidden,128), ReLU, Dropout(0.3),
  Linear(128,1) heads. Audio uses temporal mean; text uses CLS. These pooling
  choices are our unimodal adaptation, not released author unimodal heads.
- train.py: participant-disjoint teacher training, checkpoint selection, exports.
- test.py: DEV-only saved-prediction reporting. No TEST evaluation for teachers.
- cache.py: checksum-validated atomic disk shards for frozen-stage features.
- data.py: participant and aligned segment manifests; test IDs only.
- models.py: compatibility imports for earlier callers.

## Exactness boundary
The author's loader expects already prepared audio/text files per CSV row.
The raw DAIC conversion script is not released. We construct each participant's
speech using participant transcript intervals and join their usable text. This
construction (including removal of scrubbed/redacted turns) is ours. The loader
then uses only the FIRST 15 seconds of concatenated speech and FIRST 128 tokens.
There is one training sample per participant (107), not thousands of windows.
Later speech/text is intentionally discarded to match the released truncation.
Do not interpret this as proof that the paper used the same preparation.

For RA-PDS-KD, DEV selection and KD export apply the same input transformations
to aligned segments. This is a participant-to-segment input distribution shift;
DEV performance must establish whether these teachers are useful for KD.

SGCMG, BG-TPA, ETM and cross-modal attention are absent. KD itself does not require
removing these modules; separate unimodal teachers are the project design choice.

## Intentional training differences from the released demo
We retain TRAIN=107 / DEV=34 / TEST=47, TEST closed, batch 16, participant-balanced
sampling, aligned DEV macro-F1 selection (BCE tie break), and the existing two-stage
schedule with early stopping. The demo uses a random row split, batch 2, 20 epochs,
full-model AdamW and thresholded AUC. Those are not silently presented as equivalent.
Our S1 caches frozen pooled embeddings to disk; S2 fine-tunes upper encoder layers
from raw inputs. ALBERT shares layers: the last physical group is unfrozen.

## Cache and progress
Only batch features enter RAM. Encoder revision, code, source hashes, row content,
and preprocessing configuration determine cache identity. Earlier ALBERT-large or
normalized-audio features and checkpoints cannot be reused by this implementation.
Audio is grouped by exact valid length before encoding so padding cannot change
Wav2Vec2 group-normalization outputs. Variable lengths can reduce GPU throughput.
The notebook runs in its kernel with notebook tqdm bars. Do not launch training
via subprocess when native widget progress is required.

## Colab
Mount Drive and update the branch first, then run the teacher cell in
notebooks/RA_PDS_KD_pipeline.ipynb. After changing code in a live kernel, reload
modules as shown there. Do not run this cell while an earlier training job is active.
