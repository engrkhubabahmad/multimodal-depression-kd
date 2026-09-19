# DAIC-WOZ solo published teachers

This branch starts from licensed raw DAIC-WOZ files and keeps each published
teacher independent. It does not use the RA-PDS-KD segment cache, SBT code, or
the test split.

## Frozen text teacher (ready)

The text teacher is Idiap's participant-only InducT-GCN from
`idiap/bias_in_daic-woz`. The bootstrap records the exact resolved commit. Its
published model and fitted TF-IDF vectorizer are used unchanged.
Preprocessing follows the authors' contract: concatenate every non-scrubbed
`Participant` utterance, in transcript order, into one document per interview.

Published official DEV-35 result: accuracy 30/35, macro-F1 0.8493. The local
DEV-34 result is a new result because participant 440 is unavailable.

## Frozen audio teacher (checkpoint-gated)

Audio uses Ping et al.'s `DepressionEstimation` implementation and preprocessing:
Participant speech only, 60-second windows with 10-second overlap, 80-bin
log-Mel spectrogram, `n_fft=2048`, `hop_length=533`, and row normalization.

The public GitHub repository does not contain the `.pt` file. It points to a
Google Drive folder and its audio config names
`A+Conv1D-BiLSTM+PHQ-Subscores+Mel+NoGB_2022-03-25_103243_f1_score-0.5532.pt`.
The runner requires that file and validates that it contains both `audio_net`
and `evaluator`. It never substitutes random weights.

## Colab

```python
!pip -q install torch torchvision pandas numpy scipy scikit-learn scikit-image librosa tqdm optuna matplotlib tensorboard
!python -m scripts.teachers.solo_published.bootstrap
!python -m scripts.teachers.solo_published.run_text \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/solo_teachers/text
```

For audio, first put the author checkpoint on Drive, then run:

```python
!python -m scripts.teachers.solo_published.prepare_audio \
  --daic-root /content/drive/MyDrive/DAIC_WOZ \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/solo_teachers/audio_features

!python -m scripts.teachers.solo_published.run_audio \
  --source-root /content/solo_teacher_sources/DepressionEstimation \
  --features /content/drive/MyDrive/DAIC_WOZ/experiments/solo_teachers/audio_features \
  --checkpoint /content/drive/MyDrive/DAIC_WOZ/checkpoints/A+Conv1D-BiLSTM+PHQ-Subscores+Mel+NoGB_2022-03-25_103243_f1_score-0.5532.pt \
  --output /content/drive/MyDrive/DAIC_WOZ/experiments/solo_teachers/audio
```

All commands show green `tqdm` progress bars. Outputs include participant IDs,
labels, class probabilities, predictions, confusion matrix, classification
report, checksums, source commits, and excluded IDs. TEST is rejected by the
split guard.
