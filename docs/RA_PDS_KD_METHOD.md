# RA-PDS-KD Methodology

**Reliability-Aware Participant-Disjoint Segment-Level Knowledge Distillation**

## 1. Participant-disjoint protocol

```text
DAIC-WOZ
  -> participant split FIRST
       TRAIN = 107
       DEV   = 34 usable participants (440 excluded: corrupted original files)
       TEST  = 47, CLOSED until the final student is frozen
  -> segment each split independently
  -> aligned audio/text segment IDs
```

No participant can contribute segments to more than one split. Teachers and students are evaluated at **segment level only**. There is no participant-level prediction aggregation.

## 2. Teacher stage

Each aligned segment contains participant audio (<=10 s) and the corresponding transcript chunk (<=254 MiniLM tokens).

- Audio: modality-specific cache -> Wav2Vec2 embedding -> audio segment classifier.
- Text: modality-specific cache -> MiniLM embedding -> text segment classifier.
- Teacher checkpoint/model selection uses DEV segments only.
- TRAIN/DEV teacher probabilities are exported by the same `segment_id`.
- TEST is never queried by a teacher.

## 3. Student feature stage

The lightweight ReLiMP-Net student does **not** use teacher embeddings as input.

- Audio input: 64-bin log-Mel summary (mean + std = 128-D) per aligned segment.
- Text input: train-only vocabulary and fixed-length token IDs.
- Audio normalization statistics are fitted on clean TRAIN segments only.
- Text vocabulary is built from TRAIN segment text only.
- TRAIN/DEV noisy variants are generated deterministically from the same aligned segment.
- TEST features are not generated at this stage.

Default robustness settings are configurable in code. Current defaults are 10 dB additive audio noise and 30% mixed text corruption (mask/delete/replace).

## 4. Three student experiments

### Student 1: No KD

- hard-label BCE only;
- clean TRAIN input only;
- no teacher target.

### Student 2: Standard KD

- hard labels + standard dual-teacher KD;
- clean TRAIN input only;
- standard teacher target = 0.5 audio probability + 0.5 text probability;
- no reliability weighting.

### Student 3: Reliability-Aware Robust KD

TRAIN conditions:

1. clean;
2. audio missing;
3. text missing;
4. audio noisy;
5. text noisy.

Reliability is condition-aware:

`r_audio = audio_quality × audio_teacher_confidence`

`r_text = text_quality × text_teacher_confidence`

Teacher confidence is `1 - normalized binary entropy`. Input quality reflects the actual synthetic condition: clean = 1, missing = 0, noisy = a reduced value determined by corruption severity. The normalized reliabilities weight the audio/text teacher probabilities in the KD target.

Student 3 also uses availability/quality weights for feature fusion, so a missing modality contributes approximately zero and a degraded modality contributes less. Teacher outputs are not needed at inference.

## 5. DEV-only robustness evaluation

All three frozen candidate students are evaluated on DEV under the same five conditions:

- clean;
- audio missing;
- text missing;
- audio noisy;
- text noisy.

Checkpoint and decision-threshold selection use **clean DEV only**. Robustness conditions are reported as DEV robustness analyses using that same clean-selected threshold. No condition-specific threshold tuning is allowed.

## 6. Final TEST policy

After the final student mode/checkpoint/threshold is accepted, everything is frozen. Only then is TEST opened.

TEST uses:

- the same segmentation algorithm;
- the same train-built vocabulary;
- the same train-fitted audio normalization;
- the same frozen model checkpoint;
- the same fixed DEV threshold;
- **clean input only**.

No missing/noisy TEST experiments are performed. No teacher is queried on TEST. No TEST threshold search, normalization fitting, vocabulary fitting, checkpoint selection, or hyperparameter tuning is allowed.

If the official 47-participant TEST metadata does not contain labels, final metrics cannot be computed until a legitimate TEST-label file is supplied. Such labels must not be used before the final student is frozen.

## 7. Final metrics

Final clean TEST reporting is segment-level only:

- Accuracy
- Precision
- Recall
- F1
- Macro-F1
- Depressed-F1
- Balanced Accuracy
- AUROC
- Average Precision
- Confusion Matrix
- Classification Report
