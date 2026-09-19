# RA-PDS-KD Methodology

**Reliability-Aware Participant-Disjoint Segment-Level Knowledge Distillation**

## Final protocol

```text
DAIC-WOZ
  -> PARTICIPANT SPLIT FIRST
       TRAIN 107
       DEV   34 usable
       TEST  47 CLOSED
  -> segment TRAIN/DEV independently
  -> aligned audio/text segment IDs
  -> segment-level audio teacher + text teacher
  -> DEV-only teacher evaluation
  -> TRAIN/DEV aligned teacher targets
  -> three student experiments
       1. No KD
       2. Standard KD
       3. Reliability-Aware Robust KD
  -> DEV evaluation for all three:
       clean
       audio missing
       text missing
       audio noisy
       text noisy
  -> checkpoint/threshold selection from CLEAN DEV only
  -> freeze final student
  -> open TEST
  -> same frozen segmentation + student feature pipeline
  -> CLEAN TEST only
  -> final segment-level metrics
```

No participant-level prediction aggregation is used.

## Teachers and cache

The active teacher script is `scripts/teachers/sbt_unimodal/train.py`: Wav2Vec2-base audio and ALBERT-large text teachers, independently adapted from SBT-Net. Published author scores are not reproduced claims. See [the audited teacher protocol](../scripts/teachers/sbt_unimodal/README.md).

TRAIN chunks may differ from aligned KD segments. Teacher checkpoint selection uses aligned DEV macro-F1 at threshold 0.5 with BCE tie-breaking. All text overflow windows are represented. S1 uses checksummed frozen-feature disk shards; S2 bypasses them and fine-tunes from raw local-disk input. Only batches enter RAM. Batch size is 16 for each teacher. Source/model/manifest identities control cache reuse; completed epochs are resumable.

Manifest/config artifacts remain under `experiments/features/rapdskd_segments_<hash>/`. Frozen feature shards persist under `experiments/teachers/sbt_frozen_features/`. Both teacher exports must finish before downstream KD starts.

The obsolete participant-level `teacher_v2_*` cache and its frozen/full teacher runners are not part of RA-PDS-KD.

## Student variants

### Student 1: No KD
Hard-label BCE only; clean TRAIN input.

### Student 2: Standard KD
Hard labels + equal-weight audio/text soft targets; clean TRAIN input.

### Student 3: Reliability-Aware Robust KD
Hard labels + reliability-weighted audio/text soft targets. TRAIN conditions:
- clean
- audio missing
- text missing
- audio noisy
- text noisy

Reliability is condition-aware:

`r_audio = audio_quality × audio_teacher_confidence`

`r_text = text_quality × text_teacher_confidence`

Missing quality = 0, clean quality = 1, noisy quality is reduced according to corruption severity.

## Controlled comparison and limits

All primary students now sample the same number of training examples per epoch. Student 3 no longer receives five times the update budget from its five conditions. Optional `--matched-corruption-ablation` adds a control with Student 3 input conditions and fusion but standard KD loss, isolating the effect of KD weighting. The three primary modes alone evaluate a combined intervention.

Entropy confidence is a heuristic rather than calibrated correctness. Reliability uses clean teacher probabilities weighted by the student's observed corruption/availability, not teacher predictions recomputed on noisy inputs. No OOF teacher fitting is introduced. Labels are participant labels inherited by segments; within-turn audio/text subdivisions are proportional rather than word-level forced alignment. Report participant-cluster uncertainty and acknowledge repeated DEV selection bias.

Final mode is explicitly chosen with `--final-mode` (default `ra_robust_kd`), not automatically the highest-scoring student. Keep this choice explicit before final TEST.

## DEV policy

All three students are evaluated on DEV under the same five conditions. Missing/noisy results are DEV robustness analyses only.

Checkpoint and decision-threshold selection use **clean DEV only**. The clean-selected threshold is reused for all DEV robustness conditions.

## TEST policy

TEST is inaccessible during teacher training/selection, KD construction, student training, robustness analysis, checkpoint selection, and threshold selection.

After the final student is frozen, TEST uses:
- the same segmentation algorithm
- the same TRAIN-built vocabulary
- the same TRAIN-fitted audio normalization
- the same frozen model
- the same fixed DEV threshold
- **clean input only**

No teacher is queried on TEST. No missing/noisy TEST experiments are run. No threshold search or refitting is allowed on TEST.

## Active notebook

Use only:

`notebooks/RA_PDS_KD_pipeline.ipynb`

