# Reliability-Aware Distillation for DAIC-WOZ

## Current status

The current DEV-selected multimodal experiment is **participant-level student v3, Standard KD, seed 103**. Its protocol and result are recorded in [the v3 DEV freeze](docs/student_v3_dev_freeze.md). TEST-47 has no result reported in this repository and must remain a one-time final evaluation after the complete method is frozen.

This result is exploratory. It does **not** establish that Reliability-Aware KD improves over Standard KD. The text-only student is a stronger DEV comparator, and the reliability-aware variant did not improve the controlled comparison.

## Fixed split

- TRAIN: 107 participants
- DEV: 34 participants; participant 440 excluded
- TEST: 47 participants, reserved for final evaluation

All splitting is participant-disjoint. Do not fit preprocessing, tune thresholds, select checkpoints, or revise methods using TEST.

## Locked teacher references

### Audio

Ravi et al. USSD, ComParE16 + LSTM-only, released run #4 (md_35_epochs.pth).

- Author commit: c3e68649153004ed2174878a1c54c716ad26cfd7
- Frozen DEV-34 macro-F1: 0.7875; accuracy: 0.8235
- Input: 130 ComParE16 channels × 384 frames
- The paper's five-run ensemble score is not the score of this single checkpoint.

Implementation and audit: [scripts/teachers/ussd_audio](scripts/teachers/ussd_audio).

### Text

Idiap participant-level InducT-GCN, top-250 features.

- Frozen DEV-34 macro-F1: 0.8419; accuracy: 0.8529; AUROC: 0.7826
- Participant 440 is excluded from DEV.
- TRAIN targets are in-sample graph-node predictions; see the target audit before interpreting their confidence as reliability.

Implementation and audit: [scripts/teachers/idiap_text](scripts/teachers/idiap_text).

## Student v3 DEV results

All rows use DEV-34 and threshold 0.5. Confusion matrices use rows=actual and columns=predicted, class order [0, 1].

| Model | Accuracy | Macro-F1 | Depressed F1 | AUROC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| Text-only branch | 0.7941 | 0.7850 | 0.7407 | 0.8063 | [[17, 6], [1, 10]] |
| Multimodal No-KD | 0.7353 | 0.7043 | 0.6087 | 0.7470 | [[18, 5], [4, 7]] |
| Multimodal Standard KD | 0.7353 | 0.7236 | 0.6667 | 0.7668 | [[16, 7], [2, 9]] |
| Multimodal Reliability-Aware KD | 0.7353 | 0.7236 | 0.6667 | 0.7549 | [[16, 7], [2, 9]] |

Standard KD is the frozen multimodal selection under the predeclared ordering (macro-F1, depressed F1, then AUROC). Reliability-Aware KD ties Standard KD on the first two metrics and has lower AUROC. Its mean TRAIN audio/text weights are 0.507/0.493, close to equal weighting. Report this as a null/negative result for the tested reliability rule. The stronger text-only result must remain visible in any report.

See [the full v3 design](docs/student_v3_design.md) and [DEV freeze record](docs/student_v3_dev_freeze.md).

## Current code versus legacy notebooks

The v3 implementation is in scripts/students/participant_student_v3/; teacher scripts are in scripts/teachers/. These scripts produced the v3 artifacts, but the four notebooks under notebooks/ still describe older v0/v2 workflows. In particular, notebooks/04_kd.ipynb is intentionally gated for v2. **The notebooks are not an end-to-end reproduction of the frozen v3 result.**

Use the v3 scripts and their README/protocol files as the implementation reference. A small v3 orchestration notebook still needs to be added and verified before claiming notebook-level reproducibility in Colab.

## Publication limitations to resolve

- The student comparison is one seed (103) on one small DEV set (34 participants, 11 depressed); report it as exploratory and do not select the best seed after seeing DEV.
- The teacher KD targets on TRAIN are in-sample. Confidence from these targets has not been shown to predict teacher correctness on unseen participants.
- Reliability-Aware KD did not outperform Standard KD in this controlled run. Do not claim a reliability-aware performance gain from these results.
- No held-out TEST score is reported here. Preserve the test protocol in [the final TEST procedure](docs/student_v3_final_test.md); if TEST labels or results have already been inspected, disclose that and do not use TEST for further model selection.

## Archived method notes

[RA_PDS_KD_METHOD.md](docs/RA_PDS_KD_METHOD.md) describes an earlier, superseded segment-level proposal. It is not the method behind the v3 results in this README.
