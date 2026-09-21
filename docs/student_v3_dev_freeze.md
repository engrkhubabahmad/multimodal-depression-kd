# v3 DEV freeze and final multimodal selection

The DEV stage is frozen after completing the controlled No-KD, Standard-KD and Reliability-Aware-KD comparison on DEV-34 with participant 440 excluded.

## Fixed protocol

- TRAIN: 107 participants
- DEV: 34 participants
- TEST: unopened
- threshold: 0.5, no threshold search
- fusion architecture: frozen branch embeddings, audio 256 -> 96, text 64 -> 96, interaction 384 -> 64 -> 1
- trainable fusion parameters: 55,617
- full student parameters: 254,274
- seed: 103
- Standard KD and RA-KD use the same initial fusion state
- temperature: 2.0
- KD weight: 0.5
- teacher logit scaling: per-modality median TRAIN absolute logit only

## DEV-34 results

| Condition | Accuracy | Macro-F1 | Depressed F1 | AUROC | Confusion matrix |
|---|---:|---:|---:|---:|---|
| No-KD | 0.7353 | 0.7043 | 0.6087 | 0.7470 | [[18,5],[4,7]] |
| Standard KD | 0.7353 | 0.7236 | 0.6667 | 0.7668 | [[16,7],[2,9]] |
| RA-KD | 0.7353 | 0.7236 | 0.6667 | 0.7549 | [[16,7],[2,9]] |

The predefined checkpoint/method ordering is lexicographic:
1. macro-F1
2. depressed-class F1
3. AUROC

Standard KD and RA-KD tie on macro-F1 and depressed F1. Standard KD is selected because its AUROC is higher.

## Reliability finding

RA-KD uses:
- c_m = 1 - exp(-|z_m| / s_m)
- w_m = c_m / (c_audio + c_text)
- q_RA = w_audio q_audio + w_text q_text

Mean TRAIN weights were approximately:
- audio: 0.5070
- text: 0.4930

The weighting therefore remained close to equal on average and did not improve the controlled Standard-KD result.

The saved TRAIN reliability audit further shows that confidence weighting is not a reliable proxy for which teacher is correct on teacher disagreements. This diagnostic must not be used to tune a new reliability rule because the audio TRAIN target is an in-sample run-4 crop reconstruction.

## Frozen decision

Selected final **multimodal** student condition: **Standard KD**.

The independently trained text-only v3 branch remains a stronger DEV comparator (macro-F1 0.7850) than any multimodal fusion condition and must be reported transparently.

No further architecture, reliability-rule, threshold, or hyperparameter tuning is permitted on DEV-34 after this freeze. TEST remains closed until the final evaluation step.
