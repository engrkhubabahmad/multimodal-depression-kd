# Student v3: branch-pretrained compact multimodal student

## Why v3

Diagnostics on v2 showed that fusion was not the primary failure:

- text-only top-250 TF-IDF diagnostic: DEV-34 macro-F1 0.5522, AUROC 0.6087
- compact audio diagnostic: DEV-34 macro-F1 0.5467, AUROC 0.4506
- v2 multimodal peak observed during training: macro-F1 0.5983

Both unimodal student branches were substantially weaker than their locked teachers before fusion.

## v3 principle

Train compact unimodal branches first with hard labels only, using the mechanisms that made the teachers effective, then fuse their participant representations.

### Text branch

Independent InducT-style model:

- full Participant-only interview documents reconstructed from DAIC-WOZ
- TRAIN-only TF-IDF vocabulary
- TRAIN-only SelectKBest(f_classif), top 250
- word-word positive PMI graph, window 3
- word-document TF-IDF edges
- symmetric graph normalization
- 250 -> 64 -> 2, bias-free
- 16,128 trainable parameters
- published Idiap Optuna hyperparameters read from the public experiment DB
- weights trained from scratch
- frozen teacher checkpoint/vectorizer are not loaded

### Audio branch

Teacher-aligned but compressed acoustic model:

- existing participant-only ComParE16 matrices
- author TRAIN normalization artifact
- 384-frame segments
- Conv1D 130 -> 128
- BatchNorm + ReLU + MaxPool
- one-layer LSTM, hidden 128
- segment classifier
- 182,529 trainable parameters
- 16 randomly sampled segments per participant per epoch, deterministic by seed/epoch
- DEV uses all available segments and mean segment probability
- no teacher checkpoint loaded

### Planned fusion after branch validation

- audio participant embedding: mean + std of 128-D segment embeddings = 256-D
- text participant embedding: 64-D InducT representation
- projections: audio 256 -> 96, text 64 -> 96
- interaction fusion: [a, t, a*t, |a-t|]
- 384 -> 64 -> 1
- total planned student size: approximately 254,658 parameters
- approximately 22.1% of the 1,153,537-parameter audio teacher

No KD is introduced until the hard-label branches are validated independently.
TEST remains closed.


## Frozen-branch multimodal stage

After branch validation, v3 freezes the selected hard-label branch representations before comparing distillation objectives.

Selected branches:
- text: independent InducT-style top250 branch, 16,128 parameters
- audio: compressed USSD-style Conv128 + LSTM128 hard-label segment branch, 182,529 parameters
- the participant-bag audio experiment is retained as a negative pilot and is not used

Fusion:
- standardize each embedding dimension with TRAIN-107 mean/std only
- audio 256 -> 96
- text 64 -> 96
- concatenate [audio, text, audio*text, |audio-text|]
- 384 -> 64 -> 1
- fusion head: 55,617 trainable parameters
- full student including frozen branches: 254,274 parameters

For the controlled KD comparison, the branch embeddings, fusion architecture, participant split, threshold, and checkpoint-selection rule remain identical. No-KD trains first with hard labels only. Standard KD and RA-KD will reuse this exact backbone after the No-KD run is frozen.


## Standard KD (fixed controlled condition)

Standard KD reuses the frozen branch embeddings, TRAIN-only embedding standardizers, fusion architecture, optimizer hyperparameters, fixed threshold 0.5, and DEV-34 checkpoint rule from the No-KD condition.

Teacher-scale correction is fixed from TRAIN-107 only because raw audio/text teacher logits have very different scales:
- s_m = median_TRAIN(|z_m|)
- u_m = z_m / s_m
- q_m = sigmoid(u_m / T), T=2
- Standard KD target q = (q_audio + q_text)/2

Loss:
- hard = class-weighted participant BCE on TRAIN labels
- KD = T^2 * BCEWithLogits(student_logit/T, q)
- total = 0.5 * hard + 0.5 * KD

No DEV teacher targets enter training or teacher scaling. TEST remains closed. RA-KD must use these same normalized q_m targets so its only methodological change is reliability weighting.


## Reliability-Aware KD (fixed controlled condition)

RA-KD uses the exact Standard-KD backbone, initialization seed, TRAIN-only embedding standardizers, optimizer, temperature, KD weight, threshold, and checkpoint-selection rule. The only methodological change is participant-specific teacher reliability weighting.

For modality m:
- s_m = median_TRAIN(|z_m|), identical to Standard KD
- u_m = z_m / s_m
- q_m = sigmoid(u_m / T), identical to Standard KD
- c_m = 1 - exp(-|u_m|)
- w_m = c_m / (c_audio + c_text)
- q_RA = w_audio q_audio + w_text q_text

If both c values are numerically zero, weights fall back to 0.5/0.5.

Loss remains:
- hard = class-weighted participant BCE
- KD = T^2 * BCEWithLogits(student_logit/T, q_RA)
- total = 0.5 * hard + 0.5 * KD

No DEV-derived reliability, scale, threshold, or teacher target is used. TEST remains closed.
