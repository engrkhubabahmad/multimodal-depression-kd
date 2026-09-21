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
