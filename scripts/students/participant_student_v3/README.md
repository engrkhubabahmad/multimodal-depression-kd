# Participant student v3: branch-pretrained compact student

v2 diagnostics showed that both compact branches were weak before fusion:
- TF-IDF text diagnostic: DEV-34 macro-F1 0.5522, AUROC 0.6087
- compact audio diagnostic: DEV-34 macro-F1 0.5467, AUROC 0.4506

v3 therefore changes the training mechanism rather than increasing size.

## Text branch
- Participant-only full interview documents reconstructed from DAIC-WOZ.
- TRAIN-only TF-IDF and supervised top-250 selection.
- Exact InducT-style word/document graph mechanism: positive PMI word-word edges, TF-IDF word-document edges, symmetric graph normalization.
- 250 -> 64 -> 2, 16,128 trainable parameters.
- Independently trained weights; no teacher checkpoint/vectorizer.
- Published Idiap Optuna hyperparameters are read from the public experiment database, avoiding a new hyperparameter search on DEV-34.

## Audio branch
- Existing USSD-compatible ComParE16 arrays.
- Author TRAIN normalization artifact, but no teacher checkpoint.
- 130 -> 128 Conv1D + BN/ReLU/MaxPool + one-layer LSTM128 + segment head.
- 182,529 trainable parameters.
- Fixed 6,662-frame author-style TRAIN crop with run-4 seed 1300, segmented into 384-frame windows.
- Author run-4 class balancing is reproduced as 468 TRAIN segments per class; no class weights.
- Adam, batch 20, initial LR 0.003, weight decay 0, LR ×0.9 every 2 epochs.
- DEV uses all segments; author-style majority vote is the primary checkpoint metric, with soft mean probability retained for AUROC/fusion.

After these two branches are independently validated, their 64-D text and 256-D audio embeddings will feed the same compact multimodal fusion head for No-KD, Standard KD and RA-KD.


## Audio participant-bag branch (active after segment pilot)

The author-recipe compressed segment-level pilot memorized TRAIN segments and peaked at DEV-34 majority-vote macro-F1 0.6222 (AUROC 0.5257), so it is retained as a negative pilot rather than frozen.

The active hard-label audio branch uses the same verified ComParE16 arrays, author TRAIN normalization, 384-frame segmentation and 182,529-parameter compressed encoder, but supervision is applied at the participant level:
- sample 16 TRAIN segments per participant per epoch
- encode segments with the compact USSD-style encoder
- participant probability = mean sigmoid(segment logits)
- BCE is applied once per participant, with TRAIN-only pos_weight=77/30
- DEV uses all segments and mean probability
- no teacher checkpoint or teacher logits
- no threshold search; TEST closed
