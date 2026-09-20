# reliability-aware-depression-kd

## Active protocol

Strict participant-disjoint DAIC-WOZ protocol:

- TRAIN: 107 participants
- DEV: 34 participants
- participant 440 excluded from DEV
- TEST: 47 participants, closed until the final student is frozen

## Locked teachers

### Audio teacher

Ravi et al. USSD, ComParE16 + LSTM-only, released run #4.

- author commit: `c3e68649153004ed2174878a1c54c716ad26cfd7`
- checkpoint: `md_35_epochs.pth`
- frozen DEV-34 macro-F1: 0.7875
- exact active-model parameters: 1,153,537
- input per segment: 130 ComParE16 channels × 384 frames
- TRAIN KD targets: author-style run-4 seed-1300, 6662-frame crop
- no fine-tuning in the active path

Code: `scripts/teachers/ussd_audio/`

### Text teacher

Idiap participant-level InducT-GCN top-250.

- published checkpoint + vectorizer
- frozen DEV-34 macro-F1: 0.8418604651
- top-250 TF-IDF participant documents
- 250 → 64 graph projection → 2-class output
- exact trainable weights from the published architecture: 16,128
- TRAIN KD targets: checkpoint training-graph document nodes
- DEV targets: saved `A_dev`, participant 440 excluded
- TEST never opened

Code: `scripts/teachers/idiap_text/`

## Active student candidate: rich compact v2

The previous 670k `ParticipantReLiMPNet` mean/std-log-Mel student is retained as **v0 experimental history**. Its DEV-34 no-KD macro-F1 was 0.6092 and its equal Standard-KD run was 0.5729.

The active candidate is `RichParticipantStudent-v2`:

- reuses existing patient-only USSD-compatible ComParE16 **input features**, not teacher predictions or hidden states
- independently refits TRAIN-only acoustic normalization
- independently fits TRAIN-only top-250 TF-IDF from Participant transcripts
- temporal Conv1D + depthwise temporal Conv + small GRU audio encoder
- compact text MLP
- text-conditioned attention over acoustic segments
- participant-level multimodal classifier
- expected trainable size: about 227k parameters, much smaller than the audio teacher
- threshold fixed at 0.5
- TEST not prepared or opened

Code: `scripts/students/participant_student_v2/`

Design audit: `docs/student_v2_design.md`

## Colab notebooks

Run in order:

1. `notebooks/01_text_teacher.ipynb`
2. `notebooks/02_audio_teacher.ipynb`
3. `notebooks/03_student_baseline.ipynb` — active v2 rich No-KD student
4. `notebooks/04_kd.ipynb` — intentionally gated until the v2 No-KD result is frozen

Do not mix the old v0 Standard-KD checkpoint/results with the v2 student.

TEST must remain closed.
