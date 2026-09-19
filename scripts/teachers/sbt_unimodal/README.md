# SBT-inspired segment teachers: audited scope and Colab use

These are independent Wav2Vec2-base audio and ALBERT-large classifiers for
RA-PDS-KD. They are adaptations, not released author unimodal checkpoints.
Published accuracy/F1 is not guaranteed on our fixed participant split.

## Author audit (19 September 2026)

[Paper](https://link.springer.com/article/10.1186/s13040-025-00498-x):
subject-disjoint train/test folds are explicitly stated. Mean ten-fold F1 is
83.17%, separately from the audio 0.91 / text 0.87 comparison figures. These
are not fixed-split segment metrics. The described full-model schedule freezes
encoders for 25 epochs then unfreezes upper layers, with batch 16, up to 85
epochs, and validation macro-F1 selection. That schedule trains substantial
fusion modules absent from our teachers. ALBERT shares parameters, so a physical
group cannot be treated as two independently trainable logical layers.

[Author snapshot 75bb846](https://github.com/ghy-yhg/SBT-Net/tree/75bb846625683168da6bbe85dafd000211ecdf12):
the demo instead uses random 80/20 row splitting, batch 2, 20 epochs, AdamW and
AUC selection from binary predictions. It references undefined `loader1`.
Its model loads ALBERT-base and imports two absent modules. Its loader keeps
only the first 15 seconds and truncates text to 128 tokens. Exact unimodal
heads/checkpoints and fold membership are absent in this snapshot. We do not
copy its evaluation errors or claim a reproduction of its scores.

## Declared protocol

- Split first: TRAIN 107, DEV 34 (440 excluded), TEST 47 reserved. Read TEST
  IDs only. No teacher TEST inference, fitting, calibration or selection.
- TRAIN uses participant-only audio chunks up to 15 seconds and text chunks
  up to 128 content tokens; class/participant-balanced replacement sampling.
- KD retains aligned segments up to 10 seconds / 254 segmentation tokens and
  their original IDs. Labels are inherited participant labels. Within-turn
  subdivision is proportional, not word-level forced alignment.
- Select teachers on aligned DEV segment macro-F1 at threshold 0.5; predictive
  BCE breaks ties. Also report depressed-F1, AUROC, AP, Brier and BCE.
- Text overflow windows retain all content within 128-token encoder inputs.
  Mean window CLS representations yield one vector/prediction per segment.
- Audio uses fixed 15-second padding and masked pooling throughout. Wav2Vec2
  base group normalization can otherwise make features depend on batch peers.
  Fixed padding may cost more computation but makes cache semantics consistent.
  This is an adaptation, not the paper's median-based normalization.
- S1: frozen/eval encoder, train head with dropout; at most 25 epochs, patience 5.
  S2: restore best S1 head, unfreeze last two audio blocks or the shared text
  group; at most 60 epochs, patience 10. Frozen blocks stay in eval mode.
  Best aligned-DEV checkpoint across both stages wins, even if S2 fails to improve.
- AdamW, cosine warmup schedule; head LR 2e-5, encoder LR 1e-5. Audio/text
  physical batch 16, accumulation 1. These are declared adaptation settings.

## Disk cache and recovery

No dataset-sized embedding array is retained in Python/Colab RAM. Batch buffers,
model activations and operating-system file-page caching still use memory.

1. Hash TRAIN/DEV sources and stage WAV files onto local Colab disk. Hashing and
   copying use bounded buffers. Source files are never modified.
2. S1 writes small frozen-feature NPY shards to Drive with green progress bars.
   Atomic writes/checksums support interruption and corrupted-shard recovery.
   Valid shards are reused without decoding audio again. Local copies are
   memory-mapped; only requested rows enter each batch.
3. Keys include model revisions, source bytes, ordered manifests/labels/text,
   preprocessing, implementation hashes, precision and library versions.
   Models load the resolved revision rather than an unpinned model alias.
4. S2 bypasses frozen features and loads raw local-disk batches with two workers.
   DEV inference also uses mixed precision. Actual GPU name is printed.
5. S1 checkpoints contain only the small head and training state; S2 includes
   model/optimizer/scheduler/scaler. Atomic `last.pt` resumes after the last
   completed epoch; an interrupted epoch replays. Epoch seeds restore sampling
   and dropout. CPU resume is tested; cross-hardware GPU bitwise identity is not promised.
6. Runs use content-specific directories. Old v1 artifacts remain intact and
   cannot resume as v2 because input/selection semantics changed. Only superseded
   generated best files inside the current run are cleaned up.
7. Aligned exports have checksums and are reused when valid. Exports are marked
   ready only after both modalities finish. KD/student stages reject inconsistent
   provenance. Rebuild downstream targets/features after changes.

Initial extraction and S2 still require encoder computation. Ensure local disk
space for selected WAVs and checkpoints. Drive artifacts survive runtime reset;
temporary local WAV copies are reconstructed. Do not run concurrent writers to
the same experiment.

## Existing Colab runtime

After safely updating this branch, use a fresh subprocess to avoid stale imports:

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'scripts.teachers.sbt_unimodal.train',
                '--audio-batch', '16', '--text-batch', '16'], cwd=ROOT, check=True)
```

Rerun the same command after interruption. `--force-retrain` starts a new attempt
without deleting earlier checkpoints; frozen shards remain reusable. Changing
scientific settings creates a distinct run. A CUDA OOM needs an explicit batch/
memory decision; the code does not silently lower the requested batch size.

## Student methodology

The three primary modes remain. Each now samples the same number of examples
per epoch; five corruption conditions no longer give Student 3 five times the
training budget. Participant class balance uses participant counts.

Optional student flag `--matched-corruption-ablation` adds
`ra_input_standard_kd`: same corrupted inputs and reliability input fusion as
Student 3, with standard KD loss. This isolates KD weighting. The three primary
modes alone measure a combined intervention. Report actual epochs and updates.

Entropy confidence is a heuristic, not calibrated correctness. Evaluate it with
Brier/BCE and reliability diagnostics before claiming calibration. Repeated DEV
tuning causes selection bias. Use participant-cluster uncertainty for segment
metrics. TEST remains a one-time, clean-only frozen-student evaluation.

Final student mode is explicitly selected via `--final-mode` (default
`ra_robust_kd`); it is not automatically the highest DEV score. Record that choice.

## Validation

`python -m pytest tests/test_sbt_disk.py -q`

Synthetic CPU tests cover cache equivalence/reuse/corruption, identity changes,
atomic writes, text coverage, tiny real HF encoder pooling/freezing and resume.
Actual DAIC performance, Colab GPU memory and throughput require a real run.
