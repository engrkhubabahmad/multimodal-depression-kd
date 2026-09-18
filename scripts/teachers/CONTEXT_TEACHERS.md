# Cached contextual teacher experiment

Run `train_context_teachers.py` after the original segment teachers have finished.
It reads their saved TRAIN/DEV embeddings and original predictions, without
loading pretrained encoders or reading TEST audio, transcripts, or labels.
Only official TEST IDs are read to verify participant disjointness.

In the existing Colab notebook, after updating the repository:

```python
runner=runpy.run_path(str(ROOT/'scripts/teachers/train_context_teachers.py'))
selection=runner['main']([])
```

The new section 1b contains these lines. It is safe to call in a notebook with
kernel arguments present. Drive must already be mounted. No new packages are
required beyond the existing `requirements-colab.txt`.

## Candidates and selection

For each modality, compare:

1. Saved original teacher predictions (no retraining).
2. The original MLP architecture with participant/class-balanced sample weights.
3. A 64-dimensional projection, 64-unit GRU, and a target-segment branch using
   the same participant/class-balanced weights.

Each participant has equal total weight within a class; each class has equal
total weight. All normalizers are fitted on TRAIN only. Sample weights change
training BCE, while the logged `bce` metric and DEV BCE are unweighted. Checkpoint
selection uses predictive DEV BCE without the L2 penalty. Original heads were
selected by the old criterion and are historical controls, not exact reruns.

The GRU sees up to two preceding and two following retained segments. Windows
are chronological, right-padded and explicitly masked. They stop at participant
boundaries, split boundaries, missing turns/parts, and gaps over 30 seconds.
The target embedding is also provided explicitly. Mean context length is printed:
if most windows have length one, the existing 128-segment cap limits this approach.

This is an **offline contextual teacher** with privileged surrounding information.
Predictions and evaluation remain segment-level, with no participant aggregation.
No interviewer prompt embeddings are added. Teacher confidence is still uncalibrated;
improvement in DEV BCE does not guarantee improvement in F1 or downstream KD.

Each modality is selected by minimum DEV BCE among the three candidates; the
original wins ties. Threshold remains 0.5 for all comparison metrics. Report all
rows, including AUROC/AP, rather than claiming guaranteed improvement. DEV is a
selection set; these comparisons are not final held-out performance estimates.

## Artifacts and reuse

Candidates go under `segment_level_v1/seed_103/context_runs/<hash>/` (or the
configured seed). The hash covers arguments, cache files, original predictions,
manifest, implementation files, and TensorFlow version. Each completed candidate
has a checkpoint, TRAIN scaler, history, metrics, predictions, and artifact hashes.
Completed candidates are reused; an interrupted candidate restarts from its seed.
Original teacher files are never overwritten.

`active_teacher_selection.json` records the chosen prediction paths and hashes.
The updated KD target builder reads this selection and refuses changed predictions
or a changed segment manifest. **Rerun KD target construction after teacher selection.**
Existing KD targets and trained students are not automatically invalidated/retrained.
Student features need not be regenerated because segment IDs and embeddings are unchanged.

The legacy teacher's `protocol.json` must describe complete original artifacts.
Missing files fail with an error; do not delete existing caches. Cache validation
checks split hashes, metadata, IDs and cross-modal transcript signatures, but does
not rehash raw recordings. Regenerate caches if raw data were deliberately edited.

Run synthetic guard tests with `python -m unittest discover -s tests -v`.
The Keras training/save/reload smoke test runs when TensorFlow is installed.
