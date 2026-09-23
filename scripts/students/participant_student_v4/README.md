# Consistent text export and compact fusion (experimental v4)

This experiment uses the frozen v3 text checkpoint and the existing v3 audio branch embeddings. It writes only inside the requested new v4 output directories. It does not retrain either branch or read TEST. The original v3 artifacts remain the baseline.

## Why this experiment

The v3 TRAIN text export uses training-graph document nodes while DEV uses inductive inference with the checkpoint's saved word state. Mixing those embeddings in downstream fusion changes the representation between splits. `export_text` applies the **DEV inductive path to both TRAIN and DEV**, keeping the same saved word state, vectorizer, weights and dropout-off evaluation. It checks the regenerated DEV embeddings, probabilities and binary decisions against the saved v3 DEV file. Any mismatch stops the run before an export is marked complete.

`compact_fusion` then compares the fixed text-only checkpoint with four class-balanced, L2-regularized logistic heads: text (65 parameters for 64-dimensional text), audio (audio dimension plus one), concatenated text/audio, and a text-logit plus audio correction. Exact parameter counts are saved in `metrics.csv`. The constant L2 coefficient is 1, the threshold is 0.5, and TRAIN alone fits scalers and heads. All presets finish fitting before any DEV metric is computed. This is an exploratory comparison, **not** a selected final model or a new reliability-aware KD method. Upstream branch checkpoints already used DEV for selection, so do not use this comparison to repeatedly tune choices on DEV or claim an unbiased score.

## In your existing Colab session

The Python modules run as subprocesses from the same notebook, leaving your already loaded objects and mounted Drive alone. First get this PR's commit in a separate checkout; substitute the commit SHA from the PR:

```python
from pathlib import Path
import subprocess
SOURCE = Path('/content/reliability-aware-depression-kd')
CODE = Path('/content/reliability-aware-depression-kd-v4')
COMMIT = '<PR commit SHA>'
assert (SOURCE / '.git').exists(), 'Adjust SOURCE to your existing clone'
if not CODE.exists():
    subprocess.run(['git', '-C', str(SOURCE), 'fetch', 'origin', COMMIT], check=True)
    subprocess.run(['git', '-C', str(SOURCE), 'worktree', 'add', '--detach', str(CODE), COMMIT], check=True)
assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
```

The paths below match the existing participant v3 experiment layout. If your directory names differ, update only the path assignments. Files must exist before running:

```python
import sys
DAIC = Path('/content/drive/MyDrive/DAIC_WOZ')
EXP = DAIC / 'experiments/students'
TEXT = EXP / 'participant_v3/branches/text_seed17'
AUDIO = EXP / 'participant_v3/branches/audio_seed1300'
V4 = EXP / 'participant_v4'
for p in (TEXT / 'inference_state.pt', TEXT / 'vectorizer.pkl',
          TEXT / 'train_text_embeddings.npz', TEXT / 'dev_text_embeddings.npz',
          AUDIO / 'train_audio_embeddings.npz', AUDIO / 'dev_audio_embeddings.npz'):
    assert p.is_file(), f'Missing input: {p}'
subprocess.run([sys.executable, '-m', 'scripts.students.participant_student_v4.export_text',
                '--daic-root', str(DAIC), '--text-branch', str(TEXT),
                '--output', str(V4 / 'consistent_text_v1')], cwd=CODE, check=True)
```

Only run fusion once text parity has passed:

```python
subprocess.run([sys.executable, '-m', 'scripts.students.participant_student_v4.compact_fusion',
                '--text-export', str(V4 / 'consistent_text_v1'),
                '--audio-branch', str(AUDIO), '--output', str(V4 / 'compact_fusion_v1')],
               cwd=CODE, check=True)
display(__import__('pandas').read_csv(V4 / 'compact_fusion_v1/metrics.csv'))
```

The run prints TRAIN and DEV confusion matrices and classification reports for every fixed preset and saves them as CSV, with participant predictions, frozen head/scaler NPZ files and an audit JSON. Its `complete.json` hashes every output and all inputs/code in the signature. Repeating an identical completed command verifies and reuses the output; changed inputs or incomplete directories raise an error and require a fresh output directory. Keep TEST closed until your method is frozen. These fits are participant-level, use 107 TRAIN and 34 DEV participants, exclude DEV participant 440, and do not use OOF.

The text exporter reads each participant transcript as needed and transforms documents with the saved 250-term vectorizer. Its feature matrix is 107 × 250 or 34 × 250; it does not load the audio cache into Colab RAM in the notebook. The separate training subprocess loads the two branch embedding arrays (107 and 34 rows) to fit the heads, so its process memory is released at completion.

## Verify locally

```bash
python -m unittest discover -s tests -v
```

The synthetic tests check the inductive graph algebra, cached artifact integrity, alignment, frozen DEV parity, fit independence from DEV labels and saved-head inference. They are not a substitute for running the parity gate with your actual v3 checkpoints in Colab.
