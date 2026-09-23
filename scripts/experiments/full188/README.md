# Fresh 188-person seed-42 protocol

This is a **new internal DAIC-WOZ protocol**. The labeled `full_test_split.csv` is verified against the official 47-person TEST roster, then TRAIN-107 + usable DEV-34 + labeled TEST-47 are redistributed into participant-disjoint **TRAIN-132 / VAL-28 / student TEST-28**. Participant 440 is excluded. Since the official TEST roster contributes to every group, do **not** compare the resulting test score to papers using the official AVEC test partition.

The previous `participant_v4` linear proxy outputs and the original canonical teacher checkpoints cannot be reused on this split. The teacher training here starts from random weights using the author InducT-GCN and USSD ComParE16+LSTM architectures. The compact student text and audio branches are independently initialized. There is **no OOF**. TRAIN teacher logits and probabilities are generated in sample and checked against participant IDs and labels before KD. All learned text vocabulary, audio normalization and fusion scalers use only TRAIN-132. The student TEST-28 is scored only by the separate final scorer after one mode is fixed on VAL-28.

Run these modules from a clean Git worktree in your **existing Colab session**. The GitHub PR commit is pinned in the PR; your existing `GITHUB_TOKEN` Colab Secret can authenticate the `git fetch` command as before. Keep code outside Drive and artifacts under `/content/drive/MyDrive/DAIC_WOZ/experiments/students/full188_seed42/`.

After your existing Colab setup cell defines `ROOT` and authenticated `git`, fetch this PR branch without changing `main`:

```python
from pathlib import Path
import subprocess
BRANCH = 'feature/full188-fresh-teachers-student-seed42'
CODE = Path('/content/reliability-aware-depression-kd-full188')
subprocess.run(git + ['-C', str(ROOT), 'fetch', 'origin', BRANCH], check=True)
revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'FETCH_HEAD'], text=True).strip()
if CODE.exists():
    assert (CODE / '.git').exists()
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
    subprocess.run(['git', '-C', str(CODE), 'switch', '--detach', revision], check=True)
else:
    subprocess.run(['git', '-C', str(ROOT), 'worktree', 'add', '--detach', str(CODE), revision], check=True)
print('Full-188 code:', revision)
```

```python
from pathlib import Path
import subprocess, sys, json, pandas as pd
DAIC = Path('/content/drive/MyDrive/DAIC_WOZ')
CODE = Path('/content/reliability-aware-depression-kd-full188')
EXP = DAIC / 'experiments/students/full188_seed42'
CANONICAL_FEATURES = Path('REPLACE_WITH_YOUR_EXISTING_COMPARE16_CACHE')
IDIAP_SOURCE = Path('REPLACE_WITH_YOUR_LOCAL_IDIAP_AUTHOR_CLONE')
PREP = json.loads((CANONICAL_FEATURES / 'preprocessing_train.json').read_text())
CONFIG = Path(PREP['compare16_config'])
assert (DAIC / 'metadata/full_test_split.csv').is_file()
assert (CANONICAL_FEATURES / 'participant_manifest.csv').is_file()
assert (IDIAP_SOURCE / 'main.py').is_file()
assert CONFIG.is_file()
```

First freeze the labeled split, then reuse 141 feature files and extract 47 previously missing feature files. Feature preparation uses one participant at a time and displays a green progress bar. The original audio cache stays where it is.

```python
def stage(module, *arguments):
    subprocess.run([sys.executable, '-m', f'scripts.experiments.full188.{module}',
                    *map(str, arguments)], cwd=CODE, check=True)

SPLIT = EXP / 'split'
FEATURES = EXP / 'features'
stage('split', '--daic-root', DAIC, '--output', SPLIT, '--seed', 42)
stage('features', '--split-dir', SPLIT, '--daic-root', DAIC,
      '--canonical-features', CANONICAL_FEATURES, '--output', FEATURES,
      '--compare16-config', CONFIG)
```

Train the original teacher architectures from fresh weights and export their **TRAIN-132** targets. Audio training reads source arrays with memory mapping and batches 16 segments; the train-only normalization is saved. GPU is used if available.

```python
TEXT_TEACHER = EXP / 'teachers/text'
AUDIO_TEACHER = EXP / 'teachers/audio'
stage('train_text_teacher', '--daic-root', DAIC, '--split-dir', SPLIT,
      '--idiap-source', IDIAP_SOURCE, '--output', TEXT_TEACHER)
stage('train_audio_teacher', '--split-dir', SPLIT, '--features', FEATURES,
      '--output', AUDIO_TEACHER, '--batch-size', 16)
display(pd.read_csv(TEXT_TEACHER / 'train_text_targets.csv').head())
display(pd.read_csv(AUDIO_TEACHER / 'train_audio_targets.csv').head())
```

Train the compact student branches from independent weights, then fit all three fusion modes using the same split, optimizer and initial seed. VAL-28 missing-audio/text and Gaussian-noise comparisons are saved separately.

```python
BRANCHES = EXP / 'students/branches'
FUSION = EXP / 'students/fusion'
stage('train_student_branches', '--daic-root', DAIC, '--split-dir', SPLIT,
      '--features', FEATURES, '--audio-teacher', AUDIO_TEACHER, '--output', BRANCHES)
stage('train_student_fusion', '--split-dir', SPLIT, '--student-branches', BRANCHES,
      '--text-teacher', TEXT_TEACHER, '--audio-teacher', AUDIO_TEACHER,
      '--output', FUSION)
print((FUSION / 'audit.json').read_text())
display(pd.read_csv(FUSION / 'val_missing_noise.csv')
        .groupby(['mode', 'scenario', 'noise_sd'])[['macro_f1', 'depressed_f1', 'auroc']]
        .agg(['mean', 'std']).round(3))
```

The 28-person internal student test should be opened **only once after the method and mode are frozen**. This final command runs only the student branches and chosen fusion head. It never runs either teacher on student TEST; a selection lock prevents choosing a different mode afterward. Do not run it during development.

```python
CHOSEN_MODE = 'ra_kd'  # Set once after validation review
stage('score_student_test', '--daic-root', DAIC, '--split-dir', SPLIT,
      '--features', FEATURES, '--student-branches', BRANCHES,
      '--audio-teacher', AUDIO_TEACHER, '--student-fusion', FUSION,
      '--mode', CHOSEN_MODE, '--output', EXP / 'students/internal_test')
```

The code has local synthetic split and provenance tests. Torch/author-model execution and actual DAIC-WOZ results require the user's Colab runtime; do not report scores until those stages complete.
