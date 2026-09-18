# reliability-aware-depression-kd

## Teacher training

- `notebooks/02_train_frozen_teachers.ipynb` — Colab runner for frozen audio/text teachers.
- `notebooks/03_finetune_full_teachers.ipynb` — Colab runner for end-to-end teacher fine-tuning.
- `scripts/teachers/train_frozen_teachers.py` — frozen feature extraction and classifier training.
- `scripts/teachers/train_full_teachers_10epochs.py` — full encoder + classifier fine-tuning.

Python training code is kept under `scripts/`; notebooks only clone/pull `main`, install dependencies, and run the corresponding script.
