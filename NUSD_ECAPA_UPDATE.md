# NUSD ECAPA-TDNN replacement

The canonical TRAIN/DEV notebook replaces the Step-Audio2 sections with frozen inference from the public NUSD raw ECAPA-TDNN release.

## Files

- `notebooks/01_original_teachers_canonical_train_dev.ipynb`: author preprocessing, NUSD checkpoint retrieval, GPU DEV evaluation, and prediction/CM/CR export.
- `scripts/experiments/internal188/nusd_ecapa.py`: safe TRAIN/DEV metadata generation, preprocessing allowlist patch, NUSD config, compatibility shims, and prediction aggregation.
- `notebooks/01_original_teachers_internal188.ipynb`: removes the outdated Step-Audio2 cells and points to the canonical notebook.

The notebook pins `kingformatty/NUSD` at `4cfbdfa9c2fdfcf476c28a051075d57ed0ca9fdc` and `adbailey1/daic_woz_process` at `8b5f8ff5c510df9904e0750f1a806c29e649c5cf`. Audio discovery is patched to accept only the frozen 107 TRAIN and 34 DEV IDs and never extracts archives. The published model scores DEV only; TEST remains unopened.

Output is written under `DAIC_WOZ/experiments/idiap_ussd_canonical188_seed103/teachers/nusd_ecapa_frozen`. Source repositories are placed under `DAIC_WOZ/tools/solo_teacher_sources`.

This is a frozen-checkpoint evaluation, not NUSD fine-tuning. Because the published checkpoint may have used DAIC-WOZ DEV during model selection, the DEV metrics are exploratory. Train participant probabilities are not exported in this first evaluation pass.

## Remove these Step-Audio2 files from the repository

- `scripts/experiments/internal188/evaluate_stepaudio2_official.py`
- `scripts/experiments/internal188/evaluate_stepaudio2_val.py`
- `scripts/experiments/internal188/prepare_stepaudio2_official_eval.py`
- `scripts/experiments/internal188/prepare_stepaudio2_sft.py`
- `scripts/experiments/internal188/prepare_stepaudio2_val.py`
- `scripts/experiments/internal188/recover_stepaudio2_val.py`
- `scripts/experiments/internal188/stepaudio2_adapter_audit.py`
- `scripts/experiments/internal188/stepaudio2_t4_fp16_plugin.py`
