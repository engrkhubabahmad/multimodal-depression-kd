# Reproduction results

Each successful exact-v3 reproduction run is published by `notebooks/06_verify_reproduction.ipynb` into its own immutable subfolder:

`results/<run_id>/`

A result folder contains compact artifacts needed to audit DEV-34 results without Google Drive: predictions, comparative metrics, confusion matrices, classification reports, source metric JSON files, and provenance.

Large model checkpoints, raw/extracted features, and training caches stay on Google Drive and are not committed here.

Result folders are never overwritten by the notebook.
