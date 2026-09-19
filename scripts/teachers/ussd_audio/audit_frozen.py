from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import AUTHOR_COMMIT, AUTHOR_DEV35_MACRO_F1
from .common import aggregate_segments, infer_segments, load_author_model, load_author_stats, load_feature_manifest, metric_dict, normalise_segments, save_json, segment_feature, sha256


def evaluate(model, frame, mean, std, device, batch_size=64, save_segments=False):
    participants, segment_tables = [], []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Frozen USSD DEV-34 audit", colour="green"):
        feat = np.load(row.feature_path).astype(np.float32); seg = normalise_segments(segment_feature(feat), mean, std)
        pred = infer_segments(model, seg, device, batch_size); pred.insert(0, "participant_id", int(row.participant_id)); pred.insert(1, "label", int(row.label))
        participants.append(aggregate_segments(pred, row.participant_id, row.label))
        if save_segments: segment_tables.append(pred)
    return pd.DataFrame(participants), pd.concat(segment_tables, ignore_index=True) if segment_tables else None


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--features", required=True); p.add_argument("--author-root", default="/content/solo_teacher_sources/USSD-depression"); p.add_argument("--output", required=True); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--save-segments", action="store_true")
    a = p.parse_args(argv); features, author, out = Path(a.features), Path(a.author_root), Path(a.output); out.mkdir(parents=True, exist_ok=True)
    prep_path = features / "preprocessing_manifest.json"
    if not prep_path.exists(): raise FileNotFoundError(f"Missing preprocessing provenance: {prep_path}")
    prep = json.loads(prep_path.read_text(encoding="utf-8"))
    if prep.get("test_opened") is not False or "dev" not in prep.get("splits", []): raise AssertionError("Preprocessing manifest does not certify a DEV-only/test-closed run")
    dev = load_feature_manifest(features, "dev"); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, state, ckpt = load_author_model(author, device); mean, std, stats_path = load_author_stats(author)
    participants, segments = evaluate(model, dev, mean, std, device, a.batch_size, a.save_segments); metrics = metric_dict(participants)
    participants.to_csv(out / "dev_participant_predictions.csv", index=False)
    if segments is not None: segments.to_csv(out / "dev_segment_predictions.csv", index=False)
    meta = {"protocol": "frozen author checkpoint; no fitting; DEV-34 only", "author_commit": AUTHOR_COMMIT, "author_run": 4, "checkpoint": ckpt.name, "checkpoint_sha256": sha256(ckpt), "checkpoint_epoch": int(state.get("epoch", -1)) if isinstance(state, dict) else None, "normalization_artifact": stats_path.name, "normalization_sha256": sha256(stats_path), "preprocessing_manifest": str(prep_path), "compare16_config_sha256": prep.get("compare16_config_sha256"), "smile_banner": prep.get("smile_banner"), "input_shape": ["N", 130, 384], "native_output": "sigmoid depression probability", "exposed_output": "pre-sigmoid logit + sigmoid probability", "author_aggregation": "np.rint per segment -> participant vote fraction -> np.rint", "kd_soft_aggregation": "mean segment probability; participant logit=logit(mean probability)", "author_canonical_dev35_macro_f1_reference_only": AUTHOR_DEV35_MACRO_F1, "local_dev34": metrics, "participant_440_excluded": True, "test_opened": False}
    save_json(out / "audit.json", meta); print(participants.to_string(index=False)); print(json.dumps(meta, indent=2)); return meta


if __name__ == "__main__": main()
