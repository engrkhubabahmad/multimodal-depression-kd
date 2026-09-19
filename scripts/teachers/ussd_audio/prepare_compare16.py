from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .common import compare16_matrix, extract_participant_wav, one, resolve_protocol_splits, save_json, sha256


def resolve_executable(value: str) -> str:
    hit = shutil.which(value) if "/" not in value else value
    if not hit or not Path(hit).exists(): raise FileNotFoundError(f"SMILExtract not found: {value}")
    return str(hit)


def process_split(root: Path, frame: pd.DataFrame, split: str, out: Path, smile: str, config: Path, keep_wav: bool):
    rows = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"USSD {split} ComParE16", colour="green"):
        pid, label = int(row.participant_id), int(row.label)
        raw_wav = one(root, f"{pid}_AUDIO.wav"); transcript = one(root, f"{pid}_TRANSCRIPT.csv")
        wav_path = out / "patient_wav" / split / f"{pid}_P_audio_data.wav"; csv_path = out / "compare16_csv" / split / f"{pid}_P_audio_data.csv"
        feat_path = out / "participant_features" / split / f"{pid}.npy"; csv_path.parent.mkdir(parents=True, exist_ok=True); feat_path.parent.mkdir(parents=True, exist_ok=True)
        audio_meta = extract_participant_wav(raw_wav, transcript, wav_path)
        subprocess.run([smile, "-C", str(config), "-I", str(wav_path), "-D", str(csv_path)], check=True)
        feat = compare16_matrix(csv_path); np.save(feat_path, feat)
        rows.append({"split": split, "participant_id": pid, "label": label, "feature_path": str(feat_path), "compare16_csv": str(csv_path), "frames": int(feat.shape[1]), **audio_meta, "raw_audio_sha256": sha256(raw_wav), "transcript_sha256": sha256(transcript), "patient_wav_sha256": sha256(wav_path), "compare16_csv_sha256": sha256(csv_path)})
        if not keep_wav: wav_path.unlink()
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--daic-root", required=True); p.add_argument("--output", required=True); p.add_argument("--split", choices=["dev", "train", "both"], default="dev")
    p.add_argument("--smile-extract", default="SMILExtract"); p.add_argument("--compare16-config", required=True); p.add_argument("--keep-wav", action="store_true")
    a = p.parse_args(argv); root, out = Path(a.daic_root), Path(a.output); config = Path(a.compare16_config)
    if not config.exists(): raise FileNotFoundError(config)
    smile = resolve_executable(a.smile_extract); train, dev = resolve_protocol_splits(root); rows = []
    if a.split in {"train", "both"}: rows += process_split(root, train, "train", out, smile, config, a.keep_wav)
    if a.split in {"dev", "both"}: rows += process_split(root, dev, "dev", out, smile, config, a.keep_wav)
    current = out / "participant_manifest.csv"
    if current.exists():
        old = pd.read_csv(current); new = pd.DataFrame(rows); merged = pd.concat([old[~old.set_index(["split","participant_id"]).index.isin(new.set_index(["split","participant_id"]).index)], new], ignore_index=True)
    else: merged = pd.DataFrame(rows)
    merged.sort_values(["split", "participant_id"]).to_csv(current, index=False)
    help_run = subprocess.run([smile, "-h"], capture_output=True, text=True)
    banner = (help_run.stdout or help_run.stderr).splitlines(); config_hash = sha256(config)
    summary = {"prepared": int(len(rows)), "splits": sorted(set(r["split"] for r in rows)), "manifest": str(current), "compare16_config": str(config), "compare16_config_sha256": config_hash, "smile_extract": smile, "smile_banner": banner[0] if banner else "unknown", "wav_bridge": "lossless PCM16 sample copy using author time indices", "label_source": "local AVEC split CSV; no author-side relabel during adaptation", "test_opened": False}
    for split in summary["splits"]:
        split_summary = {**summary, "splits": [split], "prepared": int(sum(r["split"] == split for r in rows))}
        save_json(out / f"preprocessing_{split}.json", split_summary)
    print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__": main()
