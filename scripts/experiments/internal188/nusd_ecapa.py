"""Safe DAIC-WOZ setup and evaluation helpers for the published NUSD raw ECAPA-TDNN."""
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, average_precision_score,
                             classification_report, brier_score_loss, log_loss)

NUSD_REPO = "https://github.com/kingformatty/NUSD.git"
PREPROCESS_REPO = "https://github.com/adbailey1/daic_woz_process.git"
NUSD_COMMIT = "4cfbdfa9c2fdfcf476c28a051075d57ed0ca9fdc"
PREPROCESS_COMMIT = "8b5f8ff5c510df9904e0750f1a806c29e649c5cf"
WEIGHTS_URL = "https://drive.google.com/drive/folders/1F_1R8MdN-jYx4JQcRo8zzd-jB3_wJ5fX"


def _metadata(path):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    by_lower = {c.lower(): c for c in frame.columns}
    def col(*names):
        for name in names:
            if name.lower() in by_lower:
                return by_lower[name.lower()]
        raise ValueError(f"Missing metadata column {names}; found {list(frame.columns)}")
    pid, label = col("Participant_ID", "participant_id"), col("PHQ8_Binary", "phq8_binary")
    score = col("PHQ8_Score", "phq8_score")
    gender = col("Gender", "gender")
    return frame.rename(columns={pid:"Participant_ID", label:"PHQ8_Binary",
                                 score:"PHQ8_Score", gender:"Gender"})


def prepare_metadata(daic_root, split_dir, output):
    """Create only canonical TRAIN/DEV author-format CSVs; TEST stays empty."""
    root, split_dir, output = Path(daic_root), Path(split_dir), Path(output)
    manifest = pd.read_csv(split_dir / "manifest.csv")
    tr_ids = set(manifest.loc[manifest.split.eq("train"), "participant_id"].astype(int))
    dv_ids = set(manifest.loc[manifest.split.eq("val"), "participant_id"].astype(int))
    if len(tr_ids) != 107 or len(dv_ids) != 34 or 440 in dv_ids:
        raise ValueError(f"Expected canonical TRAIN-107 / DEV-34 excluding 440; got {len(tr_ids)}/{len(dv_ids)}")
    train = _metadata(root / "metadata" / "train_split_Depression_AVEC2017.csv")
    dev = _metadata(root / "metadata" / "dev_split_Depression_AVEC2017.csv")
    train = train[train.Participant_ID.astype(int).isin(tr_ids)].sort_values("Participant_ID")
    dev = dev[dev.Participant_ID.astype(int).isin(dv_ids)].sort_values("Participant_ID")
    if set(train.Participant_ID.astype(int)) != tr_ids or set(dev.Participant_ID.astype(int)) != dv_ids:
        raise ValueError("Canonical split IDs do not match metadata CSVs")
    allowed = sorted(tr_ids | dv_ids)
    indexed = {pid: {} for pid in allowed}
    # Directory listing only: never read reserved TEST audio or transcripts.
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in {"experiments", "tools", ".git"}]
        parent = Path(directory)
        for filename in filenames:
            match = re.fullmatch(r"(\d{3})_(AUDIO\.wav|TRANSCRIPT\.csv)", filename, re.IGNORECASE)
            if not match:
                continue
            pid = int(match.group(1))
            if pid not in indexed:
                continue
            kind = "audio_path" if match.group(2).lower().endswith(".wav") else "transcript_path"
            if kind in indexed[pid]:
                raise ValueError(f"Multiple {kind} files for TRAIN/DEV participant {pid}")
            indexed[pid][kind] = str(parent / filename)
    for pid, files in indexed.items():
        if set(files) != {"audio_path", "transcript_path"}:
            raise FileNotFoundError(f"Missing extracted TRAIN/DEV WAV or transcript for {pid}: {files}")
    out = output / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    dev.to_csv(out / "dev.csv", index=False)
    pd.concat([train, dev]).sort_values("Participant_ID").to_csv(out / "train_dev.csv", index=False)
    # The author code reads this while initializing the validation loader.
    # Empty rows ensure TEST media and labels are not touched.
    train.iloc[:0].to_csv(out / "test_empty.csv", index=False)
    (out / "allowed_audio_ids.txt").write_text("\n".join(map(str, allowed)) + "\n")
    pd.DataFrame([{"participant_id": pid, **indexed[pid]} for pid in allowed]).to_csv(out / "audio_manifest.csv", index=False)
    return {"metadata": out, "allowed_ids": allowed, "train_n": len(train), "dev_n": len(dev)}


def _restrict_audio_scanner(source):
    """Patch helper to scan only allowlisted participant folders; never unzip archives."""
    path = Path(source) / "utils" / "file_analysis.py"
    text = path.read_text()
    if "CODEX_NUSD_TRAIN_DEV_ALLOWLIST" not in text:
        start = text.index("def get_meta_data(dataset_path):")
    body = '''def get_meta_data(dataset_path):
    """Collect original preprocessing paths for explicitly allowlisted IDs.

    # CODEX_NUSD_TRAIN_DEV_ALLOWLIST
    Archives are never extracted here because they may include reserved TEST.
    """
    import os
    import csv
    manifest = os.environ.get("DAIC_ALLOWED_AUDIO_MANIFEST")
    if not manifest or not os.path.isfile(manifest):
        raise RuntimeError("Set DAIC_ALLOWED_AUDIO_MANIFEST to the frozen TRAIN/DEV audio manifest")
    folder_list, audio_paths, transcript_paths = [], [], []
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            participant_id = int(row["participant_id"])
            audio, transcript = row["audio_path"], row["transcript_path"]
            folder = f"{participant_id:03d}_P"
            if os.path.basename(audio).lower() != f"{participant_id}_audio.wav".lower() or os.path.basename(transcript).lower() != f"{participant_id}_transcript.csv".lower():
                raise ValueError(f"Invalid TRAIN/DEV filename for {participant_id}")
            folder_list.append(folder)
            audio_paths.append(audio)
            transcript_paths.append(transcript)
    return folder_list, audio_paths, transcript_paths
'''
    if "CODEX_NUSD_TRAIN_DEV_ALLOWLIST" not in text:
        path.write_text(text[:start] + body)
    utility = Path(source) / "utils" / "utilities.py"
    utext = utility.read_text()
    if "CODEX_NUSD_AUDIO_GENSIM" not in utext:
        utext = utext.replace("from gensim import corpora", "# CODEX_NUSD_AUDIO_GENSIM\ntry:\n    from gensim import corpora\nexcept ImportError:\n    corpora = None  # audio preprocessing does not use text corpora")
    utext = utext.replace("trial = i.split('/')[-2]", "trial = os.path.basename(i).split('_')[0]")
    utility.write_text(utext)
    audio = Path(source) / "audio" / "audio_file_analysis.py"
    atext = audio.read_text().replace("folder_name = filename.split('/')[-2]", "folder_name = os.path.basename(filename).split('_')[0] + '_P'")
    audio.write_text(atext)


def _replace(pattern, replacement, text, label):
    text, n = re.subn(pattern, replacement, text, count=1)
    if n != 1:
        raise ValueError(f"Could not configure {label}")
    return text


def configure_preprocessor(preprocess_source, daic_root, experiment, metadata_dir):
    """Set author's DAIC preprocessing to raw/16-kHz/SNV, with TRAIN/DEV only."""
    src, daic, exp, meta = map(Path, (preprocess_source, daic_root, experiment, metadata_dir))
    _restrict_audio_scanner(src)
    audio_module = src / "audio" / "audio_file_analysis.py"
    audio_text = audio_module.read_text()
    old = "np.save(current_directory+'/on_times.npy', on_off_times)"
    new = "np.save(current_directory+'/on_times.npy', np.asarray(on_off_times, dtype=object), allow_pickle=True)"
    if old not in audio_text and new not in audio_text:
        raise ValueError("Author timing save statement has changed; inspect the preprocessing source")
    audio_module.write_text(audio_text.replace(old, new))
    # The original runner imports the text pipeline even for raw audio. Its
    # gensim dependency is unnecessary here and can fail on current Colab.
    runner = src / "run" / "__main__.py"
    runner_text = runner.read_text()
    runner_text = runner_text.replace("from text import text_file_analysis\n", "")
    runner_text, count = re.subn(
        r"(?ms)^    if feature_type\[0:4\] == 'text':\n.*?^    print\('Finished Processing'\)",
        "    if feature_type[0:4] == 'text':\n"
        "        from text import text_file_analysis\n"
        "        text_file_analysis.startup()\n"
        "    else:\n"
        "        audio_file_analysis.startup()\n\n"
        "    print('Finished Processing')", runner_text, count=1)
    if count != 1:
        raise ValueError("Author runner layout changed; inspect run/__main__.py")
    runner.write_text(runner_text)
    config = src / "config_files" / "config.py"
    text = config.read_text()
    text = _replace(r"(?ms)^EXPERIMENT_DETAILS\s*=\s*\{.*?\}\s*$",
        "EXPERIMENT_DETAILS = {'FEATURE_EXP': 'raw', 'FREQ_BINS': 1, 'DATASET_IS_BACKGROUND': False, 'WHOLE_TRAIN': False, 'WINDOW_SIZE': 1024, 'OVERLAP': 50, 'SNV': True, 'SAMPLE_RATE': 16000, 'REMOVE_BACKGROUND': True}", text, "preprocessing options")
    paths = {
        "DATASET": str(daic), "WORKSPACE_MAIN_DIR": str(exp / "nusd_audio_cache"),
        "WORKSPACE_FILES_DIR": str(src), "TRAIN_SPLIT_PATH": str(meta / "train.csv"),
        "DEV_SPLIT_PATH": str(meta / "dev.csv"),
        "TEST_SPLIT_PATH_1": str(meta / "test_empty.csv"),
        "TEST_SPLIT_PATH_2": str(meta / "test_empty.csv"),
        "TEST_SPLIT_PATH": str(meta / "test_empty.csv"),
        "FULL_TRAIN_SPLIT_PATH": str(meta / "train_dev.csv"),
        "COMP_DATASET_PATH": str(meta / "train_dev.csv"),
    }
    for key, value in paths.items():
        text = _replace(rf"(?m)^{key}\s*=.*$", f"{key} = {value!r}", text, f"preprocessor {key}")
    config.write_text(text)


def patch_modern_nusd_compat(nusd_source):
    """Small runtime-compatibility edits for current Colab/PyTorch/NumPy."""
    src = Path(nusd_source)
    main = src / "main_disent_fscore_grad.py"
    text = main.read_text()
    text = text.replace("from distutils.dir_util import copy_tree", "from shutil import copytree as copy_tree")
    main.write_text(text)
    data_gen = src / "data_loader" / "data_gen.py"
    text = data_gen.read_text().replace("dtype=np.int)", "dtype=int)")
    data_gen.write_text(text)
    utility = src / "utilities" / "utilities_main.py"
    text = utility.read_text().replace("torch.load(checkpoint_path)", "torch.load(checkpoint_path, weights_only=False)")
    utility.write_text(text)


def configure_nusd(nusd_source, daic_root, experiment, metadata_dir, run_dir, num_runs):
    """Point original NUSD evaluation at the author cache and downloaded checkpoint."""
    src, daic, exp, meta, run = map(Path, (nusd_source, daic_root, experiment, metadata_dir, run_dir))
    patch_modern_nusd_compat(src)
    config = src / "exp_run" / "config_disent_raw_grad.py"
    text = config.read_text()
    paths = {"DATASET": str(daic), "WORKSPACE_MAIN_DIR": str(exp / "nusd_audio_cache"),
             "WORKSPACE_FILES_DIR": str(src), "TRAIN_SPLIT_PATH": str(meta / "train.csv"),
             "DEV_SPLIT_PATH": str(meta / "dev.csv"), "TEST_SPLIT_PATH": str(meta / "test_empty.csv"),
             "FULL_TRAIN_SPLIT_PATH": str(meta / "train_dev.csv"),
             "COMP_DATASET_PATH": str(meta / "train_dev.csv")}
    for key, value in paths.items():
        text = _replace(rf"(?m)^{key}\s*=.*$", f"{key} = {value!r}", text, f"NUSD {key}")
    text = _replace(r"(?m)^\s*'EXP_RUNTHROUGH'\s*:\s*\d+",
                    f"                      'EXP_RUNTHROUGH': {int(num_runs)}", text, "NUSD run count")
    feature_dir = exp / "nusd_audio_cache" / "raw_svn_exp"
    try:
        subdir = run.relative_to(feature_dir)
    except ValueError as e:
        raise ValueError(f"Checkpoint run directory must be under {feature_dir}: {run}") from e
    text = _replace(r"(?ms)^EXPERIMENT_DETAILS\['SUB_DIR'\]\s*=.*?(?=\n\n#)",
                    f"EXPERIMENT_DETAILS['SUB_DIR'] = {str(subdir)!r}", text, "NUSD checkpoint path")
    config.write_text(text)
    return {"feature_dir": str(feature_dir), "sub_dir": str(subdir), "num_runs": int(num_runs)}


def discover_checkpoint_run(feature_dir):
    """Find a published NUSD run containing saved best-epoch metadata."""
    feature_dir = Path(feature_dir)
    candidates = []
    for marker in feature_dir.rglob("model/1/best_scores_total.pickle"):
        run = marker.parent.parent.parent
        count = 0
        while (run / "model" / str(count + 1) / "best_scores_total.pickle").is_file():
            count += 1
        if count:
            candidates.append((count, run))
    if not candidates:
        raise FileNotFoundError(f"No NUSD checkpoint under {feature_dir}. Download the official release folder there first: {WEIGHTS_URL}")
    candidates.sort(key=lambda x: (x[0], len(str(x[1]))), reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise ValueError("Multiple NUSD checkpoints found; specify one explicitly: " + ", ".join(str(p) for _,p in candidates[:5]))
    return candidates[0][1], candidates[0][0]


def export_validation_predictions(run_dir, output, manifest_path, num_runs):
    """Aggregate author crops and released runs into DEV participant probabilities."""
    import pickle
    run, output = Path(run_dir), Path(output)
    manifest = pd.read_csv(manifest_path)
    expected = manifest[manifest.split.eq("val")][["participant_id", "label"]].copy()
    expected.participant_id = expected.participant_id.astype(int)
    result_dirs = sorted(run.glob("results_dict_total"))
    if not result_dirs:
        raise FileNotFoundError(f"NUSD outputs missing under {run}; run original test --validate first")
    result_dir = result_dirs[-1]
    files = sorted(result_dir.glob("*.pickle"), key=lambda p: int(p.stem))
    if len(files) != num_runs:
        raise ValueError(f"Expected {num_runs} NUSD result files, found {len(files)} in {result_dir}")
    frame = expected.copy()
    prob_cols = []
    for path in files:
        with path.open("rb") as f:
            data = pickle.load(f)
        rows = pd.DataFrame({"participant_id": np.asarray(data["folder"]).reshape(-1).astype(int),
                             "label": np.asarray(data["target"]).reshape(-1).astype(int),
                             "probability": np.asarray(data["output"]).reshape(-1).astype(float)})
        rows = rows.groupby(["participant_id", "label"], as_index=False).probability.mean()
        col = f"probability_run_{path.stem}"
        frame = frame.merge(rows.rename(columns={"probability": col}), on=["participant_id", "label"], how="left", validate="one_to_one")
        prob_cols.append(col)
    if frame[prob_cols].isna().any().any() or len(frame) != 34:
        raise ValueError("NUSD outputs did not cover all canonical DEV-34 participants")
    frame["probability"] = frame[prob_cols].mean(axis=1)
    clipped = np.clip(frame["probability"].to_numpy(), 1e-7, 1.0 - 1e-7)
    frame["logit"] = np.log(clipped / (1.0 - clipped))
    frame["prediction"] = (frame.probability >= 0.5).astype(int)
    y, pred, prob = frame.label.to_numpy(), frame.prediction.to_numpy(), frame.probability.to_numpy()
    metrics = {"n": len(frame), "accuracy": float(accuracy_score(y,pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
        "depressed_f1": float(f1_score(y,pred,pos_label=1,zero_division=0)),
        "auroc": float(roc_auc_score(y,prob)), "average_precision": float(average_precision_score(y,prob)),
        "brier": float(brier_score_loss(y,prob)), "log_loss": float(log_loss(y,prob,labels=[0,1])),
        "confusion_matrix": confusion_matrix(y,pred,labels=[0,1]).tolist(),
        "classification_report": classification_report(y,pred,labels=[0,1],target_names=["Non-depressed","Depressed"],output_dict=True,zero_division=0)}
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "val_predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=["actual_0", "actual_1"],
                 columns=["predicted_0", "predicted_1"]).to_csv(output / "confusion_matrix.csv")
    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(output / "classification_report.csv")
    (output / "val_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (output / "provenance.json").write_text(json.dumps({"method":"published NUSD raw ECAPA-TDNN",
        "aggregation":"mean author-crop probability per participant, then mean released-run probability",
        "logit":"logit of the final clipped mean probability; raw model output is sigmoid probability",
        "threshold":0.5,"runs":num_runs,"split":"canonical DEV-34; TEST untouched",
        "preprocessing":"author daic_woz_process; raw 16 kHz audio, SNV, interruption/background fixes",
        "preprocessing_repository":PREPROCESS_REPO,"preprocessing_revision":PREPROCESS_COMMIT,
        "model_repository":NUSD_REPO,"model_revision":NUSD_COMMIT,
        "caveat":"Released checkpoint may have prior DAIC-WOZ validation exposure; DEV results are exploratory."},indent=2)+"\n")
    return metrics


def _main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("prepare"); q.add_argument("--daic-root",required=True); q.add_argument("--split-dir",required=True); q.add_argument("--output",required=True); q.add_argument("--preprocess-source",required=True)
    a=p.parse_args()
    result=prepare_metadata(a.daic_root,a.split_dir,a.output)
    for pid in result["allowed_ids"]:
        folder=Path(a.daic_root)/f"{pid:03d}_P"
        if not folder.is_dir(): raise FileNotFoundError(f"TRAIN/DEV audio folder missing (TEST not searched): {folder}")
        if not any(folder.glob("*.wav")): raise FileNotFoundError(f"No WAV in TRAIN/DEV folder: {folder}")
    configure_preprocessor(a.preprocess_source,a.daic_root,a.output,result["metadata"])
    print(json.dumps({k:(str(v) if isinstance(v,Path) else v) for k,v in result.items()},indent=2))

if __name__=="__main__": _main()
