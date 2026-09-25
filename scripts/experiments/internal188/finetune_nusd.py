"""Fine-tune the released single-run NUSD ECAPA teacher on canonical TRAIN-107."""
from __future__ import annotations
import argparse, json, os, pickle, subprocess, sys
from pathlib import Path
import pandas as pd
from .nusd_ecapa import (configure_nusd_finetune, discover_checkpoint_run,
                         select_best_nusd_run)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--name", default="nusd_run2_finetune_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args(argv)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime")
    experiment = args.experiment.resolve()
    metadata = experiment / "teachers" / "nusd_ecapa_frozen" / "metadata"
    split = pd.read_csv(experiment / "split" / "manifest.csv")
    train = set(split.loc[split.split.eq("train"), "participant_id"].astype(int))
    dev = set(split.loc[split.split.eq("val"), "participant_id"].astype(int))
    if len(train) != 107 or len(dev) != 34 or train & dev or 440 in train | dev:
        raise ValueError("Expected disjoint canonical TRAIN-107 and DEV-34, excluding 440")
    for filename, expected in (("train.csv", train), ("dev.csv", dev)):
        actual = set(pd.read_csv(metadata / filename).Participant_ID.astype(int))
        if actual != expected:
            raise ValueError(f"{filename} IDs differ from frozen split")
    frozen = experiment / "teachers" / "nusd_ecapa_frozen"
    cache = frozen / "nusd_audio_cache" / "raw_svn_exp"
    for filename in ("complete_database.h5", "summary.pickle"):
        if not (cache / filename).is_file():
            raise FileNotFoundError(cache / filename)
    released, _ = discover_checkpoint_run(cache)
    chosen, author_scores = select_best_nusd_run(released)
    destination = cache / args.name
    if destination.exists():
        raise FileExistsError(f"Fine-tuning output already exists: {destination}; choose a new --name")
    daic_root = experiment.parents[1]
    configure_nusd_finetune(args.source, daic_root, frozen, metadata,
                            args.name, args.epochs, args.lr)
    provenance = {"split":"canonical TRAIN-107/DEV-34", "train_ids":sorted(train),
                  "dev_ids":sorted(dev), "test_opened":False,
                  "source_checkpoint":chosen, "author_validation_scores":author_scores,
                  "epochs":args.epochs, "learning_rate":args.lr,
                  "output":str(destination), "interpretation":"exploratory: source checkpoint was selected on canonical DEV"}
    record = frozen / f"{args.name}_provenance.json"
    record.write_text(json.dumps(provenance, indent=2) + "\n")
    env = os.environ.copy(); env["DAIC_NUSD_WARM_START"] = chosen["weights"]
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("DAIC_NUSD_RUN_INDEX", None)
    env.pop("DAIC_NUSD_FROZEN_EVAL", None)
    log_path = frozen / f"{args.name}_train.log"
    print("Fine-tuning single NUSD checkpoint:", chosen, flush=True)
    print("TRAIN=107 DEV=34; output:", destination, flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen([sys.executable, "-u", "main_disent_fscore_grad.py",
                                 "train", "--validate", "--cuda", "--position=1"],
                                cwd=args.source, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        code = proc.wait()
    if code:
        raise RuntimeError(f"Fine-tuning exited {code}; inspect {log_path}")
    marker = destination / "model" / "1" / "best_scores_fscore.pickle"
    if not marker.is_file():
        raise FileNotFoundError(f"Training exited without best checkpoint: {marker}; inspect {log_path}")
    with marker.open("rb") as f:
        scores = pickle.load(f)
    result = {"best_epoch":int(scores[-1]), "author_recorded_dev_macro_f1":float(scores[10]),
              "checkpoint":str(marker.parent / f"md_{scores[-1]}_epochs.pth")}
    (frozen / f"{args.name}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Fine-tuning complete:", result, flush=True)


if __name__ == "__main__":
    main()
