"""Run the full-188 protocol in ordered, resumable Colab stages."""
import argparse
import importlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


STAGES = ("split", "features", "train_text_teacher", "train_audio_teacher",
          "train_student_branches", "train_student_fusion")


def required(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, default=Path("/content/drive/MyDrive/DAIC_WOZ"))
    p.add_argument("--seed", type=int, choices=(42, 103), default=103)
    p.add_argument("--stage", choices=("all", *STAGES), default="all")
    p.add_argument("--archive-incomplete", action="store_true",
                   help="Move an incomplete stage to failed_attempts before retrying")
    a = p.parse_args([str(value) for value in argv] if argv is not None else None)
    daic = a.daic_root.resolve()
    exp = daic / f"experiments/full188_80_10_10_seed{a.seed}"
    prior = daic / "experiments/full188_seed42/features"
    use_prior = a.seed == 103 and (prior / "provenance.json").is_file() and (prior / "participant_manifest.csv").is_file()
    cache = prior if use_prior else daic / "experiments/ussd_compare16"
    tools = daic.parent / "tools"
    idiap = tools / "solo_teacher_sources/bias_in_daic-woz"
    smile_drive = tools / "opensmile-3.0.2-linux-x86_64"
    smile_local = Path("/content/opensmile-3.0.2-linux-x86_64")

    for path in (daic / "metadata/full_test_split.csv",
                 daic / "metadata/train_split_Depression_AVEC2017.csv",
                 daic / "metadata/dev_split_Depression_AVEC2017.csv",
                 cache / "participant_manifest.csv", idiap / "main.py"):
        required(path)
    needs_extraction = a.stage in ("all", "features") and not (exp / "features/provenance.json").is_file() and not use_prior
    if needs_extraction:
        if not smile_local.exists():
            if not smile_drive.is_dir():
                raise FileNotFoundError(f"OpenSMILE release missing: {smile_drive}")
            print(f"Copying OpenSMILE to {smile_local}...", flush=True)
            shutil.copytree(smile_drive, smile_local)
        config = required(smile_local / "config/compare16/ComParE_2016.conf")
        binary = required(smile_local / "bin/SMILExtract")
        binary.chmod(binary.stat().st_mode | 0o111)
    else:
        config = smile_drive / "config/compare16/ComParE_2016.conf" if use_prior else smile_local / "config/compare16/ComParE_2016.conf"
        binary = smile_local / "bin/SMILExtract"

    split = exp / "split"
    features = exp / "features"
    text_teacher = exp / "teachers/text"
    audio_teacher = exp / "teachers/audio"
    branches = exp / "students/branches"
    fusion = exp / "students/fusion"
    specs = {
        "split": (split, "complete.json", ["--daic-root", daic, "--output", split, "--seed", a.seed]),
        "features": (features, "provenance.json", ["--split-dir", split, "--daic-root", daic,
                     "--canonical-features", cache, "--output", features,
                     "--compare16-config", config, "--smile-extract", binary]),
        "train_text_teacher": (text_teacher, "audit.json", ["--daic-root", daic,
                               "--split-dir", split, "--idiap-source", idiap, "--output", text_teacher, "--seed", a.seed]),
        "train_audio_teacher": (audio_teacher, "audit.json", ["--split-dir", split,
                                "--features", features, "--output", audio_teacher, "--batch-size", 16, "--seed", a.seed]),
        "train_student_branches": (branches, "audit.json", ["--daic-root", daic,
                                   "--split-dir", split, "--features", features,
                                   "--audio-teacher", audio_teacher, "--output", branches, "--seed", a.seed]),
        "train_student_fusion": (fusion, "audit.json", ["--split-dir", split,
                                 "--student-branches", branches, "--text-teacher", text_teacher,
                                 "--audio-teacher", audio_teacher, "--output", fusion, "--seed", a.seed]),
    }
    exp.mkdir(parents=True, exist_ok=True)
    for name in (STAGES if a.stage == "all" else (a.stage,)):
        output, marker, args = specs[name]
        if (output / marker).is_file():
            if name == "split" and json.loads((output / marker).read_text())["signature"]["seed"] != a.seed:
                raise ValueError("Completed split belongs to another seed")
            print(f"{name}: complete, skipped ({output})", flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            if not a.archive_incomplete:
                raise RuntimeError(f"{name} has incomplete output at {output}. Inspect it or pass --archive-incomplete to preserve it and retry.")
            archive = exp / "failed_attempts" / f"{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
            print(f"Preserved incomplete {name} at {archive}", flush=True)
        print(f"Starting {name}: {output}", flush=True)
        importlib.import_module(f"scripts.experiments.full188.{name}").main(list(map(str, args)))
        required(output / marker)
        print(f"Finished {name}", flush=True)
    print(f"Experiment: {exp}", flush=True)


if __name__ == "__main__":
    main()
