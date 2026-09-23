"""One student-only evaluation on the reserved 21-person internal holdout."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .common import cached_or_create, complete, scores, sha, write_json
from .compact_fusion import predict
from .export_text import document, transcript_paths
from .split42_baseline import audio_paths, audio_summary


def run(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--local-audio-root", type=Path)
    p.add_argument("--split-baseline", type=Path, required=True)
    p.add_argument("--student-run", type=Path, required=True)
    p.add_argument("--student-mode", choices=("no_kd", "standard_kd", "ra_kd"), required=True)
    p.add_argument("--allow-exploratory-proxy-test", action="store_true",
                   help="Acknowledge this student uses shallow proxy teachers, not the original teacher architectures")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if not a.allow_exploratory_proxy_test:
        raise ValueError("Reserved student test is closed for the proxy experiment. "
                         "Only an explicitly exploratory proxy test may use --allow-exploratory-proxy-test; "
                         "retrain the original teachers first for publication claims.")
    if a.output.resolve() == a.student_run.resolve() or a.output.resolve() == a.split_baseline.resolve():
        raise ValueError("Use a distinct test-only output directory")
    for root in (a.split_baseline, a.student_run):
        marker = json.loads((root / "complete.json").read_text())
        cached_or_create(root, marker["signature"])
    audit = json.loads((a.student_run / "audit.json").read_text())["protocol"]
    if audit["internal_student_test_scored"] or audit["teacher_target_participants"] != 99:
        raise ValueError("Student training artifact is incompatible")
    lock_path = a.student_run / "student_test_selection.json"
    chosen = {"mode": a.student_mode, "output": str(a.output.resolve())}
    if lock_path.exists() and json.loads(lock_path.read_text()) != chosen:
        raise ValueError("Student test already evaluated with a different mode/output")
    manifest = pd.read_csv(a.split_baseline / "split_manifest.csv")
    if manifest.experimental_split.value_counts().to_dict() != {"train": 99, "dev": 21, "holdout": 21}:
        raise ValueError("Invalid split")
    holdout = manifest.loc[manifest.experimental_split.eq("holdout")].sort_values("participant_id")
    if set(holdout.participant_id) & set(pd.read_csv(a.student_run / "train_teacher_logits_probabilities.csv").participant_id):
        raise ValueError("Teacher target leakage into student test")
    paths = transcript_paths(a.daic_root, holdout.participant_id)
    # The cache manifest may contain TRAIN+DEV; only these 21 feature files are read.
    full = manifest[["participant_id", "label"]]
    audio = audio_paths(a.features, full, a.local_audio_root)
    head_path = a.student_run / f"{a.student_mode}_student.npz"
    source = {str(q): sha(q) for q in (a.split_baseline / "complete.json",
              a.student_run / "complete.json", a.split_baseline / "split_manifest.csv",
              a.split_baseline / "train_vectorizer.pkl", head_path)}
    for path in (*paths.values(), *(audio[int(pid)] for pid in holdout.participant_id)):
        source[str(path)] = sha(path)
    signature = {"version": 1, "student_mode": a.student_mode, "inputs": source,
                 "code": {n: sha(Path(__file__).with_name(n)) for n in
                          ("score_split42_student.py", "split42_baseline.py", "compact_fusion.py", "common.py")}}
    if cached_or_create(a.output, signature):
        if not lock_path.exists(): write_json(lock_path, chosen)
        print("Verified previous student-only evaluation:", a.output); return
    with (a.split_baseline / "train_vectorizer.pkl").open("rb") as f: vectorizer = pickle.load(f)
    ids = holdout.participant_id.to_numpy(int); y = holdout.label.to_numpy(int)
    text = vectorizer.transform([document(paths[int(pid)]) for pid in ids]).toarray()
    audio_x = np.stack([audio_summary(audio[int(pid)]) for pid in ids])
    with np.load(head_path, allow_pickle=False) as z: head = {k: z[k] for k in z.files}
    probability = predict(head, np.column_stack([text, audio_x]), np.zeros(len(ids)))
    result = scores(y, probability)
    pd.DataFrame({"participant_id": ids, "label": y, "probability": probability,
                  "prediction": (probability >= .5).astype(int)}).to_csv(a.output / "student_test_predictions.csv", index=False)
    write_json(a.output / "student_test_metrics.json", {"student_mode": a.student_mode,
        "student_only": True, "n": 21, "official_blind_test_opened": False,
        "teacher_inference_on_student_test": False, "metrics": result})
    complete(a.output, signature)
    write_json(lock_path, chosen)
    print(f"Student-only internal test ({a.student_mode}): macroF1={result['macro_f1']:.4f}"
          f" depressedF1={result['depressed_f1']:.4f} CM={result['confusion_matrix']}")


if __name__ == "__main__": run()
