"""Score exactly one frozen student mode on the 19-person internal test."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.students.participant_student_v3.audio_model import CompactAudioBranch
from scripts.students.participant_student_v3.pretrain_text import InductText, transcript_map, participant_doc
from scripts.students.participant_student_v3.fusion_model import FrozenBranchFusion
from .features import verified_split
from .split import digest
from .train_audio_teacher import evaluate, rows_for
from .train_student_fusion import predict
from .train_text_teacher import metric


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--student-branches", type=Path, required=True)
    p.add_argument("--audio-teacher", type=Path, required=True, help="Read TRAIN-only normalization, never teacher weights")
    p.add_argument("--student-fusion", type=Path, required=True)
    p.add_argument("--mode", choices=("no_kd", "standard_kd", "ra_kd"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    split = verified_split(a.split_dir)
    test = split.loc[split.split.eq("student_test")].sort_values("participant_id")
    expected_sha = digest(a.split_dir / "manifest.csv")
    for folder in (a.student_branches, a.audio_teacher, a.student_fusion):
        audit = json.loads((folder / "audit.json").read_text())
        if audit["signature"].get("split_sha256") != expected_sha:
            raise ValueError(f"Incompatible split artifact: {folder}")
    lock = a.student_fusion / "student_test_selection.json"
    selected = {"mode": a.mode, "output": str(a.output.resolve())}
    if lock.exists() and json.loads(lock.read_text()) != selected:
        raise ValueError("Student test already scored with another selection")
    if a.output.exists() and any(a.output.iterdir()):
        marker = a.output / "complete.json"
        saved = json.loads(marker.read_text()) if marker.is_file() else {}
        predictions = a.output / "student_test_predictions.csv"
        if saved.get("selection") == selected and predictions.is_file() and \
                saved.get("predictions_sha256") == digest(predictions):
            print("Student test already scored; reusing:", a.output); return
        raise ValueError("Refusing to overwrite student test output")
    indexed = rows_for(a.features, split)
    paths = transcript_map(a.daic_root, test.participant_id)
    with (a.student_branches / "text_vectorizer.pkl").open("rb") as f: v = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model = InductText(250).to(device)
    text_state = torch.load(a.student_branches / "text_best.pt", map_location=device, weights_only=False)
    text_model.load_state_dict(text_state["weights"]); text_model.eval()
    docs = [participant_doc(paths[int(pid)]) for pid in test.participant_id]
    x = torch.tensor(v.transform(docs).toarray(), dtype=torch.float32, device=device)
    with torch.inference_mode():
        text_embedding, _ = text_model.dev_repr_logits(x, text_state["word_state"])
        text_embedding = text_embedding.cpu().numpy().astype(np.float32)
    with np.load(a.audio_teacher / "train_only_normalization.npz") as stats:
        mean, std = stats["mean"], stats["std"]
    audio_model = CompactAudioBranch().to(device)
    audio_state = torch.load(a.student_branches / "audio_best.pt", map_location=device, weights_only=False)
    audio_model.load_state_dict(audio_state["weights"])
    _, audio_embedding = evaluate(audio_model, test, indexed, mean, std, device, 16)
    with np.load(a.student_fusion / "train_only_scalers.npz") as z:
        audio_x = ((audio_embedding - z["audio_mean"]) / z["audio_std"]).astype(np.float32)
        text_x = ((text_embedding - z["text_mean"]) / z["text_std"]).astype(np.float32)
    student = FrozenBranchFusion().to(device)
    state = torch.load(a.student_fusion / f"{a.mode}_best.pt", map_location=device, weights_only=False)
    student.load_state_dict(state["weights"])
    prob = predict(student, audio_x, text_x, device)
    result = metric(test.label.to_numpy(int), prob)
    a.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"participant_id": test.participant_id.to_numpy(int),
                  "label": test.label.to_numpy(int), "probability": prob,
                  "prediction": (prob >= .5).astype(int)}).to_csv(a.output / "student_test_predictions.csv", index=False)
    (a.output / "result.json").write_text(json.dumps({"selection": selected,
        "internal_test_n": 19, "teacher_inference_on_test": False,
        "canonical_avec_test": False, "metrics": result}, indent=2) + "\n")
    (a.output / "complete.json").write_text(json.dumps({"selection": selected,
        "split_sha256": expected_sha,
        "predictions_sha256": digest(a.output / "student_test_predictions.csv")}, indent=2) + "\n")
    lock.write_text(json.dumps(selected, indent=2) + "\n")
    print("Internal student test:", a.mode, result)


if __name__ == "__main__": main()
