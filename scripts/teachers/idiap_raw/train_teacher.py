"""Train an Idiap-style InducT-GCN teacher from raw DAIC-WOZ participants.

The public Idiap checkpoint is not used for training because its original
prepared TRAIN documents are unavailable.  This script uses the released
InducTGCN implementation, but fits all data-dependent preprocessing on the
official local TRAIN participants only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

TRAIN_EXCLUDED = {440}


def one(root: Path, pattern: str) -> Path:
    files = list(root.rglob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern!r}; found {len(files)}")
    return files[0]


def split_labels(path: Path) -> list[tuple[int, int]]:
    frame = pd.read_csv(path)
    cols = {c.casefold(): c for c in frame.columns}
    if "participant_id" not in cols or "phq8_binary" not in cols:
        raise ValueError(f"{path} needs Participant_ID and PHQ8_Binary")
    return [(int(pid), int(label)) for pid, label in zip(frame[cols["participant_id"]], frame[cols["phq8_binary"]])]


def participant_document(root: Path, pid: int) -> str:
    path = one(root, f"{pid}_TRANSCRIPT.csv")
    frame = pd.read_csv(path, sep="\t").fillna("")
    required = {"start_time", "stop_time", "speaker", "value"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{path}: missing {sorted(missing)}")
    frame = frame.sort_values(["start_time", "stop_time"], kind="stable")
    speaker = frame.speaker.astype(str).str.strip().str.casefold()
    value = frame.value.astype(str).str.replace("\x00", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()
    keep = (speaker == "participant") & value.ne("") & ~value.str.contains("scrubbed_entry", case=False, regex=False)
    document = " ".join(value[keep].tolist())
    if not document:
        raise ValueError(f"{pid}: no usable Participant text")
    return document


def load_author(path: Path):
    spec = importlib.util.spec_from_file_location("idiap_author_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def document_split(root: Path, csv_path: Path, split: str, excluded: set[int]):
    samples = [(pid, label) for pid, label in split_labels(csv_path) if pid not in excluded]
    rows = []
    for pid, label in tqdm(samples, desc=f"Raw {split} participant documents", colour="green"):
        rows.append({"participant_id": pid, "label": label, "document": participant_document(root, pid)})
    return pd.DataFrame(rows)


def probabilities(model, documents: list[str]) -> np.ndarray:
    model.Conv_0_Test = None
    output = model(documents).detach().cpu().numpy()
    return output[:, list(model.classes_).index("positive")]


def report(y: np.ndarray, p: np.ndarray) -> dict:
    prediction = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(classification_report(y, prediction, output_dict=True, zero_division=0)["macro avg"]["f1-score"]),
        "auroc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "threshold": 0.5,
    }


def build_vectorizer(train_documents: list[str], y_train: np.ndarray, features: int):
    initial = TfidfVectorizer(stop_words="english")
    initial.fit(train_documents)
    names = initial.get_feature_names_out()
    matrix = initial.transform(train_documents)
    keep = SelectKBest(f_classif, k=min(features, matrix.shape[1])).fit(matrix, y_train).get_support()
    vocabulary = [names[i] for i, selected in enumerate(keep) if selected]
    vectorizer = TfidfVectorizer(stop_words="english", vocabulary=vocabulary)
    vectorizer.fit(train_documents)
    return vectorizer


def train_trial(author, train_docs, y_train, dev_docs, y_dev, vectorizer, lr, epochs, patience, device):
    classes = ["negative", "positive"]
    author.DEVICE = device
    model = author.InducTGCN(64, classes, 0.5, vectorizer)
    model.build_graph(train_docs, window_size=3, verbose=False)
    model.to(device)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    weight = torch.tensor(weights, dtype=torch.float, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_state, best_metrics, stalled = None, None, 0
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    for epoch in tqdm(range(1, epochs + 1), desc=f"Idiap train lr={lr:g}", colour="green"):
        model.train()
        model.zero_grad()
        loss = model.cross_entropy_loss_on_document_nodes(y_tensor, class_weight=weight)
        loss.backward()
        optimizer.step()

        p_dev = probabilities(model, dev_docs)
        current = report(y_dev, p_dev)
        current["epoch"] = epoch
        current["loss"] = float(loss.item())
        if best_metrics is None or current["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = current
            best_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
        if stalled >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_metrics


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--daic-root", required=True)
    p.add_argument("--source-root", default="/content/solo_teacher_sources/bias_in_daic-woz")
    p.add_argument("--output", required=True)
    p.add_argument("--train-csv")
    p.add_argument("--dev-csv")
    p.add_argument("--features", type=int, default=250)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--learning-rates", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--exclude", type=int, nargs="*", default=[])
    a = p.parse_args(argv)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    root, source, out = Path(a.daic_root), Path(a.source_root), Path(a.output)
    excluded = set(a.exclude)
    train_csv = Path(a.train_csv) if a.train_csv else one(root, "train_split_Depression_AVEC2017.csv")
    dev_csv = Path(a.dev_csv) if a.dev_csv else one(root, "dev_split_Depression_AVEC2017.csv")
    train = document_split(root, train_csv, "TRAIN", excluded)
    dev = document_split(root, dev_csv, "DEV", excluded)

    if len(train) != 107 or len(dev) != 34:
        raise AssertionError(f"Expected TRAIN=107 and DEV=34, got {len(train)}, {len(dev)}")
    if set(train.participant_id) & set(dev.participant_id):
        raise AssertionError("Participant overlap detected")
    if train.label.nunique() != 2:
        raise AssertionError("TRAIN must contain both classes")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    author = load_author(source / "main.py")
    vectorizer = build_vectorizer(train.document.tolist(), train.label.to_numpy(), a.features)
    candidates = []
    best_model, best_info = None, None
    for lr in a.learning_rates:
        model, info = train_trial(
            author, train.document.tolist(), train.label.to_numpy(),
            dev.document.tolist(), dev.label.to_numpy(), vectorizer,
            lr, a.epochs, a.patience, device
        )
        info["learning_rate"] = lr
        candidates.append(info)
        print({"learning_rate": lr, **info})
        if best_info is None or info["macro_f1"] > best_info["macro_f1"]:
            best_model, best_info = model, info

    out.mkdir(parents=True, exist_ok=True)
    train_p = probabilities(best_model, train.document.tolist())
    dev_p = probabilities(best_model, dev.document.tolist())
    for name, frame, prob in (("train", train, train_p), ("dev", dev, dev_p)):
        table = frame[["participant_id", "label"]].copy()
        table["prob_depressed"] = prob
        table["logit_depressed"] = np.log(np.clip(prob, 1e-6, 1 - 1e-6) / np.clip(1 - prob, 1e-6, 1))
        table["prediction"] = (prob >= 0.5).astype(int)
        table.to_csv(out / f"{name}_predictions.csv", index=False)

    state = {
        "model_state_dict": best_model.state_dict(),
        "classes_": best_model.classes_,
        "embedding_dim": 64,
        "features": a.features,
        "seed": a.seed,
        "learning_rate": best_info["learning_rate"],
        "selection": "DEV macro_f1 at fixed threshold 0.5",
    }
    torch.save(state, out / "frozen_idiap_style_teacher.pt")
    with (out / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    summary = {
        "protocol": "raw participant-level Idiap-style InducT-GCN",
        "train": report(train.label.to_numpy(), train_p),
        "dev": report(dev.label.to_numpy(), dev_p),
        "selected": best_info,
        "candidates": candidates,
        "train_participants": int(len(train)),
        "dev_participants": int(len(dev)),
        "test_opened": False,
        "excluded_ids": sorted(excluded),
    }
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
