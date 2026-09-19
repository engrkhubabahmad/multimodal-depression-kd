"""Raw DAIC-WOZ participant-level inference using Idiap InducT-GCN artifacts.

This runner deliberately uses only licensed local transcript CSV files for
inference. It never opens TEST, never creates segments, and never silently
uses the checkpoint-stored DEV feature matrix.
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm

DEV_IDS = (302,307,331,335,346,367,377,381,382,388,389,390,395,403,404,406,413,417,418,420,422,436,439,440,451,458,472,476,477,482,483,484,489,490,492)

def one(root, pattern):
    files = list(root.rglob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one {pattern}; found {len(files)}")
    return files[0]

def participant_document(root, pid):
    path = one(root, f"{pid}_TRANSCRIPT.csv")
    frame = pd.read_csv(path, sep="\\t").fillna("")
    required = {"start_time","stop_time","speaker","value"}
    if required - set(frame.columns):
        raise ValueError(f"{path}: required transcript columns are absent")
    frame = frame.sort_values(["start_time","stop_time"], kind="stable")
    speaker = frame["speaker"].astype(str).str.strip().str.casefold()
    value = frame["value"].astype(str).str.replace("\\x00"," ", regex=False).str.replace(r"\\s+"," ", regex=True).str.strip()
    keep = (speaker == "participant") & value.ne("") & ~value.str.contains("scrubbed_entry", case=False, regex=False)
    text = " ".join(value[keep].tolist())
    if not text:
        raise ValueError(f"{pid}: no non-scrubbed Participant utterances")
    return text

def load_author(path):
    spec = importlib.util.spec_from_file_location("idiap_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_graph(model, documents, vectorizer):
    # Matches InducT-GCN test construction: fixed vocabulary nodes + new document nodes.
    x = torch.as_tensor(vectorizer.transform(documents).toarray(), dtype=torch.float32)
    vocab = len(model._vocab)
    x = x[:, :vocab]
    h0 = torch.cat([torch.eye(vocab), x], dim=0)
    adjacency = torch.zeros((len(documents), vocab + len(documents)), dtype=torch.float32)
    adjacency[:, :vocab] = x
    adjacency[:, vocab:] = torch.eye(len(documents))
    model = model.cpu().eval()
    with torch.no_grad():
        h_docs = model.get_H_1(adjacency @ h0)
        h1 = torch.cat([model.H_1_words[:vocab].cpu(), h_docs], dim=0)
        return torch.softmax(model.node_emb2out(adjacency @ h1), dim=-1).numpy()

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--daic-root", required=True)
    p.add_argument("--source-root", default="/content/solo_teacher_sources/bias_in_daic-woz")
    p.add_argument("--dev-csv")
    p.add_argument("--output", required=True)
    p.add_argument("--exclude", type=int, nargs="*", default=[440])
    a = p.parse_args(argv)
    root, source, out = Path(a.daic_root), Path(a.source_root), Path(a.output)
    split = Path(a.dev_csv) if a.dev_csv else one(root, "dev_split_Depression_AVEC2017.csv")
    labels = pd.read_csv(split)
    columns = {c.casefold(): c for c in labels.columns}
    label_map = dict(zip(labels[columns["participant_id"]].astype(int), labels[columns["phq8_binary"]].astype(int)))
    ids = [pid for pid in DEV_IDS if pid not in set(a.exclude)]
    documents = [participant_document(root, pid) for pid in tqdm(ids, desc="Idiap raw participant preprocessing", colour="green")]
    author = load_author(source / "main.py")
    with (source / "model/Participant/vtzer_inductgcn[250].pkl").open("rb") as f:
        vectorizer = pickle.load(f)
    state = torch.load(source / "model/Participant/model_inductgcn[250].pkl", map_location="cpu", weights_only=False)
    model = author.InducTGCN(state["embedding_dim"], state["classes_"], 0, vectorizer)
    model.load_state_dict(state["model_state_dict"])
    model.A_B = state["A_dev"]
    model.classes_ = state["classes_"]
    probability = run_graph(model, documents, vectorizer)[:, list(model.classes_).index("positive")]
    pred = (probability >= 0.5).astype(int)
    y = np.array([label_map[pid] for pid in ids])
    result = {"n": len(ids), "accuracy": float(accuracy_score(y, pred)), "macro_f1": float(classification_report(y, pred, output_dict=True, zero_division=0)["macro avg"]["f1-score"]), "auroc": float(roc_auc_score(y, probability)), "confusion_matrix": confusion_matrix(y, pred, labels=[0,1]).tolist(), "threshold": 0.5, "input": "raw_local_transcripts", "test_opened": False, "excluded_ids": sorted(a.exclude)}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"participant_id":ids, "label":y, "prob_depressed":probability, "prediction":pred, "document":documents}).to_json(out / "dev_raw_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    (out / "dev_raw_metrics.json").write_text(__import__("json").dumps(result, indent=2) + "\\n")
    print(result)
    return result

if __name__ == "__main__":
    main()
