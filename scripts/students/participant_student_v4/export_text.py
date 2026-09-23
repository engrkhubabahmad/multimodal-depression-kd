"""Reuse v3 weights/vectorizer; export both splits via its DEV inference path."""
from __future__ import annotations
import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from .common import (cached_or_create, check_splits, complete, load_embeddings,
                     scores, sha, shift_summary, write_json)


def inductive_forward(x, w1, wout, words):
    """Exact eval-mode algebra of v3 dev_repr_logits, without a dense graph.

    B=[X,I], H0=[I;X], so B@H0=2X and B@[Hwords;Hdev]=X@Hwords+Hdev.
    Preserve the checkpoint's word state; do not sample dropout or refit it.
    """
    x, w1, wout, words = [np.asarray(a, dtype=np.float32) for a in (x, w1, wout, words)]
    if x.ndim != 2 or w1.ndim != 2 or words.shape != (x.shape[1], w1.shape[0]):
        raise ValueError("Invalid text representation dimensions")
    if w1.shape[1] != x.shape[1] or wout.shape != (2, w1.shape[0]):
        raise ValueError("Invalid classifier dimensions")
    if not all(np.isfinite(a).all() for a in (x, w1, wout, words)):
        raise ValueError("Non-finite text inference input")
    embedding = x @ words + np.maximum((2 * x) @ w1.T, 0)
    logits = embedding @ wout.T
    return embedding, expit(logits[:, 1] - logits[:, 0])


def load_weights(path):
    import torch  # Colab dependency, needed only for the trusted saved checkpoint.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    return tuple(a.detach().cpu().numpy() for a in
                 (state["h1.0.weight"], state["out.weight"], ckpt["H1_words"]))


def split_table(root, split):
    # Only official TRAIN/DEV files are named or read by this module.
    p = root / "metadata" / f"{split}_split_Depression_AVEC2017.csv"
    d = pd.read_csv(p); d.columns = d.columns.str.strip().str.lower()
    d = d[["participant_id", "phq8_binary"]].rename(columns={"phq8_binary": "label"})
    if d.isna().any().any() or not np.isin(d.label, [0, 1]).all():
        raise ValueError(f"Invalid split metadata: {p}")
    if not np.equal(d.participant_id, d.participant_id.astype(int)).all():
        raise ValueError("Non-integer participant ID")
    d = d.astype(int)
    if split == "dev": d = d.loc[d.participant_id != 440]
    if d.participant_id.duplicated().any(): raise ValueError("Duplicate split participant")
    return d.sort_values("participant_id").reset_index(drop=True), p


def transcript_paths(root, ids):
    wanted = set(map(int, ids)); found = {}
    for p in root.rglob("*_TRANSCRIPT.csv"):
        m = re.fullmatch(r"(\d+)_TRANSCRIPT\.csv", p.name, re.I)
        if m and int(m[1]) in wanted: found.setdefault(int(m[1]), []).append(p)
    bad = [pid for pid in wanted if len(found.get(pid, [])) != 1]
    if bad: raise ValueError(f"Missing or duplicate transcripts: {sorted(bad)}")
    return {pid: found[pid][0] for pid in wanted}


def document(path):
    d = pd.read_csv(path, sep="\t"); d.columns = d.columns.str.strip().str.lower()
    if not {"speaker", "value"} <= set(d.columns):
        d = pd.read_csv(path, sep=None, engine="python"); d.columns = d.columns.str.strip().str.lower()
    v = d.loc[d.speaker.astype(str).str.strip().str.lower().eq("participant"), "value"]
    text = " ".join(v.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()).strip()
    if not text: raise ValueError(f"Empty participant document: {path.name}")
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--text-branch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv); source = a.text_branch
    tables, inputs = {}, {}
    for split in ("train", "dev"):
        tables[split], path = split_table(a.daic_root, split); inputs[str(path)] = sha(path)
    check_splits(*[{"participant_ids": d.participant_id.to_numpy(), "labels": d.label.to_numpy()}
                   for d in tables.values()])
    paths = transcript_paths(a.daic_root, pd.concat([d.participant_id for d in tables.values()]))
    for path in paths.values(): inputs[str(path)] = sha(path)
    for name in ("inference_state.pt", "vectorizer.pkl", "metrics.json",
                 "train_text_embeddings.npz", "dev_text_embeddings.npz"):
        inputs[str(source / name)] = sha(source / name)
    signature = {"version": 1, "method": "v3_fixed_word_state_inductive_eval_both_splits",
                 "code": {n: sha(Path(__file__).with_name(n)) for n in ("export_text.py", "common.py")},
                 "inputs": inputs}
    if cached_or_create(a.output, signature):
        print("Verified existing consistent text export:", a.output); return
    with (source / "vectorizer.pkl").open("rb") as f: vectorizer = pickle.load(f)
    weights = load_weights(source / "inference_state.pt")
    old, new, comparison = {}, {}, {}
    for split, d in tables.items():
        print(f"Exporting {split}: {len(d)} participant transcripts")
        docs = [document(paths[pid]) for pid in d.participant_id]
        # transform only: vocabulary, IDF, weights and word state are all frozen.
        x = vectorizer.transform(docs).toarray().astype(np.float32)
        embedding, probability = inductive_forward(x, *weights)
        new[split] = {"participant_ids": d.participant_id.to_numpy(), "labels": d.label.to_numpy(),
                      "embedding": embedding, "probability": probability}
        old[split] = load_embeddings(source / f"{split}_text_embeddings.npz", len(d))
        for key in ("participant_ids", "labels"):
            if not np.array_equal(new[split][key], old[split][key]):
                raise ValueError(f"Source metadata mismatch: {split}/{key}")
        comparison[split] = {"old": scores(d.label, old[split]["probability"]),
                             "consistent": scores(d.label, probability)}
    # This is an implementation equivalence check, not DEV fitting or selection.
    if not np.allclose(new["dev"]["embedding"], old["dev"]["embedding"], rtol=1e-4, atol=1e-5):
        raise ValueError("DEV embedding parity failed. Stop and inspect checkpoint/vectorizer provenance.")
    if not np.allclose(new["dev"]["probability"], old["dev"]["probability"], rtol=0, atol=1e-6):
        raise ValueError("DEV probability parity failed")
    if not np.array_equal(new["dev"]["probability"] >= .5, old["dev"]["probability"] >= .5):
        raise ValueError("DEV prediction parity failed")
    for split, data in new.items():
        np.savez_compressed(a.output / f"{split}_text_embeddings.npz", **data)
    audit = {"protocol": {"train_participants": 107, "dev_participants": 34,
              "participant_440_excluded": True, "same_inference_path": True,
              "word_state": "fixed checkpoint H1_words; no new dropout",
              "vocabulary_refit": False, "model_retrained": False,
              "oof": False, "reads_test_data": False, "dev_parity_passed": True},
             "metrics": comparison,
             "old_embedding_shift": shift_summary(old["train"]["embedding"], old["dev"]["embedding"]),
             "consistent_embedding_shift": shift_summary(new["train"]["embedding"], new["dev"]["embedding"]),
             "source_branch": str(source.resolve())}
    write_json(a.output / "audit.json", audit)
    complete(a.output, signature)
    print("Text export complete; unchanged DEV predictions verified:", a.output)


if __name__ == "__main__": main()
