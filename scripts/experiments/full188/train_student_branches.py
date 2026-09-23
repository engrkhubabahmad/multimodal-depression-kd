"""Independently train compact text/audio student branches on TRAIN-132."""
from __future__ import annotations
import argparse
import json
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scripts.students.participant_student_v3.pretrain_text import (
    InductText, select_vectorizer, build_graph, transcript_map, participant_doc)
from scripts.students.participant_student_v3.audio_model import CompactAudioBranch
from .features import verified_split
from .split import digest
from .train_audio_teacher import Segments, evaluate, rows_for
from .train_text_teacher import metric


def train_text(root, train, val, output, seed, epochs):
    paths = transcript_map(root, pd.concat([train.participant_id, val.participant_id]))
    docs = {int(pid): participant_doc(paths[int(pid)]) for pid in pd.concat([train.participant_id, val.participant_id])}
    tr_docs = [docs[int(pid)] for pid in train.participant_id]
    dv_docs = [docs[int(pid)] for pid in val.participant_id]
    y = train.label.to_numpy(int); vy = val.label.to_numpy(int)
    vectorizer = select_vectorizer(tr_docs, y, 250)
    graph, conv, _ = build_graph(tr_docs, vectorizer, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InductText(250).to(device); graph = graph.to(device); conv = conv.to(device)
    xtrain = torch.tensor(vectorizer.transform(tr_docs).toarray(), dtype=torch.float32, device=device)
    xval = torch.tensor(vectorizer.transform(dv_docs).toarray(), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    class_weight = torch.tensor([len(y) / (2 * sum(y == 0)), len(y) / (2 * sum(y == 1))],
                                device=device, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.long, device=device)
    best = (-1., -1., -1.); stale = 0
    for epoch in range(1, epochs + 1):
        model.train(); opt.zero_grad(set_to_none=True)
        _, logits = model.train_repr_logits(graph, conv)
        loss = torch.nn.functional.cross_entropy(logits[250:], target, weight=class_weight)
        loss.backward(); opt.step()
        if epoch % 10 != 0 and epoch != epochs: continue
        words = model.H_1_words.detach().clone(); model.eval()
        with torch.inference_mode():
            _, z = model.dev_repr_logits(xval, words)
            prob = torch.softmax(z, 1)[:, 1].cpu().numpy()
        m = metric(vy, prob); key = (m["macro_f1"], m["depressed_f1"], m["auroc"])
        print(f"Student text {epoch}/{epochs}: val macroF1={m['macro_f1']:.4f}")
        if key > best:
            best = key; stale = 0
            torch.save({"weights": model.state_dict(), "word_state": words.cpu(),
                        "best_epoch": epoch}, output / "text_best.pt")
        else: stale += 1
        if stale >= 25: break
    state = torch.load(output / "text_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["weights"]); model.eval()
    for name, frame, x in (("train", train, xtrain), ("val", val, xval)):
        with torch.inference_mode():
            emb, z = model.dev_repr_logits(x, state["word_state"])
            prob = torch.softmax(z, 1)[:, 1].cpu().numpy()
        np.savez_compressed(output / f"{name}_text_embeddings.npz",
                            participant_ids=frame.participant_id.to_numpy(int),
                            labels=frame.label.to_numpy(int),
                            embedding=emb.cpu().numpy().astype(np.float32), probability=prob)
    with (output / "text_vectorizer.pkl").open("wb") as f: pickle.dump(vectorizer, f)
    return {"best_epoch": state["best_epoch"], "val": metric(vy, np.load(output / "val_text_embeddings.npz")["probability"])}


def train_audio(train, val, indexed, mean, std, output, seed, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Segments(train, indexed, mean, std, samples=4, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=gen, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    model = CompactAudioBranch().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    positive = torch.tensor(float(sum(train.label == 0) / sum(train.label == 1)), device=device)
    best = (-1., -1., -1.); stale = 0
    for epoch in range(1, epochs + 1):
        dataset.epoch = epoch; model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=positive)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        val_df, _ = evaluate(model, val, indexed, mean, std, device, 16)
        m = metric(val_df.label.to_numpy(int), val_df.audio_probability.to_numpy(float))
        key = (m["macro_f1"], m["depressed_f1"], m["auroc"])
        print(f"Student audio {epoch}/{epochs}: val macroF1={m['macro_f1']:.4f}")
        if key > best:
            best = key; stale = 0
            torch.save({"weights": model.state_dict(), "best_epoch": epoch}, output / "audio_best.pt")
        else: stale += 1
        if stale >= 8: break
    state = torch.load(output / "audio_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["weights"])
    for name, frame in (("train", train), ("val", val)):
        result, embedding = evaluate(model, frame, indexed, mean, std, device, 16)
        np.savez_compressed(output / f"{name}_audio_embeddings.npz",
                            participant_ids=result.participant_id.to_numpy(int),
                            labels=result.label.to_numpy(int), embedding=embedding,
                            probability=result.audio_probability.to_numpy(float))
    return {"best_epoch": state["best_epoch"], "val": metric(
        val.label.to_numpy(int), np.load(output / "val_audio_embeddings.npz")["probability"])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--audio-teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text-epochs", type=int, default=300)
    p.add_argument("--audio-epochs", type=int, default=30)
    a = p.parse_args(argv)
    full = verified_split(a.split_dir)
    indexed = rows_for(a.features, full)
    train = full.loc[full.split.eq("train")].sort_values("participant_id")
    val = full.loc[full.split.eq("val")].sort_values("participant_id")
    audio_audit = json.loads((a.audio_teacher / "audit.json").read_text())
    if audio_audit["signature"]["split_sha256"] != digest(a.split_dir / "manifest.csv"):
        raise ValueError("Audio teacher normalization from another split")
    with np.load(a.audio_teacher / "train_only_normalization.npz") as norm:
        mean, std = norm["mean"], norm["std"]
    signature = {"split_sha256": digest(a.split_dir / "manifest.csv"),
                 "feature_manifest_sha256": digest(a.features / "participant_manifest.csv"),
                 "student_seed": 1042, "text_epochs": a.text_epochs, "audio_epochs": a.audio_epochs,
                 "normalization_sha256": digest(a.audio_teacher / "train_only_normalization.npz")}
    if a.output.exists() and any(a.output.iterdir()):
        audit = a.output / "audit.json"
        if audit.is_file() and json.loads(audit.read_text()).get("signature") == signature:
            print("Verified existing student branches:", a.output); return
        raise ValueError("Use a fresh student-branch output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    random.seed(1042); np.random.seed(1042); torch.manual_seed(1042)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(1042)
    text_report = train_text(a.daic_root, train, val, a.output, 1042, a.text_epochs)
    audio_report = train_audio(train, val, indexed, mean, std, a.output, 1042, a.audio_epochs)
    (a.output / "audit.json").write_text(json.dumps({"signature": signature,
        "teacher_checkpoint_loaded": False, "teacher_features_loaded": False,
        "split_train": 132, "split_val": 28, "student_test_opened": False,
        "text": text_report, "audio": audio_report}, indent=2) + "\n")
    print("Saved independently trained student branches:", a.output)


if __name__ == "__main__": main()
