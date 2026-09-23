"""Fresh Idiap InducT-GCN architecture on the full-188 TRAIN-132 split."""
from __future__ import annotations
import argparse
import json
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from scripts.teachers.idiap_text.export_targets import load_author
from scripts.students.participant_student_v3.pretrain_text import select_vectorizer, transcript_map, participant_doc
from .features import verified_split
from .split import digest


def inductive(model, vectorizer, docs, word_state):
    """Author eval algebra, with fixed checkpoint word nodes."""
    x = torch.tensor(vectorizer.transform(docs).toarray(), dtype=torch.float32)
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.inference_mode():
        h = model.get_H_1(2 * x)
        representation = x @ word_state.to(device) + h
        logits = model.node_emb2out(representation)
        prob = torch.softmax(logits, dim=1)[:, 1]
    return representation.cpu().numpy().astype(np.float32), prob.cpu().numpy()


def metric(y, p):
    pred = (p >= .5).astype(int)
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "depressed_f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--daic-root", type=Path, required=True)
    a.add_argument("--split-dir", type=Path, required=True)
    a.add_argument("--idiap-source", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--epochs", type=int, default=300)
    a.add_argument("--eval-every", type=int, default=10)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--patience-evals", type=int, default=25)
    args = a.parse_args(argv)
    if args.seed != 42 or args.epochs < 1 or args.eval_every < 1: raise ValueError("Invalid fixed protocol settings")
    manifest = verified_split(args.split_dir)
    tr = manifest.loc[manifest.split.eq("train")].sort_values("participant_id")
    va = manifest.loc[manifest.split.eq("val")].sort_values("participant_id")
    pathmap = transcript_map(args.daic_root, pd.concat([tr.participant_id, va.participant_id]))
    docs = {int(pid): participant_doc(pathmap[int(pid)]) for pid in pd.concat([tr.participant_id, va.participant_id])}
    train_docs = [docs[int(pid)] for pid in tr.participant_id]
    val_docs = [docs[int(pid)] for pid in va.participant_id]
    y = tr.label.to_numpy(int); vy = va.label.to_numpy(int)
    signature = {"seed": 42, "epochs": args.epochs, "eval_every": args.eval_every,
                 "lr": args.lr, "patience_evals": args.patience_evals,
                 "split_sha256": digest(args.split_dir / "manifest.csv"),
                 "author_code_sha256": digest(args.idiap_source / "main.py"),
                 "source_transcripts_sha256": {str(p): digest(p) for p in pathmap.values()}}
    if args.output.exists() and any(args.output.iterdir()):
        audit = args.output / "audit.json"
        if audit.is_file() and json.loads(audit.read_text()).get("signature") == signature:
            print("Verified existing text teacher:", args.output); return
        raise ValueError("Use a fresh text-teacher output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    vectorizer = select_vectorizer(train_docs, y, 250)
    author = load_author(args.idiap_source / "main.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    author.DEVICE = device
    model = author.InducTGCN(64, np.array(["negative", "positive"]), .5, vectorizer)
    model.build_graph(train_docs, window_size=3, verbose=True)
    model.to(device)
    weight = torch.tensor([len(y) / (2 * sum(y == 0)), len(y) / (2 * sum(y == 1))],
                          dtype=torch.float32, device=device)
    target = torch.tensor(y, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = (-1., -1., -1.); stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = model.cross_entropy_loss_on_document_nodes(target, class_weight=weight)
        loss.backward(); optimizer.step()
        if epoch % args.eval_every != 0 and epoch != args.epochs: continue
        words = model.H_1_words.detach().clone()
        model.eval()
        _, prob = inductive(model, vectorizer, val_docs, words)
        m = metric(vy, prob); key = (m["macro_f1"], m["depressed_f1"], m["auroc"])
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()), **m})
        print(f"Text epoch {epoch}/{args.epochs} val macroF1={m['macro_f1']:.4f} depressedF1={m['depressed_f1']:.4f}")
        if key > best:
            best = key; stale = 0
            torch.save({"model_state_dict": model.state_dict(), "word_state": words.cpu(),
                        "best_epoch": epoch, "signature": signature}, args.output / "best.pt")
        else: stale += 1
        if stale >= args.patience_evals: break
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    state = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(dict(state["model_state_dict"])); model.eval()
    words = state["word_state"].to(device)
    with torch.inference_mode():
        h = model.get_H_1(model.Conv_0.to(device))
        rep = (model.A_norm.to(device) @ h)[250:].cpu().numpy().astype(np.float32)
        logits = model.node_emb2out(torch.tensor(rep, device=device)).cpu().numpy()
        train_prob = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    val_rep, val_prob = inductive(model, vectorizer, val_docs, words)
    for name, frame, embedding, prob in (("train", tr, rep, train_prob), ("val", va, val_rep, val_prob)):
        p = np.clip(prob.astype(float), 1e-6, 1 - 1e-6); logit = np.log(p / (1 - p))
        pd.DataFrame({"participant_id": frame.participant_id.to_numpy(int),
                      "label": frame.label.to_numpy(int), "text_probability": p,
                      "text_logit": logit}).to_csv(args.output / f"{name}_text_targets.csv", index=False)
        np.savez_compressed(args.output / f"{name}_text_embeddings.npz",
                            participant_ids=frame.participant_id.to_numpy(int), labels=frame.label.to_numpy(int),
                            embedding=embedding, probability=p)
    with (args.output / "vectorizer.pkl").open("wb") as f: pickle.dump(vectorizer, f)
    (args.output / "audit.json").write_text(json.dumps({"signature": signature, "architecture": "fresh Idiap InducT-GCN",
       "train_participants": 132, "val_participants": 28, "student_test_opened": False,
       "oof": False, "best_epoch": state["best_epoch"],
       "train": metric(y, train_prob), "val": metric(vy, val_prob)}, indent=2) + "\n")
    print("Saved fresh text teacher:", args.output)


if __name__ == "__main__": main()
