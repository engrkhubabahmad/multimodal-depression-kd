"""Train fresh Idiap participant InducT-GCN on TRAIN-113; evaluate VAL-37."""
from __future__ import annotations
import argparse
import json
import pickle
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from scripts.teachers.idiap_text.export_targets import load_author
from scripts.students.participant_student_v3.pretrain_text import select_vectorizer, transcript_map, participant_doc
from .split import digest


def verified_split(split_dir, coverage_dir):
    marker = json.loads((split_dir / 'complete.json').read_text())
    if digest(split_dir / 'manifest.csv') != marker['manifest_sha256']:
        raise ValueError('Split manifest changed')
    manifest = pd.read_csv(split_dir / 'manifest.csv')
    if manifest.split.value_counts().to_dict() != {'train': 113, 'val': 37, 'student_test': 38}:
        raise ValueError('Unexpected split counts')
    covered = pd.read_csv(coverage_dir / 'participant_manifest.csv')
    expected = manifest.loc[manifest.split.isin(['train', 'val'])].sort_values('participant_id')
    covered = covered.sort_values('participant_id')
    if not covered[['participant_id', 'label', 'split']].reset_index(drop=True).equals(
            expected[['participant_id', 'label', 'split']].reset_index(drop=True)):
        raise ValueError('TRAIN/VAL coverage differs from split')
    audit = json.loads((coverage_dir / 'audit.json').read_text())
    if audit['split_manifest_sha256'] != digest(split_dir / 'manifest.csv'):
        raise ValueError('Coverage belongs to a different split')
    return manifest


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
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "depressed_f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(y, pred, labels=[0, 1],
                    target_names=["Non-depressed", "Depressed"], output_dict=True, zero_division=0)}


def initialize_from_published(model, new_vectorizer, checkpoint_path, vectorizer_path):
    """Copy compatible learned layers, remapping input columns by vocabulary token."""
    with vectorizer_path.open('rb') as f: old_vectorizer = pickle.load(f)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if list(checkpoint['classes_']) != ['negative', 'positive']:
        raise ValueError('Published class ordering differs')
    weights = checkpoint['model_state_dict']
    source_input = weights['get_H_1.0.weight']
    source_output = weights['node_emb2out.weight']
    input_layer = model.get_H_1[0].weight
    output_layer = model.node_emb2out.weight
    if (source_input.shape[0] != input_layer.shape[0] or
            source_output.shape != output_layer.shape or
            source_input.shape[1] != len(old_vectorizer.vocabulary_)):
        raise ValueError('Published checkpoint architecture incompatible with new graph')
    shared = sorted(set(old_vectorizer.vocabulary_) & set(new_vectorizer.vocabulary_))
    if not shared: raise ValueError('No shared vocabulary tokens for weight transfer')
    with torch.no_grad():
        for token in shared:
            input_layer[:, new_vectorizer.vocabulary_[token]].copy_(
                source_input[:, old_vectorizer.vocabulary_[token]])
        output_layer.copy_(source_output)
    return {'shared_vocabulary_tokens': len(shared),
            'new_vocabulary_tokens': len(new_vectorizer.vocabulary_),
            'copied_layers': ['get_H_1.0.weight (shared token columns)', 'node_emb2out.weight'],
            'checkpoint_sha256': digest(checkpoint_path),
            'published_vectorizer_sha256': digest(vectorizer_path)}


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--daic-root", type=Path, required=True)
    a.add_argument("--split-dir", type=Path, required=True)
    a.add_argument("--coverage-dir", type=Path, required=True)
    a.add_argument("--idiap-source", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument('--published-checkpoint', type=Path)
    a.add_argument('--published-vectorizer', type=Path)
    a.add_argument("--seed", type=int, default=103)
    a.add_argument("--epochs", type=int, default=300)
    a.add_argument("--eval-every", type=int, default=1)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--patience-evals", type=int, default=25)
    args = a.parse_args(argv)
    if bool(args.published_checkpoint) != bool(args.published_vectorizer):
        raise ValueError('Published checkpoint and vectorizer must both be supplied')
    if args.seed not in (42, 103) or args.epochs < 1 or args.eval_every != 1: raise ValueError("This protocol evaluates every epoch")
    manifest = verified_split(args.split_dir, args.coverage_dir)
    tr = manifest.loc[manifest.split.eq("train")].sort_values("participant_id")
    va = manifest.loc[manifest.split.eq("val")].sort_values("participant_id")
    pathmap = transcript_map(args.daic_root, pd.concat([tr.participant_id, va.participant_id]))
    covered_paths = pd.read_csv(args.coverage_dir / 'participant_manifest.csv').set_index('participant_id')
    for pid, path in pathmap.items():
        if digest(path) != digest(covered_paths.loc[pid, 'transcript_path']):
            raise ValueError(f'{pid}: transcript changed since coverage audit')
    docs = {int(pid): participant_doc(pathmap[int(pid)]) for pid in pd.concat([tr.participant_id, va.participant_id])}
    train_docs = [docs[int(pid)] for pid in tr.participant_id]
    val_docs = [docs[int(pid)] for pid in va.participant_id]
    y = tr.label.to_numpy(int); vy = va.label.to_numpy(int)
    signature = {"seed": args.seed, "epochs": args.epochs, "eval_every": args.eval_every,
                 "lr": args.lr, "patience_evals": args.patience_evals,
                 "initialization": 'published_transfer' if args.published_checkpoint else 'fresh',
                 "checkpoint_sha256": digest(args.published_checkpoint) if args.published_checkpoint else None,
                 "published_vectorizer_sha256": digest(args.published_vectorizer) if args.published_vectorizer else None,
                 "use_pagerank": False,
                 "split_sha256": digest(args.split_dir / "manifest.csv"),
                 "coverage_sha256": digest(args.coverage_dir / 'participant_manifest.csv'),
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
    # The author assigns this global only inside its CLI entry point. Importing
    # the class directly leaves build_graph() without a value for it.
    author.USE_PAGERANK = False
    if not all(hasattr(author, flag) for flag in ("USE_WEIGHTED_PMI", "USE_P_TOK2TOK_WEIGHT", "ACTIVATION_FUNC")):
        raise RuntimeError("The Idiap author source is missing required graph settings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    author.DEVICE = device
    model = author.InducTGCN(64, np.array(["negative", "positive"]), .5, vectorizer)
    model.build_graph(train_docs, window_size=3, verbose=True)
    transfer = (initialize_from_published(model, vectorizer, args.published_checkpoint,
                                         args.published_vectorizer) if args.published_checkpoint else None)
    if transfer: print('Published weight transfer:', transfer)
    model.to(device)
    weight = torch.tensor([len(y) / (2 * sum(y == 0)), len(y) / (2 * sum(y == 1))],
                          dtype=torch.float32, device=device)
    target = torch.tensor(y, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = (-1., -1., -1.); stale = 0; history = []
    progress = tqdm(range(1, args.epochs + 1), desc="Text teacher epochs", unit="epoch")
    for epoch in progress:
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = model.cross_entropy_loss_on_document_nodes(target, class_weight=weight)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            h = model.get_H_1(model.Conv_0.to(device))
            train_logits = model.node_emb2out(model.A_norm.to(device) @ h)[len(vectorizer.vocabulary_):]
            train_prob_epoch = torch.softmax(train_logits, dim=1)[:, 1].cpu().numpy()
            words = h[:len(vectorizer.vocabulary_)].detach().clone().cpu()
        _, prob = inductive(model, vectorizer, val_docs, words)
        tm = metric(y, train_prob_epoch)
        m = metric(vy, prob); key = (m["macro_f1"], m["depressed_f1"], m["auroc"])
        progress.set_postfix(train_f1=f"{tm['macro_f1']:.3f}", val_f1=f"{m['macro_f1']:.3f}")
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu()),
                        "train_macro_f1": tm["macro_f1"], "train_depressed_f1": tm["depressed_f1"],
                        "val_macro_f1": m["macro_f1"], "val_depressed_f1": m["depressed_f1"],
                        "val_auroc": m["auroc"]})
        print(f"Text epoch {epoch}/{args.epochs} train macroF1={tm['macro_f1']:.4f} val macroF1={m['macro_f1']:.4f} val depressedF1={m['depressed_f1']:.4f}")
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
    (args.output / "audit.json").write_text(json.dumps({"signature": signature, "architecture": "Idiap InducT-GCN with rebuilt TRAIN graph",
       "initialization": transfer or 'random',
       "published_checkpoint_prior_exposure": {
           "original_train_ids_in_val": int(sum(va.source_split.eq('canonical_train'))),
           "original_train_ids_in_internal_test": int(sum(manifest.loc[manifest.split.eq('student_test'),
                                                      'source_split'].eq('canonical_train'))),
           "clean_holdout_claim_valid": False if transfer else True},
       "graph_settings": {"use_pagerank": False, "window_size": 3, "vocabulary_size": 250},
       "train_participants": 113, "val_participants": 37, "student_test_opened": False,
       "oof": False, "best_epoch": state["best_epoch"],
       "train": metric(y, train_prob), "val": metric(vy, val_prob)}, indent=2) + "\n")
    print("Saved fresh text teacher:", args.output)


if __name__ == "__main__": main()
