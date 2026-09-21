from __future__ import annotations
import argparse,json,pickle,shutil,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm

from scripts.teachers.ussd_audio.common import (
    compare16_matrix,extract_participant_wav,load_author_stats,sha256
)
from .audio_model import CompactAudioBranch,parameter_count as audio_parameter_count
from .pretrain_audio import participant_segments
from .pretrain_text import InductText,one,participant_doc,transcript_map
from .fusion_model import FrozenBranchFusion,parameter_count as fusion_parameter_count

EXPECTED_AUDIO_PARAMS=182529
EXPECTED_FUSION_PARAMS=55617
EXPECTED_FULL_PARAMS=254274

def test_ids(root:Path):
    p=one(root,"test_split_Depression_AVEC2017.csv")
    d=pd.read_csv(p); c={x.casefold():x for x in d.columns}
    if "participant_id" not in c: raise ValueError(f"{p}: missing participant_ID")
    ids=np.sort(d[c["participant_id"]].astype(int).unique())
    if len(ids)!=47: raise AssertionError(f"Expected TEST=47 identifiers; got {len(ids)}")
    return ids,p

def resolve_exec(v):
    p=shutil.which(v) if "/" not in v else v
    if not p or not Path(p).exists(): raise FileNotFoundError(f"SMILExtract not found: {v}")
    return str(p)

def load_ids_npz(path):
    with np.load(path) as z: return set(z["participant_ids"].astype(int).tolist())

def infer_audio(model,feat,mean,std,device,batch=128):
    logits=[]; embs=[]; buf=[]
    model.eval()
    with torch.inference_mode():
        for seg in participant_segments(feat,mean,std):
            buf.append(seg)
            if len(buf)==batch:
                z,e=model(torch.from_numpy(np.stack(buf)).to(device)); logits.append(z.cpu().numpy()); embs.append(e.cpu().numpy()); buf=[]
        if buf:
            z,e=model(torch.from_numpy(np.stack(buf)).to(device)); logits.append(z.cpu().numpy()); embs.append(e.cpu().numpy())
    l=np.concatenate(logits); e=np.concatenate(embs); p=1/(1+np.exp(-l))
    return {
      "embedding":np.concatenate([e.mean(0),e.std(0)]).astype(np.float32),
      "mean_probability":float(p.mean()),
      "vote_fraction":float((p>=.5).mean()),
      "n_segments":int(len(p))
    }

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--daic-root",required=True)
    ap.add_argument("--text-branch",required=True)
    ap.add_argument("--audio-branch",required=True)
    ap.add_argument("--no-kd-reference",required=True)
    ap.add_argument("--standard-kd-reference",required=True)
    ap.add_argument("--dev-freeze",required=True)
    ap.add_argument("--author-root",required=True)
    ap.add_argument("--smile-extract",required=True)
    ap.add_argument("--compare16-config",required=True)
    ap.add_argument("--audio-cache",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--audio-batch-size",type=int,default=128)
    a=ap.parse_args(argv)

    root,td,ad=Path(a.daic_root),Path(a.text_branch),Path(a.audio_branch)
    nk,std,freeze=Path(a.no_kd_reference),Path(a.standard_kd_reference),Path(a.dev_freeze)
    author,cache,out=Path(a.author_root),Path(a.audio_cache),Path(a.output)
    out.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)

    if (out/"FINAL_TEST_METRICS.json").exists():
        raise RuntimeError("Final TEST metrics already exist. Do not regenerate blind predictions after scoring.")

    fr=json.loads((freeze/"final_dev_selection.json").read_text())
    sm=json.loads((std/"metrics.json").read_text())
    nm=json.loads((nk/"metrics.json").read_text())
    tm=json.loads((td/"metrics.json").read_text())
    am=json.loads((ad/"metrics.json").read_text())
    assert fr["protocol_status"]=="DEV stage frozen; TEST unopened"
    assert fr["selected_multimodal_condition"]=="standard_kd"
    assert fr["test_opened"] is False and fr["no_more_dev_tuning_after_freeze"] is True
    for d in [sm,nm,tm,am]: assert d["protocol"]["test_opened"] is False
    assert sm["protocol"]["threshold"]==0.5 and sm["protocol"]["threshold_search"] is False
    assert sm["protocol"]["best_epoch"]==2
    assert sm["protocol"]["full_student_params"]==EXPECTED_FULL_PARAMS

    ids,id_file=test_ids(root)
    train_ids=load_ids_npz(ad/"train_audio_embeddings.npz")
    dev_ids=load_ids_npz(ad/"dev_audio_embeddings.npz")
    assert not set(ids.tolist()) & train_ids
    assert not set(ids.tolist()) & dev_ids

    # Reconstruct TEST text embeddings from the already-frozen v3 text state.
    tmaps=transcript_map(root,ids)
    docs=[participant_doc(tmaps[int(pid)]) for pid in ids]
    assert all(docs)
    with (td/"vectorizer.pkl").open("rb") as f: vectorizer=pickle.load(f)
    X=torch.tensor(vectorizer.transform(docs).toarray(),dtype=torch.float32)
    tstate=torch.load(td/"inference_state.pt",map_location="cpu",weights_only=False)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model=InductText(len(vectorizer.vocabulary_)).to(device)
    text_model.load_state_dict(tstate["model_state_dict"]); text_model.eval()
    Hwords=tstate["H1_words"].to(device)
    with torch.inference_mode():
        Rt,_=text_model.dev_repr_logits(X.to(device),Hwords)
        text_emb=Rt.cpu().numpy().astype(np.float32)
    assert text_emb.shape==(47,64)
    np.savez_compressed(out/"blind_test_text_embeddings.npz",
                        participant_ids=ids,embedding=text_emb)

    # Prepare exact USSD-compatible TEST ComParE16 features. No TEST labels are read.
    smile=resolve_exec(a.smile_extract); config=Path(a.compare16_config)
    if not config.exists(): raise FileNotFoundError(config)
    feat_dir=cache/"participant_features"/"test"; csv_dir=cache/"compare16_csv"/"test"; wav_dir=cache/"patient_wav"/"test"
    feat_dir.mkdir(parents=True,exist_ok=True); csv_dir.mkdir(parents=True,exist_ok=True); wav_dir.mkdir(parents=True,exist_ok=True)
    feat_paths=[]; prep_rows=[]; reused=0
    bar=tqdm(ids,desc="FINAL TEST blind ComParE16",colour="green")
    for pid0 in bar:
        pid=int(pid0); feat=feat_dir/f"{pid}.npy"; csv=csv_dir/f"{pid}_P_audio_data.csv"; wav=wav_dir/f"{pid}_P_audio_data.wav"
        ok=False
        if feat.exists():
            try:
                x=np.load(feat,mmap_mode="r"); ok=x.ndim==2 and x.shape[0]==130 and x.shape[1]>0 and np.isfinite(x).all()
            except Exception: ok=False
        if ok:
            reused+=1
        else:
            raw=one(root,f"{pid}_AUDIO.wav"); tr=tmaps[pid]
            meta=extract_participant_wav(raw,tr,wav)
            subprocess.run([smile,"-C",str(config),"-I",str(wav),"-D",str(csv)],
                           stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,check=True)
            x=compare16_matrix(csv); np.save(feat,x); wav.unlink(missing_ok=True)
            prep_rows.append({"participant_id":pid,"frames":int(x.shape[1]),**meta})
        feat_paths.append(feat); bar.set_postfix(reused=reused,new=len(feat_paths)-reused)

    h=subprocess.run([smile,"-h"],capture_output=True,text=True)
    banner=(h.stdout or h.stderr).splitlines()
    prep={
      "mode":"blind final TEST feature preparation; no labels loaded",
      "test_participants":47,"identifier_source":str(id_file),
      "compare16_config":str(config),"compare16_config_sha256":sha256(config),
      "smile_extract":smile,"smile_banner":banner[0] if banner else "unknown",
      "reused_features":reused,"new_features":47-reused,
      "test_labels_loaded":False
    }
    (cache/"preprocessing_test_blind.json").write_text(json.dumps(prep,indent=2)+"\n")

    # Frozen compressed audio branch.
    mean,stdv,stats_path=load_author_stats(author)
    audio_model=CompactAudioBranch().to(device)
    ast=torch.load(ad/"best.pt",map_location=device,weights_only=False)
    audio_model.load_state_dict(ast["model_state_dict"]); audio_model.eval()
    assert audio_parameter_count(audio_model)==EXPECTED_AUDIO_PARAMS
    audio_emb=[]
    for pid,feat in tqdm(list(zip(ids,feat_paths)),desc="FINAL TEST frozen audio branch",colour="green"):
        z=infer_audio(audio_model,feat,mean,stdv,device,a.audio_batch_size)
        audio_emb.append(z["embedding"])
    audio_emb=np.stack(audio_emb).astype(np.float32)
    assert audio_emb.shape==(47,256)
    np.savez_compressed(out/"blind_test_audio_embeddings.npz",
                        participant_ids=ids,embedding=audio_emb)

    # Frozen Standard-KD fusion using exactly the No-KD TRAIN standardizers.
    sc=np.load(nk/"train_only_standardizers.npz")
    A=((audio_emb-sc["audio_mean"])/sc["audio_std"]).astype(np.float32)
    T=((text_emb-sc["text_mean"])/sc["text_std"]).astype(np.float32)
    assert np.isfinite(A).all() and np.isfinite(T).all()
    bc=json.loads((nk/"backbone_config.json").read_text())
    fusion=FrozenBranchFusion(dropout=float(bc["dropout"])).to(device)
    fst=torch.load(std/"best.pt",map_location=device,weights_only=False)
    fusion.load_state_dict(fst["model_state_dict"]); fusion.eval()
    assert fusion_parameter_count(fusion)==EXPECTED_FUSION_PARAMS
    with torch.inference_mode():
        pf=torch.sigmoid(fusion(torch.from_numpy(A).to(device),torch.from_numpy(T).to(device))).cpu().numpy()

    blind=pd.DataFrame({
      "participant_id":ids,
      "student_probability":pf,
      "student_prediction":(pf>=.5).astype(int)
    })
    blind.to_csv(out/"blind_test_predictions.csv",index=False)

    manifest={
      "status":"BLIND TEST PREDICTIONS FROZEN; labels never loaded",
      "test_participants":47,
      "selected_condition":"standard_kd_student",
      "selected_threshold":0.5,
      "selected_standard_kd_best_epoch":int(sm["protocol"]["best_epoch"]),
      "selected_checkpoint_sha256":sha256(std/"best.pt"),
      "audio_branch_checkpoint_sha256":sha256(ad/"best.pt"),
      "text_inference_state_sha256":sha256(td/"inference_state.pt"),
      "train_standardizers_sha256":sha256(nk/"train_only_standardizers.npz"),
      "dev_freeze_sha256":sha256(freeze/"final_dev_selection.json"),
      "author_normalization_sha256":sha256(stats_path),
      "compare16_config_sha256":sha256(config),
      "teacher_targets_used_on_test":False,
      "test_metrics_scope":"selected frozen student only",
      "unimodal_branch_test_metrics_permitted":False,
      "test_labels_loaded":False,
      "fitting_on_test":False,
      "threshold_search_on_test":False
    }
    (out/"blind_test_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
    print("\nBLIND TEST INFERENCE: PASS")
    print("Selected STUDENT predictions frozen. TEST LABELS HAVE NOT BEEN LOADED.")
    print("Next and only next: run score_final_test_once.py.")

if __name__=="__main__": main()
