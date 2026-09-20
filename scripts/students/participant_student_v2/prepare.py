from __future__ import annotations
import argparse,json,pickle,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,f_classif
from tqdm.auto import tqdm

from . import SEGMENT_FRAMES,MAX_AUDIO_SEGMENTS,TEXT_FEATURES

def read_split(root,name):
    p=Path(root)/"metadata"/name
    d=pd.read_csv(p); d.columns=d.columns.str.strip().str.lower()
    assert {"participant_id","phq8_binary"}<=set(d.columns),p
    out=pd.DataFrame({"participant_id":pd.to_numeric(d.participant_id,errors="raise").astype(int),
                      "label":pd.to_numeric(d.phq8_binary,errors="raise").astype(int)})
    assert not out.participant_id.duplicated().any() and out.label.isin([0,1]).all()
    return out

def transcript_index(root,pids):
    want=set(map(int,pids)); found={}
    for p in Path(root).rglob("*_TRANSCRIPT.csv"):
        m=re.fullmatch(r"(\d+)_TRANSCRIPT\.csv",p.name,re.I)
        if m and int(m.group(1)) in want: found.setdefault(int(m.group(1)),[]).append(p)
    for pid in want: assert len(found.get(pid,[]))==1,f"{pid}: transcript coverage={len(found.get(pid,[]))}"
    return {pid:v[0] for pid,v in found.items()}

def participant_document(path):
    d=pd.read_csv(path,sep="\t"); d.columns=d.columns.str.strip().str.lower()
    if not {"speaker","value"}<=set(d.columns):
        d=pd.read_csv(path,sep=None,engine="python"); d.columns=d.columns.str.strip().str.lower()
    assert {"speaker","value"}<=set(d.columns),path
    d=d.loc[d.speaker.astype(str).str.strip().str.lower().eq("participant")].copy()
    v=d.value.fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
    v=v.loc[v.ne("") & ~v.str.contains("scrubbed|redacted",case=False,regex=True)]
    doc=" ".join(v.tolist()).strip(); assert doc,f"{path}: empty Participant document"; return doc

def compare_manifest(root):
    p=Path(root)/"participant_manifest.csv"; d=pd.read_csv(p)
    assert {"split","participant_id","label","feature_path","frames"}<=set(d.columns),p
    d=d.loc[d.split.astype(str).str.lower().isin(["train","dev"])].copy()
    d["split"]=d.split.astype(str).str.lower(); d["participant_id"]=d.participant_id.astype(int); d["label"]=d.label.astype(int)
    assert not d.duplicated(["split","participant_id"]).any()
    for r in d.itertuples():
        x=np.load(r.feature_path,mmap_mode="r")
        assert x.ndim==2 and x.shape[0]==130 and x.shape[1]>0 and int(r.frames)==x.shape[1],r.participant_id
    return d

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--daic-root",required=True); p.add_argument("--compare16-root",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(argv); root,comp,out=Path(a.daic_root),Path(a.compare16_root),Path(a.output); out.mkdir(parents=True,exist_ok=True)

    train=read_split(root,"train_split_Depression_AVEC2017.csv")
    dev=read_split(root,"dev_split_Depression_AVEC2017.csv"); dev=dev.loc[dev.participant_id.ne(440)].reset_index(drop=True)
    assert (len(train),len(dev))==(107,34) and not set(train.participant_id)&set(dev.participant_id)

    cm=compare_manifest(comp)
    trm=cm.loc[cm.split.eq("train")].merge(train,on=["participant_id","label"],validate="one_to_one").sort_values("participant_id")
    dvm=cm.loc[cm.split.eq("dev")].merge(dev,on=["participant_id","label"],validate="one_to_one").sort_values("participant_id")
    assert len(trm)==107 and len(dvm)==34 and 440 not in set(dvm.participant_id)

    # Student-owned TRAIN-only acoustic normalization over all cached ComParE16 frames.
    sum_x=np.zeros(130,np.float64); sum_x2=np.zeros(130,np.float64); count=0
    for r in tqdm(trm.itertuples(),total=len(trm),desc="Fit TRAIN-only ComParE16 normalization",colour="green"):
        x=np.load(r.feature_path,mmap_mode="r").astype(np.float64)
        sum_x+=x.sum(1); sum_x2+=(x*x).sum(1); count+=x.shape[1]
    mean=sum_x/count; var=np.maximum(sum_x2/count-mean*mean,1e-8); std=np.sqrt(var)
    np.savez_compressed(out/"audio_standardizer.npz",mean=mean.astype(np.float32),std=std.astype(np.float32),frames=np.int64(count))

    # Independent student text representation: same high-level top-k TF-IDF principle as the text teacher,
    # but fitted from scratch using TRAIN-107 only. No teacher vectorizer/checkpoint is consumed.
    tids=transcript_index(root,pd.concat([train.participant_id,dev.participant_id]))
    train_docs=[participant_document(tids[int(pid)]) for pid in train.sort_values("participant_id").participant_id]
    dev_docs=[participant_document(tids[int(pid)]) for pid in dev.sort_values("participant_id").participant_id]
    y=train.sort_values("participant_id").label.to_numpy(int)

    base=TfidfVectorizer(stop_words="english")
    x0=base.fit_transform(train_docs); k=min(TEXT_FEATURES,x0.shape[1])
    selector=SelectKBest(f_classif,k=k).fit(x0,y)
    terms=base.get_feature_names_out()[selector.get_support()].tolist()
    vectorizer=TfidfVectorizer(stop_words="english",vocabulary=terms)
    xtr=vectorizer.fit_transform(train_docs).toarray().astype(np.float32)
    xdv=vectorizer.transform(dev_docs).toarray().astype(np.float32)
    assert xtr.shape==(107,k) and xdv.shape==(34,k) and np.isfinite(xtr).all() and np.isfinite(xdv).all()
    np.save(out/"train_text_tfidf.npy",xtr); np.save(out/"dev_text_tfidf.npy",xdv)
    with (out/"text_vectorizer.pkl").open("wb") as f: pickle.dump(vectorizer,f)
    pd.DataFrame({"term":terms}).to_csv(out/"text_top250_terms.csv",index=False)

    trm[["participant_id","label","feature_path","frames"]].to_csv(out/"train_manifest.csv",index=False)
    dvm[["participant_id","label","feature_path","frames"]].to_csv(out/"dev_manifest.csv",index=False)

    def segstats(m):
        n=np.ceil(m.frames.to_numpy(float)/SEGMENT_FRAMES).astype(int)
        return {"min":int(n.min()),"median":float(np.median(n)),"max":int(n.max()),"mean":float(n.mean()),
                "capped_at_32":int(np.minimum(n,MAX_AUDIO_SEGMENTS).sum())}
    summary={
        "protocol":"rich compact participant student v2",
        "train_participants":107,"dev_participants":34,"participant_440_excluded":True,
        "test_opened":False,"test_prepared":False,
        "audio_source":"existing frozen USSD-compatible ComParE16 participant features; no teacher prediction/hidden state",
        "audio_feature_shape":"130 x variable_frames","audio_segment_frames":SEGMENT_FRAMES,"max_audio_segments_per_participant":MAX_AUDIO_SEGMENTS,
        "audio_normalization_source":"all TRAIN-107 ComParE16 frames only",
        "text_source":"Participant-only full interview transcript",
        "text_representation":"independent TRAIN-only TF-IDF + SelectKBest(f_classif), top-250; teacher vectorizer not used",
        "text_features":int(k),"train_audio_segment_stats":segstats(trm),"dev_audio_segment_stats":segstats(dvm),
        "compare16_manifest":str(comp/"participant_manifest.csv")
    }
    (out/"feature_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2)); print("TEST CLOSED. No raw audio extraction or openSMILE rerun was performed.")
    return summary

if __name__=="__main__": main()
