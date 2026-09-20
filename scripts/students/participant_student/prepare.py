from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd,soundfile as sf
from scipy.signal import resample_poly
from tqdm.auto import tqdm
from . import SEGMENT_CONFIG
from .data import read_split,find_sources,read_turns,segment_turns,build_vocab,encode_tokens,logmel_summary,fit_standardizer,standardize,sha256_file

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--daic-root",required=True); p.add_argument("--output",required=True)
    a=p.parse_args(argv); root,out=Path(a.daic_root),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    train,tp=read_split(root,"train_split_Depression_AVEC2017.csv"); dev,dp=read_split(root,"dev_split_Depression_AVEC2017.csv")
    dev=dev.loc[dev.participant_id.ne(440)].reset_index(drop=True)
    assert (len(train),len(dev))==(107,34) and not set(train.participant_id)&set(dev.participant_id)
    manifest=pd.concat([train.assign(split="train"),dev.assign(split="dev")],ignore_index=True)
    sources=find_sources(root,manifest.participant_id); segs={}
    for r in tqdm(list(manifest.itertuples()),desc="Parse aligned participant turns",colour="green"):
        segs[int(r.participant_id)]=segment_turns(r.participant_id,read_turns(sources[int(r.participant_id)][1]),SEGMENT_CONFIG)
    vocab=build_vocab([segs[int(x)] for x in train.participant_id],SEGMENT_CONFIG["max_vocab"],SEGMENT_CONFIG["min_token_freq"])
    (out/"vocab.json").write_text(json.dumps(vocab,indent=2,ensure_ascii=False)+"\n")
    vhash=hashlib.sha256(json.dumps(vocab,sort_keys=True).encode()).hexdigest()
    signature=hashlib.sha256(json.dumps({"cfg":SEGMENT_CONFIG,"vocab_sha256":vhash},sort_keys=True).encode()).hexdigest()
    cache=out/"cache"; cache.mkdir(exist_ok=True)
    rows=[]; cached=0
    bar=tqdm(list(manifest.itertuples()),desc="Build/resume participant features",colour="green")
    for r in bar:
        pid=int(r.participant_id); cp=cache/r.split/f"{pid}.npz"; cp.parent.mkdir(parents=True,exist_ok=True); ss=segs[pid]
        valid=False
        if cp.exists():
            try:
                with np.load(cp,allow_pickle=False) as z: valid=str(z["signature"].item())==signature and len(z["segment_ids"])==len(ss)
            except Exception: valid=False
        if valid:
            cached+=1; bar.set_postfix(pid=pid,status="cached",cached=cached)
        else:
            ap,_=sources[pid]; af=[]; tx=[]; starts=[]; stops=[]; ids=[]
            bar.set_postfix(pid=pid,status="extract",segments=len(ss),cached=cached)
            with sf.SoundFile(ap) as wav:
                sr=int(wav.samplerate)
                for s in ss:
                    wav.seek(min(round(float(s["start"])*sr),len(wav)))
                    x=wav.read(max(1,round((float(s["stop"])-float(s["start"]))*sr)),dtype="float32",always_2d=True).mean(1)
                    assert len(x) and np.isfinite(x).all()
                    if sr!=SEGMENT_CONFIG["sample_rate"]:
                        g=math.gcd(sr,SEGMENT_CONFIG["sample_rate"])
                        x=resample_poly(x,SEGMENT_CONFIG["sample_rate"]//g,sr//g).astype(np.float32)
                    af.append(logmel_summary(x,SEGMENT_CONFIG["sample_rate"],SEGMENT_CONFIG["n_mels"]))
                    tx.append(encode_tokens(s["tokens"],vocab,SEGMENT_CONFIG["max_text_tokens"]))
                    starts.append(s["start"]); stops.append(s["stop"]); ids.append(s["segment_id"])
            with open(cp.with_suffix(".tmp"),"wb") as f:
                np.savez_compressed(f,audio_raw=np.asarray(af,np.float32),text_ids=np.asarray(tx,np.int64),
                    starts=np.asarray(starts,np.float32),stops=np.asarray(stops,np.float32),segment_ids=np.asarray(ids,dtype="U32"),signature=signature)
            cp.with_suffix(".tmp").replace(cp)
        rows.append({"participant_id":pid,"split":r.split,"label":int(r.label),"n_segments":len(ss),"cache":str(cp)})
    meta=pd.DataFrame(rows).sort_values(["split","participant_id"]); meta.to_csv(out/"participant_manifest.csv",index=False)
    tr_audio=[]
    for r in meta.loc[meta.split.eq("train")].itertuples():
        with np.load(r.cache,allow_pickle=False) as z: tr_audio.append(z["audio_raw"].astype(np.float32))
    mean,std=fit_standardizer(np.concatenate(tr_audio)); np.savez_compressed(out/"audio_standardizer.npz",mean=mean,std=std)
    maxs=SEGMENT_CONFIG["max_segments_per_participant"]; adim=2*SEGMENT_CONFIG["n_mels"]; l=SEGMENT_CONFIG["max_text_tokens"]
    def assemble(split):
        m=meta.loc[meta.split.eq(split)].reset_index(drop=True); n=len(m)
        audio=np.zeros((n,maxs,adim),np.float32); text=np.zeros((n,maxs,l),np.int64); mask=np.zeros((n,maxs),bool); ids=np.full((n,maxs),"",dtype="U32")
        for i,r in enumerate(m.itertuples()):
            with np.load(r.cache,allow_pickle=False) as z:
                x=standardize(z["audio_raw"],mean,std); t=z["text_ids"]; sid=z["segment_ids"].astype(str); k=len(x)
            audio[i,:k]=x; text[i,:k]=t; mask[i,:k]=True; ids[i,:k]=sid
        np.savez_compressed(out/f"{split}_features.npz",participant_ids=m.participant_id.to_numpy(np.int64),labels=m.label.to_numpy(np.int64),
            audio=audio,text=text,segment_mask=mask,segment_ids=ids,n_segments=mask.sum(1).astype(np.int64))
        return m
    assemble("train"); assemble("dev")
    summary={"protocol":"participant-level ReLiMP-Net student preparation","train_participants":107,"dev_participants":34,
        "participant_440_excluded":True,"test_prepared":False,"test_opened":False,"segment_config":SEGMENT_CONFIG,
        "vocab_size":len(vocab),"vocab_source":"TRAIN-107 only","audio_standardizer_source":"TRAIN-107 valid segments only",
        "train_split_sha256":sha256_file(tp),"dev_split_sha256":sha256_file(dp),
        "train_segments":int(meta.loc[meta.split.eq("train"),"n_segments"].sum()),"dev_segments":int(meta.loc[meta.split.eq("dev"),"n_segments"].sum())}
    (out/"feature_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2)); return summary
if __name__=="__main__": main()
