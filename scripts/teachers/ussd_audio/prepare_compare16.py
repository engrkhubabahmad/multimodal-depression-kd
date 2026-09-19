from __future__ import annotations

import argparse,json,shutil,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from tqdm.auto import tqdm
from .common import compare16_matrix,extract_participant_wav,one,resolve_protocol_splits,save_json,sha256

def resolve_executable(v):
    p=shutil.which(v) if "/" not in v else v
    if not p or not Path(p).exists(): raise FileNotFoundError(f"SMILExtract not found: {v}")
    return str(p)

def cached_feature(csv_path,feat_path):
    try:
        if feat_path.exists():
            x=np.load(feat_path)
            if x.ndim==2 and x.shape[0]==130 and x.shape[1]>0 and np.isfinite(x).all(): return x
        if csv_path.exists():
            x=compare16_matrix(csv_path)
            if x.shape[0]==130 and x.shape[1]>0 and np.isfinite(x).all():
                np.save(feat_path,x); return x
    except Exception: pass
    return None

def process_split(root,frame,split,out,smile,config,keep_wav):
    rows=[]; reused=0
    bar=tqdm(frame.itertuples(index=False),total=len(frame),desc=f"USSD {split} ComParE16",colour="green")
    for row in bar:
        pid,label=int(row.participant_id),int(row.label)
        raw=one(root,f"{pid}_AUDIO.wav"); tr=one(root,f"{pid}_TRANSCRIPT.csv")
        wav=out/"patient_wav"/split/f"{pid}_P_audio_data.wav"; csv=out/"compare16_csv"/split/f"{pid}_P_audio_data.csv"; feat=out/"participant_features"/split/f"{pid}.npy"
        csv.parent.mkdir(parents=True,exist_ok=True); feat.parent.mkdir(parents=True,exist_ok=True)
        x=cached_feature(csv,feat); meta={"sample_rate":None,"samples":None,"seconds":None,"wav_bridge":"reused cached ComParE16"}
        if x is not None: reused+=1
        else:
            csv.unlink(missing_ok=True); feat.unlink(missing_ok=True)
            meta=extract_participant_wav(raw,tr,wav)
            subprocess.run([smile,"-C",str(config),"-I",str(wav),"-D",str(csv)],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,check=True)
            x=compare16_matrix(csv); np.save(feat,x)
            if not keep_wav: wav.unlink(missing_ok=True)
        rows.append({"split":split,"participant_id":pid,"label":label,"feature_path":str(feat),"compare16_csv":str(csv),"frames":int(x.shape[1]),**meta,
                     "raw_audio_sha256":sha256(raw),"transcript_sha256":sha256(tr),"patient_wav_sha256":sha256(wav) if wav.exists() else None,
                     "compare16_csv_sha256":sha256(csv),"reused_cache":x is not None and meta["wav_bridge"]=="reused cached ComParE16"})
        bar.set_postfix(reused=reused,new=len(rows)-reused)
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--daic-root",required=True); p.add_argument("--output",required=True); p.add_argument("--split",choices=["dev","train","both"],default="dev")
    p.add_argument("--smile-extract",default="SMILExtract"); p.add_argument("--compare16-config",required=True); p.add_argument("--keep-wav",action="store_true")
    a=p.parse_args(argv); root,out,config=Path(a.daic_root),Path(a.output),Path(a.compare16_config)
    if not config.exists(): raise FileNotFoundError(config)
    smile=resolve_executable(a.smile_extract); train,dev=resolve_protocol_splits(root); rows=[]
    if a.split in {"train","both"}: rows+=process_split(root,train,"train",out,smile,config,a.keep_wav)
    if a.split in {"dev","both"}: rows+=process_split(root,dev,"dev",out,smile,config,a.keep_wav)
    cur=out/"participant_manifest.csv"; new=pd.DataFrame(rows)
    if cur.exists():
        old=pd.read_csv(cur); keys=new.set_index(["split","participant_id"]).index
        new=pd.concat([old[~old.set_index(["split","participant_id"]).index.isin(keys)],new],ignore_index=True)
    new.sort_values(["split","participant_id"]).to_csv(cur,index=False)
    h=subprocess.run([smile,"-h"],capture_output=True,text=True); banner=(h.stdout or h.stderr).splitlines()
    summary={"prepared":len(rows),"reused":sum(bool(r["reused_cache"]) for r in rows),"splits":sorted({r["split"] for r in rows}),"manifest":str(cur),
             "compare16_config":str(config),"compare16_config_sha256":sha256(config),"smile_extract":smile,"smile_banner":banner[0] if banner else "unknown",
             "label_source":"local AVEC split CSV; no author-side relabel during adaptation","test_opened":False}
    for s in summary["splits"]: save_json(out/f"preprocessing_{s}.json",{**summary,"splits":[s],"prepared":sum(r["split"]==s for r in rows)})
    print(json.dumps(summary,indent=2)); return summary

if __name__=="__main__": main()
