from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from . import AUTHOR_TRAIN_CROP_FRAMES
from .common import aggregate_segments,infer_segments,load_author_model,load_author_stats,load_feature_manifest,metric_dict,normalise_segments,save_json,segment_feature

RUN4_SEED=1300

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--author-root",required=True); p.add_argument("--output",required=True); p.add_argument("--batch-size",type=int,default=64)
    a=p.parse_args(argv); features,author,out=Path(a.features),Path(a.author_root),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    train=load_feature_manifest(features,"train"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,_,_=load_author_model(author,device); mean,std,_=load_author_stats(author); rng=random.Random(RUN4_SEED); rows=[]
    for r in train.itertuples(index=False):
        x=np.load(r.feature_path).astype(np.float32); width=x.shape[1]
        if width<AUTHOR_TRAIN_CROP_FRAMES: raise AssertionError(f"{r.participant_id}: {width} < {AUTHOR_TRAIN_CROP_FRAMES}")
        start=rng.randint(0,width-AUTHOR_TRAIN_CROP_FRAMES); crop=x[:,start:start+AUTHOR_TRAIN_CROP_FRAMES]
        seg=normalise_segments(segment_feature(crop),mean,std); pred=infer_segments(model,seg,device,a.batch_size)
        z=aggregate_segments(pred,r.participant_id,r.label); z.update({"crop_start":start,"crop_frames":AUTHOR_TRAIN_CROP_FRAMES,"run_seed":RUN4_SEED}); rows.append(z)
    df=pd.DataFrame(rows); m=metric_dict(df); df.to_csv(out/"train_run4_crop_kd_targets.csv",index=False)
    s={"protocol":"Author-style run #4 TRAIN crop view for KD export","seed":RUN4_SEED,"crop_frames":AUTHOR_TRAIN_CROP_FRAMES,"participants":len(df),"test_opened":False,"metrics":m}
    save_json(out/"train_run4_crop_audit.json",s); print(json.dumps(s,indent=2)); return s
if __name__=="__main__": main()
