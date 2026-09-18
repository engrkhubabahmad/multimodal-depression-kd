"""Pure NumPy/Pandas guards and windows for cached segment teachers."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)); tmp.replace(path)


def validate_manifest(meta,train,dev,test_ids):
    required={'segment_id','participant_id','split','label','start','stop','source_turn','part','parts'}
    if not required<=set(meta): raise ValueError('Incomplete segment manifest')
    if meta.segment_id.duplicated().any() or set(meta.split)!={'train','dev'}: raise ValueError('Invalid segments/splits')
    tr=set(train.participant_id); dv=set(dev.participant_id); te=set(test_ids)
    if tr&dv or tr&te or dv&te: raise ValueError('Participant leakage')
    for split,official in [('train',train),('dev',dev)]:
        part=meta.loc[meta.split.eq(split)]
        if set(part.participant_id)!=set(official.participant_id): raise ValueError(f'{split}: participant coverage mismatch')
        labels=official.set_index('participant_id').label
        if not np.array_equal(part.label.to_numpy(),part.participant_id.map(labels).to_numpy()):
            raise ValueError(f'{split}: label mismatch')
    if not np.isfinite(meta[['start','stop']].to_numpy()).all() or (meta.stop<=meta.start).any():
        raise ValueError('Invalid segment times')


def participant_weights(meta):
    if meta.groupby('participant_id').label.nunique().max()!=1: raise ValueError('Inconsistent participant labels')
    people=meta[['participant_id','label']].drop_duplicates()
    counts=people.label.value_counts(); sizes=meta.participant_id.value_counts()
    if set(counts.index)!={0,1}: raise ValueError('Both classes required')
    w=np.array([1/(counts[y]*sizes[p]) for p,y in zip(meta.participant_id,meta.label)],np.float32)
    return w/w.mean()


def adjacent(a,b,max_gap):
    gap=float(b.start-a.stop)
    if gap < -0.05 or gap>max_gap: return False
    if b.source_turn==a.source_turn: return b.part==a.part+1 and b.parts==a.parts
    return b.source_turn==a.source_turn+1 and a.part==a.parts-1 and b.part==0


def window_indices(meta,window=5,max_gap=30.0):
    """Centered, right-padded windows; never bridge a discarded segment/turn."""
    if window<1 or window%2!=1 or max_gap<0: raise ValueError('Use an odd positive window and nonnegative gap')
    meta=meta.reset_index(drop=True); result=np.full((len(meta),window),-1,np.int64); centers=np.zeros(len(meta),np.int64)
    radius=window//2
    for _,group in meta.groupby(['split','participant_id'],sort=False):
        ordered=group.sort_values(['start','stop','source_turn','part']); ids=ordered.index.to_numpy(); rows=list(ordered.itertuples())
        links=[adjacent(rows[j],rows[j+1],max_gap) for j in range(len(rows)-1)]
        for j,idx in enumerate(ids):
            left=right=j
            while left>max(0,j-radius) and links[left-1]: left-=1
            while right<min(len(ids)-1,j+radius) and links[right]: right+=1
            seq=ids[left:right+1]; result[idx,:len(seq)]=seq; centers[idx]=j-left
    return result,centers


def materialize_windows(x,indices):
    mask=indices>=0; values=x[np.maximum(indices,0)].copy(); values[~mask]=0
    return values,mask


def align_predictions(path,meta):
    d=pd.read_csv(path); keys=['segment_id','participant_id','split','label']
    if d.segment_id.duplicated().any() or len(d)!=len(meta): raise ValueError(f'{path}: prediction coverage')
    d=meta[keys].merge(d[keys+['probability']],on=keys,how='left',validate='one_to_one')
    p=d.probability.to_numpy(float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError(f'{path}: prediction alignment/probabilities')
    return p
