from __future__ import annotations
from pathlib import Path
import math,numpy as np,torch
from torch.utils.data import Dataset
from . import SEGMENT_FRAMES,MAX_AUDIO_SEGMENTS

class RichParticipantDataset(Dataset):
    def __init__(self,manifest,text_path,standardizer,split,seed=103,max_segments=MAX_AUDIO_SEGMENTS,local_audio_root=None):
        import pandas as pd
        self.meta=pd.read_csv(manifest).sort_values("participant_id").reset_index(drop=True)
        self.text=np.load(text_path).astype(np.float32); assert len(self.meta)==len(self.text)
        s=np.load(standardizer); self.mean=s["mean"].astype(np.float32); self.std=s["std"].astype(np.float32)
        self.split=split; self.seed=int(seed); self.epoch=0; self.max_segments=int(max_segments)
        self.local=Path(local_audio_root) if local_audio_root else None
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __len__(self): return len(self.meta)
    def _path(self,r):
        if self.local is None: return Path(r.feature_path)
        p=self.local/self.split/f"{int(r.participant_id)}.npy"
        if not p.exists(): raise FileNotFoundError(p)
        return p
    def _indices(self,n,pid):
        if n<=self.max_segments: return np.arange(n,dtype=int)
        if self.split=="train":
            rng=np.random.default_rng(self.seed+1000003*self.epoch+7919*int(pid))
            return np.sort(rng.choice(n,size=self.max_segments,replace=False))
        return np.linspace(0,n-1,self.max_segments,dtype=int)
    def __getitem__(self,i):
        r=self.meta.iloc[i]; pid=int(r.participant_id); x=np.load(self._path(r),mmap_mode="r")
        assert x.ndim==2 and x.shape[0]==130 and x.shape[1]>0
        n=math.ceil(x.shape[1]/SEGMENT_FRAMES); idx=self._indices(n,pid)
        seg=np.zeros((len(idx),130,SEGMENT_FRAMES),np.float32)
        for j,q in enumerate(idx):
            a=q*SEGMENT_FRAMES; b=min((q+1)*SEGMENT_FRAMES,x.shape[1])
            seg[j,:,:b-a]=(np.asarray(x[:,a:b],np.float32)-self.mean[:,None])/self.std[:,None]
        return {"participant_id":pid,"label":int(r.label),"audio":seg,"text":self.text[i],"n_segments":len(idx)}

def collate(batch):
    s=max(x["n_segments"] for x in batch); b=len(batch)
    audio=np.zeros((b,s,130,SEGMENT_FRAMES),np.float32); mask=np.zeros((b,s),bool)
    for i,x in enumerate(batch): audio[i,:x["n_segments"]]=x["audio"]; mask[i,:x["n_segments"]]=True
    return {"participant_id":torch.tensor([x["participant_id"] for x in batch]).long(),
            "label":torch.tensor([x["label"] for x in batch]).float(),
            "audio":torch.from_numpy(audio).float(),"text":torch.from_numpy(np.stack([x["text"] for x in batch])).float(),
            "mask":torch.from_numpy(mask).bool()}
