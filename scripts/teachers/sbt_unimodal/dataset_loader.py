"""Released SBT loader semantics with participant-only RA-PDS-KD input construction.
Source: ghy-yhg/SBT-Net/SBT-Net/depression_model/dataset_loader.py.
16 kHz librosa audio, first 15 seconds, no waveform standardization;
ALBERT-base tokenizer, first 128 tokens including special tokens, fixed text padding.
The original raw-DAIC-to-files conversion is not released. Our participant speech
concatenation is explicitly an adaptation, not a claimed reproduction of that step.
"""
import json
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.teachers.sbt_unimodal.data import KD_CFG

def load_sample_audio(row):
    limit=int(KD_CFG['sample_rate']*KD_CFG['teacher_audio_seconds'])
    intervals=json.loads(row.audio_intervals) if 'audio_intervals' in row.index and isinstance(row.audio_intervals,str) else [[float(row.start),float(row.stop)]]
    pieces=[]; remaining=limit
    for start,stop in intervals:
        if remaining<=0: break
        x,_=librosa.load(row.audio_path,sr=KD_CFG['sample_rate'],mono=True,
                         offset=float(start),duration=min(float(stop-start),remaining/KD_CFG['sample_rate']))
        x=x[:remaining]; pieces.append(x); remaining-=len(x)
    x=np.concatenate(pieces).astype(np.float32) if pieces else np.empty(0,np.float32)
    if len(x)<400 or not np.isfinite(x).all(): raise ValueError('Invalid or too-short audio sample')
    return x

class AudioDataset(Dataset):
    def __init__(self,df,id_col): self.df=df.reset_index(drop=True); self.id_col=id_col
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; x=load_sample_audio(r)
        return x,float(r.label),int(r.participant_id),str(r[self.id_col]),str(r.split)

class TextDataset(Dataset):
    def __init__(self,df,id_col): self.df=df.reset_index(drop=True); self.id_col=id_col
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; return str(r.text),float(r.label),int(r.participant_id),str(r[self.id_col]),str(r.split)

class AudioCollator:
    def __init__(self,extractor): self.extractor=extractor
    def __call__(self,batch):
        x,y,pid,sid,split=zip(*batch)
        # Padding is transport only. The model removes it before encoder execution.
        limit=int(KD_CFG['teacher_audio_seconds']*KD_CFG['sample_rate'])
        waves=[torch.as_tensor(v[:limit],dtype=torch.float32) for v in x]
        lengths=torch.tensor([len(v) for v in waves])
        values=torch.nn.utils.rnn.pad_sequence(waves,batch_first=True)
        mask=torch.arange(values.shape[1])[None,:]<lengths[:,None]
        return dict(input_values=values,attention_mask=mask,label=torch.tensor(y,dtype=torch.float32),
                    participant_id=pid,sample_id=sid,split=split)

class TextCollator:
    def __init__(self,tokenizer,max_len=128): self.tokenizer=tokenizer; self.max_len=max_len
    def __call__(self,batch):
        x,y,pid,sid,split=zip(*batch)
        z=self.tokenizer(list(x),padding='max_length',truncation=True,max_length=self.max_len,
                         return_tensors='pt')
        return dict(input_ids=z.input_ids,attention_mask=z.attention_mask,label=torch.tensor(y,dtype=torch.float32),
                    window_owner=torch.arange(len(y)),n_samples=len(y),
                    participant_id=pid,sample_id=sid,split=split)

