from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib,math,os,re,unicodedata
import numpy as np,pandas as pd,soundfile as sf
from scipy.signal import resample_poly

PAD_ID=0; UNK_ID=1

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def clean_text(v):
    v=unicodedata.normalize("NFC",str(v)).replace("\x00"," ")
    return re.sub(r"\s+"," ",v).strip()

def simple_tokens(v):
    return re.findall(r"[A-Za-z0-9']+|[^\w\s]",clean_text(v).lower())

def read_split(root,filename):
    p=Path(root)/"metadata"/filename; d=pd.read_csv(p); d.columns=d.columns.str.strip().str.lower()
    assert {"participant_id","phq8_binary"}<=set(d.columns),p
    pid=pd.to_numeric(d.participant_id,errors="raise").astype(int)
    y=pd.to_numeric(d.phq8_binary,errors="raise").astype(int)
    assert not pid.duplicated().any() and y.isin([0,1]).all()
    return pd.DataFrame({"participant_id":pid,"label":y}),p

def find_sources(root,pids):
    want=set(map(int,pids)); audio={}; text={}
    for base,dirs,names in os.walk(root):
        dirs[:]=[d for d in dirs if d not in {"experiments","processed",".git","__pycache__"}]
        for name in names:
            a=re.fullmatch(r"(\d+)_AUDIO\.wav",name,re.I); t=re.fullmatch(r"(\d+)_TRANSCRIPT\.(?:csv|txt)",name,re.I)
            if a and int(a.group(1)) in want: audio.setdefault(int(a.group(1)),[]).append(Path(base)/name)
            if t and int(t.group(1)) in want: text.setdefault(int(t.group(1)),[]).append(Path(base)/name)
    out={}
    for pid in sorted(want):
        assert len(audio.get(pid,[]))==1,f"{pid}: audio coverage={len(audio.get(pid,[]))}"
        assert len(text.get(pid,[]))==1,f"{pid}: transcript coverage={len(text.get(pid,[]))}"
        out[pid]=(audio[pid][0],text[pid][0])
    return out

def read_turns(path):
    d=pd.read_csv(path,sep="\t"); d.columns=d.columns.str.strip().str.lower()
    if not {"start_time","stop_time","speaker","value"}<=set(d.columns):
        d=pd.read_csv(path,sep=None,engine="python"); d.columns=d.columns.str.strip().str.lower()
    assert {"start_time","stop_time","speaker","value"}<=set(d.columns),path
    d=d.loc[d.speaker.astype(str).str.strip().str.lower().eq("participant")].copy()
    d["value"]=d.value.fillna("").map(clean_text)
    d=d.loc[d.value.ne("") & ~d.value.str.contains("scrubbed|redacted",case=False,regex=True)]
    for c in ["start_time","stop_time"]: d[c]=pd.to_numeric(d[c],errors="raise")
    d=d.loc[(d.start_time>=0)&(d.stop_time>d.start_time)].sort_values("start_time").reset_index(drop=True)
    assert len(d),f"{path}: no usable Participant turns"; return d

def segment_turns(pid,turns,cfg):
    rows=[]
    for ti,r in enumerate(turns.itertuples()):
        tok=simple_tokens(r.value)
        if not tok: continue
        dur=float(r.stop_time-r.start_time)
        n=max(1,math.ceil(dur/cfg["max_audio_seconds"]),math.ceil(len(tok)/cfg["max_text_tokens"]))
        for k in range(n):
            start=float(r.start_time)+dur*k/n; stop=float(r.start_time)+dur*(k+1)/n
            if stop-start<cfg["min_audio_seconds"]: continue
            a=round(len(tok)*k/n); b=round(len(tok)*(k+1)/n); piece=(tok[a:b] if b>a else tok)[:cfg["max_text_tokens"]]
            rows.append({"segment_id":f"{int(pid)}_{ti:04d}_{k:02d}","start":start,"stop":stop,"tokens":piece})
    assert rows,f"{pid}: no aligned segments"
    if len(rows)>cfg["max_segments_per_participant"]:
        idx=np.linspace(0,len(rows)-1,cfg["max_segments_per_participant"],dtype=int); rows=[rows[i] for i in idx]
    return rows

def build_vocab(segment_lists,max_vocab=10000,min_freq=2):
    c=Counter()
    for segs in segment_lists:
        for s in segs: c.update(s["tokens"])
    items=sorted(((t,n) for t,n in c.items() if n>=min_freq),key=lambda x:(-x[1],x[0]))
    v={"<PAD>":PAD_ID,"<UNK>":UNK_ID}
    for t,_ in items[:max(0,max_vocab-2)]: v[t]=len(v)
    return v

def encode_tokens(tokens,vocab,max_tokens):
    x=np.full(max_tokens,PAD_ID,dtype=np.int64)
    ids=[vocab.get(t,UNK_ID) for t in tokens[:max_tokens]]
    if ids: x[:len(ids)]=ids
    return x

def load_audio_segment(path,start,stop,target_sr):
    with sf.SoundFile(path) as wav:
        sr=int(wav.samplerate); wav.seek(min(round(start*sr),len(wav)))
        x=wav.read(max(1,round((stop-start)*sr)),dtype="float32",always_2d=True).mean(1)
    assert len(x) and np.isfinite(x).all()
    if sr!=target_sr:
        g=math.gcd(sr,target_sr); x=resample_poly(x,target_sr//g,sr//g).astype(np.float32)
    return x.astype(np.float32)

def logmel_summary(x,sr,n_mels):
    import librosa
    m=librosa.feature.melspectrogram(y=np.asarray(x,np.float32),sr=sr,n_mels=n_mels,n_fft=512,hop_length=160,win_length=400,power=2.0)
    z=librosa.power_to_db(np.maximum(m,1e-10),ref=np.max)
    f=np.concatenate([z.mean(1),z.std(1)]).astype(np.float32)
    assert f.shape==(2*n_mels,) and np.isfinite(f).all(); return f

def fit_standardizer(x):
    x=np.asarray(x,np.float32); mean=x.mean(0).astype(np.float32); std=x.std(0).astype(np.float32)
    return mean,np.where(std<1e-6,1.0,std).astype(np.float32)

def standardize(x,mean,std): return ((np.asarray(x,np.float32)-mean)/std).astype(np.float32)
