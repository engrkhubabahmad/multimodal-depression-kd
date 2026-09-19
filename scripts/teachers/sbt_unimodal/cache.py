"""Content-addressed disk features. No dataset-sized embedding array in RAM."""
from pathlib import Path
import hashlib, json, os, shutil, tempfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

CACHE_VERSION='disk-shards-v1'

def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def frame_digest(df):
    # Paths may change from Drive to local disk; source bytes are hashed separately.
    cols=sorted(c for c in df.columns if not c.endswith('_path'))
    return hashlib.sha256(df[cols].to_json(orient='split',index=False,double_precision=15).encode()).hexdigest()

def atomic_write(path,writer):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: writer(f); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_json(path,value):
    atomic_write(path,lambda f:f.write(json.dumps(value,sort_keys=True,allow_nan=False).encode()))

def atomic_torch(path,value):
    atomic_write(path,lambda f:torch.save(value,f))

def stage_sources(manifest,local_root):
    """Read TRAIN/DEV source bytes once per invocation; use local disk for training."""
    root=Path(local_root)/'sources'; root.mkdir(parents=True,exist_ok=True)
    mapping={}; fingerprints=[]
    for r in tqdm(list(manifest.itertuples()),desc='Verify/stage TRAIN+DEV audio',colour='green'):
        if r.split not in {'train','dev'}: raise ValueError('TEST source staging forbidden')
        record={'participant_id':int(r.participant_id),'split':r.split}
        for column in ('audio_path','transcript_path'):
            source=Path(getattr(r,column)); sha=digest_file(source); record[column]=sha
            if column=='audio_path':
                dest=root/(sha+source.suffix)
                if not dest.exists() or digest_file(dest)!=sha:
                    atomic_write(dest,lambda f:copy_file(source,f))
                    if digest_file(dest)!=sha: raise ValueError('Source changed while copying: '+str(source))
                mapping[str(source)]=str(dest)
        fingerprints.append(record)
    return mapping,fingerprints

def copy_file(source,dest):
    with open(source,'rb') as f: shutil.copyfileobj(f,dest,length=1024*1024)

class DiskFeatures(Dataset):
    def __init__(self,root,df,batch_size):
        self.root=Path(root); self.df=df.reset_index(drop=True); self.batch_size=batch_size
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        shard=np.load(self.root/f'{i//self.batch_size:06d}.npy',mmap_mode='r',allow_pickle=False)
        x=torch.from_numpy(np.array(shard[i%self.batch_size],copy=True)); r=self.df.iloc[i]
        return {'features':x,'label':torch.tensor(float(r.label)),
                'participant_id':int(r.participant_id),'sample_id':str(r.sample_id),'split':str(r.split)}

@torch.no_grad()
def build_features(model,dataset,collate,df,identity,persistent_root,local_root,batch_size,device,encode):
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise ValueError('Embedding cache is only valid for a frozen encoder')
    key=digest_json(dict(version=CACHE_VERSION,identity=identity,rows=frame_digest(df),batch=batch_size))
    remote=Path(persistent_root)/key; local=Path(local_root)/'features'/key
    remote.mkdir(parents=True,exist_ok=True); local.mkdir(parents=True,exist_ok=True)
    atomic_json(remote/'identity.json',{'key':key,'identity':identity,'rows':frame_digest(df)})
    model.eval(); count=0
    # Valid shards survive interruption. Metadata is written only after shard bytes.
    for j,start in enumerate(tqdm(range(0,len(df),batch_size),desc=f"{identity['modality']} frozen feature cache",colour='green')):
        n=min(batch_size,len(df)-start)
        p=remote/f'{j:06d}.npy'; meta=p.with_suffix('.json'); dest=local/p.name
        expected={'rows':n,'width':model.encoder.config.hidden_size,'key':key}
        valid=False
        try:
            info=json.loads(meta.read_text())
            valid=all(info.get(k)==v for k,v in expected.items()) and digest_file(p)==info['sha256']
            if valid:
                a=np.load(p,mmap_mode='r',allow_pickle=False)
                valid=a.shape==(n,expected['width']) and a.dtype==np.float32 and np.isfinite(a).all()
                del a
        except (OSError,ValueError,KeyError): pass
        if not valid:
            raw=[dataset[i] for i in range(start,start+n)]
            b=collate(raw)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.startswith('cuda')):
                features=encode(model,b,identity['modality'],device)
            array=features.float().cpu().numpy()
            if array.shape!=(n,expected['width']) or not np.isfinite(array).all():
                raise ValueError('Invalid encoder features')
            atomic_write(p,lambda f:np.save(f,array,allow_pickle=False))
            info=dict(expected,sha256=digest_file(p)); atomic_json(meta,info)
        if not dest.exists() or digest_file(dest)!=info['sha256']:
            atomic_write(dest,lambda f:copy_file(p,f))
        count+=n
    if count!=len(df): raise ValueError('Incomplete feature cache')
    atomic_json(remote/'complete.json',{'key':key,'rows':count})
    return DiskFeatures(local,df,batch_size)
