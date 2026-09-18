"""RA-PDS-KD segment-level audio/text teacher training for DAIC-WOZ.

Protocol:
- official participant split first;
- participant 440 excluded from DEV because original files are corrupted;
- aligned audio/text segments are created only inside TRAIN/DEV;
- audio and text embeddings are cached separately;
- teachers train/evaluate at SEGMENT level only;
- TEST participant IDs are used only for overlap/count checks. TEST is never
  featurized, queried, scored, tuned on, or used for teacher selection here.
"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import os,re,json,math,gc,hashlib,random,unicodedata
from importlib.metadata import version
import numpy as np
import pandas as pd

DATA_ROOT=Path('/content/drive/MyDrive/DAIC_WOZ')
EXP_ROOT=DATA_ROOT/'experiments'
SEED=103; MAX_EPOCHS=200; PATIENCE=20; BATCH_SIZE=32
EXCLUDED={440:'Original participant files corrupted; excluded by user decision'}
CFG=dict(protocol='RA-PDS-KD-segment-v1',segmenter='aligned_turn_partition_v1',
         max_segments_per_participant=128,max_audio_seconds=10.0,min_audio_seconds=0.5,
         sample_rate=16000,max_text_tokens=254,
         audio_model='facebook/wav2vec2-base-960h',
         text_model='sentence-transformers/all-MiniLM-L6-v2')
assert DATA_ROOT.is_dir(),DATA_ROOT
os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
random.seed(SEED); np.random.seed(SEED)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)); tmp.replace(path)

def read_split(name,labelled):
    p=DATA_ROOT/'metadata'/name
    d=pd.read_csv(p); d.columns=d.columns.str.strip().str.lower()
    ids=pd.to_numeric(d['participant_id'],errors='raise').astype(int)
    assert len(ids) and not ids.duplicated().any()
    out=pd.DataFrame({'participant_id':ids})
    if labelled:
        y=pd.to_numeric(d['phq8_binary'],errors='raise').astype(int)
        assert y.isin([0,1]).all(); out['label']=y
    return out,sha256_file(p)

train,train_hash=read_split('train_split_Depression_AVEC2017.csv',True)
dev,dev_hash=read_split('dev_split_Depression_AVEC2017.csv',True)
test_ids,test_hash=read_split('test_split_Depression_AVEC2017.csv',False)
train_ids=set(train.participant_id); dev_ids=set(dev.participant_id); test_set=set(test_ids.participant_id)
assert not train_ids&dev_ids and not train_ids&test_set and not dev_ids&test_set,'Participant leakage'
dev=dev.loc[~dev.participant_id.isin(EXCLUDED)].reset_index(drop=True)
manifest=pd.concat([train.assign(split='train'),dev.assign(split='dev')],ignore_index=True)
assert (len(train),len(dev),len(test_ids))==(107,34,47),(len(train),len(dev),len(test_ids))
assert manifest.groupby('split').label.nunique().eq(2).all()
print('Participants | train:',len(train),'dev:',len(dev),'test reserved:',len(test_ids))
print('TEST CLOSED: IDs used only for overlap/count checks.')

audio_index={}; text_index={}
for base,dirs,files in os.walk(DATA_ROOT):
    dirs[:]=[d for d in dirs if d not in {'experiments','processed','.git','__pycache__'}]
    for name in files:
        a=re.fullmatch(r'(\d+)_AUDIO\.wav',name,re.I)
        t=re.fullmatch(r'(\d+)_TRANSCRIPT\.(?:csv|txt)',name,re.I)
        if a: audio_index.setdefault(int(a.group(1)),[]).append(Path(base)/name)
        if t: text_index.setdefault(int(t.group(1)),[]).append(Path(base)/name)

problems=[]
for r in manifest.itertuples():
    for kind,index in [('audio',audio_index),('transcript',text_index)]:
        paths=index.get(int(r.participant_id),[])
        if len(paths)!=1: problems.append((int(r.participant_id),r.split,kind,len(paths)))
assert not problems,f'File coverage problems: {problems[:20]}'
manifest['audio_path']=[str(audio_index[int(p)][0]) for p in manifest.participant_id]
manifest['transcript_path']=[str(text_index[int(p)][0]) for p in manifest.participant_id]

def clean_text(v):
    v=unicodedata.normalize('NFC',str(v)).replace('\x00',' ')
    return re.sub(r'\s+',' ',v).strip()

def read_turns(path):
    d=pd.read_csv(path,sep='\t'); d.columns=d.columns.str.strip().str.lower()
    if not {'start_time','stop_time','speaker','value'}<=set(d.columns):
        d=pd.read_csv(path,sep=None,engine='python'); d.columns=d.columns.str.strip().str.lower()
    req={'start_time','stop_time','speaker','value'}; assert req<=set(d.columns),path
    d=d.loc[d.speaker.astype(str).str.strip().str.lower().eq('participant')].copy()
    d['value']=d.value.fillna('').map(clean_text)
    d=d.loc[d.value.ne('') & ~d.value.str.contains('scrubbed|redacted',case=False,regex=True)]
    for c in ['start_time','stop_time']: d[c]=pd.to_numeric(d[c],errors='raise')
    d=d.loc[(d.start_time>=0)&(d.stop_time>d.start_time)].sort_values('start_time').reset_index(drop=True)
    assert len(d),f'No usable participant turns: {path}'
    return d

from huggingface_hub import model_info
REV_PATH=EXP_ROOT/'features'/'rapdskd_encoder_revisions.json'
if REV_PATH.exists(): revisions=json.loads(REV_PATH.read_text())
else:
    revisions={CFG[k]:model_info(CFG[k]).sha for k in ['audio_model','text_model']}
    save_json(REV_PATH,revisions)
assert all(CFG[k] in revisions for k in ['audio_model','text_model'])

from transformers import AutoTokenizer
text_name=CFG['text_model']
text_tokenizer=AutoTokenizer.from_pretrained(text_name,revision=revisions[text_name])

def aligned_segments(pid,turns):
    """Create one shared segment ID for audio and text."""
    rows=[]
    for ti,row in enumerate(turns.itertuples()):
        ids=text_tokenizer.encode(row.value,add_special_tokens=False)
        if not ids: continue
        dur=float(row.stop_time-row.start_time)
        n=max(1,math.ceil(dur/CFG['max_audio_seconds']),math.ceil(len(ids)/CFG['max_text_tokens']))
        for k in range(n):
            s=float(row.start_time)+dur*k/n; e=float(row.start_time)+dur*(k+1)/n
            if e-s<CFG['min_audio_seconds']: continue
            a=round(len(ids)*k/n); b=round(len(ids)*(k+1)/n)
            piece=(ids[a:b] if b>a else ids)[:CFG['max_text_tokens']]
            txt=text_tokenizer.decode(piece,skip_special_tokens=True).strip() or row.value
            rows.append(dict(segment_id=f'{int(pid)}_{ti:04d}_{k:02d}',start=s,stop=e,text=txt,
                             source_turn=int(ti),part=int(k),parts=int(n),token_count=int(len(piece))))
    assert rows,f'{pid}: no aligned segments'
    if len(rows)>CFG['max_segments_per_participant']:
        idx=np.linspace(0,len(rows)-1,CFG['max_segments_per_participant'],dtype=int)
        rows=[rows[i] for i in idx]
    return rows

turns_by_id={int(r.participant_id):read_turns(r.transcript_path) for r in manifest.itertuples()}
segments_by_id={pid:aligned_segments(pid,turns) for pid,turns in turns_by_id.items()}
seg_rows=[]
for r in manifest.itertuples():
    for s in segments_by_id[int(r.participant_id)]:
        seg_rows.append({k:v for k,v in s.items() if k!='text'}|
                        {'participant_id':int(r.participant_id),'split':r.split,'label':int(r.label)})
segment_manifest=pd.DataFrame(seg_rows)
assert not segment_manifest.segment_id.duplicated().any()
assert set(segment_manifest.split)=={'train','dev'}
print(segment_manifest.groupby(['split','label']).size().rename('segments'))

cache_config=dict(CFG,revisions=revisions,split_hashes={'train':train_hash,'dev':dev_hash},
                  packages={p:version(p) for p in ['transformers','torch','numpy','scipy','soundfile']})
cache_key=hashlib.sha256(json.dumps(cache_config,sort_keys=True).encode()).hexdigest()[:16]
CACHE_ROOT=EXP_ROOT/'features'/f'rapdskd_segments_{cache_key}'
CACHE_ROOT.mkdir(parents=True,exist_ok=True)
save_json(CACHE_ROOT/'config.json',cache_config)
segment_manifest.to_csv(CACHE_ROOT/'segment_manifest_train_dev.csv',index=False)
print('Feature cache:',CACHE_ROOT)

from tqdm.auto import tqdm
signatures={}
for r in tqdm(list(manifest.itertuples()),desc='Fingerprint train/dev sources'):
    pid=int(r.participant_id); a=sha256_file(r.audio_path); t=sha256_file(r.transcript_path)
    signatures[pid]={'audio':{'audio_sha256':a,'transcript_sha256':t},
                     'text':{'transcript_sha256':t}}

def cache_path(pid,modality):
    p=CACHE_ROOT/modality; p.mkdir(parents=True,exist_ok=True)
    return p/f'{int(pid)}.npz'

def valid_cache(pid,modality):
    p=cache_path(pid,modality)
    if not p.exists(): return False
    try:
        with np.load(p,allow_pickle=False) as z:
            stored=json.loads(str(z['signature'].item()))
            ids=z['segment_ids'].astype(str).tolist()
            expected_ids=[s['segment_id'] for s in segments_by_id[int(pid)]]
            x=z['embeddings']
            return stored==signatures[int(pid)][modality] and ids==expected_ids and x.ndim==2 and len(x)==len(ids) and np.isfinite(x).all()
    except (ValueError,OSError,KeyError,TypeError,json.JSONDecodeError):
        return False

def save_cache(pid,modality,embeddings):
    p=cache_path(pid,modality); tmp=p.with_suffix('.tmp')
    ids=np.array([s['segment_id'] for s in segments_by_id[int(pid)]],dtype='U32')
    x=np.asarray(embeddings,dtype=np.float32)
    assert x.ndim==2 and len(x)==len(ids) and np.isfinite(x).all()
    with open(tmp,'wb') as f:
        np.savez_compressed(f,participant_id=int(pid),segment_ids=ids,embeddings=x,
                            signature=json.dumps(signatures[int(pid)][modality],sort_keys=True))
    tmp.replace(p)

import torch,soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor,AutoModel
device='cuda' if torch.cuda.is_available() else 'cpu'
print('Encoder device:',device)

pending=[r for r in manifest.itertuples() if not valid_cache(r.participant_id,'audio')]
if pending:
    name=CFG['audio_model']; proc=AutoFeatureExtractor.from_pretrained(name,revision=revisions[name])
    enc=AutoModel.from_pretrained(name,revision=revisions[name],use_safetensors=True).to(device).eval()
    enc.requires_grad_(False)
    for r in tqdm(pending,desc='Audio cache'):
        pid=int(r.participant_id); feats=[]
        with sf.SoundFile(r.audio_path) as wav:
            rate=wav.samplerate
            for s in segments_by_id[pid]:
                wav.seek(min(round(s['start']*rate),len(wav)))
                x=wav.read(max(1,round((s['stop']-s['start'])*rate)),dtype='float32',always_2d=True).mean(1)
                assert len(x) and np.isfinite(x).all()
                if rate!=CFG['sample_rate']:
                    g=math.gcd(rate,CFG['sample_rate'])
                    x=resample_poly(x,CFG['sample_rate']//g,rate//g).astype(np.float32)
                inputs=proc(x,sampling_rate=CFG['sample_rate'],return_tensors='pt')
                with torch.inference_mode():
                    h=enc(input_values=inputs.input_values.to(device)).last_hidden_state
                    feats.append(h.mean(1).squeeze(0).cpu().numpy())
        save_cache(pid,'audio',feats)
    del enc,proc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print('Audio caches ready:',sum(valid_cache(p,'audio') for p in manifest.participant_id))

pending=[r for r in manifest.itertuples() if not valid_cache(r.participant_id,'text')]
if pending:
    name=CFG['text_model']
    enc=AutoModel.from_pretrained(name,revision=revisions[name],use_safetensors=True).to(device).eval()
    enc.requires_grad_(False)
    for r in tqdm(pending,desc='Text cache'):
        pid=int(r.participant_id); texts=[s['text'] for s in segments_by_id[pid]]; feats=[]
        for i in range(0,len(texts),32):
            b=text_tokenizer(texts[i:i+32],padding=True,truncation=True,max_length=CFG['max_text_tokens']+2,return_tensors='pt')
            b={k:v.to(device) for k,v in b.items()}
            with torch.inference_mode():
                h=enc(**b).last_hidden_state; m=b['attention_mask'].unsqueeze(-1)
                e=(h*m).sum(1)/m.sum(1).clamp(min=1)
                e=torch.nn.functional.normalize(e,p=2,dim=1)
                feats.extend(e.cpu().numpy())
        save_cache(pid,'text',feats)
    del enc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
print('Text caches ready:',sum(valid_cache(p,'text') for p in manifest.participant_id))
del text_tokenizer

def load_matrix(modality,split):
    rows=manifest.loc[manifest.split.eq(split)]
    xs=[]; meta=[]
    for r in rows.itertuples():
        pid=int(r.participant_id)
        with np.load(cache_path(pid,modality),allow_pickle=False) as z:
            x=z['embeddings'].astype(np.float32); ids=z['segment_ids'].astype(str)
        xs.append(x)
        meta.extend({'participant_id':pid,'segment_id':sid,'split':split,'label':int(r.label)} for sid in ids)
    X=np.concatenate(xs,axis=0); M=pd.DataFrame(meta)
    assert len(X)==len(M) and np.isfinite(X).all()
    return X,M

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix,classification_report
import joblib,tensorflow as tf
tf.random.set_seed(SEED)

RUN_ROOT=EXP_ROOT/'teachers'/'segment_level_v1'/f'seed_{SEED}'
RUN_ROOT.mkdir(parents=True,exist_ok=True)
save_json(RUN_ROOT/'protocol.json',{
    'name':'RA-PDS-KD','level':'segment','participant_split_first':True,
    'train_participants':107,'dev_participants':34,'test_participants_reserved':47,
    'test_used_for_teacher_training_or_selection':False,'cache_root':str(CACHE_ROOT),'config':CFG})

def calc_metrics(y,p,threshold=0.5):
    pred=(p>=threshold).astype(int)
    return {'n_segments':int(len(y)),'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'depressed_f1':float(f1_score(y,pred,zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
            'threshold':float(threshold)}

def save_eval(meta,p,out,prefix):
    d=meta.copy(); d['probability']=p; d['prediction']=(p>=0.5).astype(int)
    q=np.clip(p,1e-7,1-1e-7); d['logit']=np.log(q/(1-q))
    d.to_csv(out/f'{prefix}_predictions.csv',index=False)
    y=d.label.to_numpy(); pred=d.prediction.to_numpy()
    cm=pd.DataFrame(confusion_matrix(y,pred,labels=[0,1]),index=['actual_0','actual_1'],columns=['pred_0','pred_1'])
    cr=pd.DataFrame(classification_report(y,pred,labels=[0,1],target_names=['non_depressed','depressed'],
                                         output_dict=True,zero_division=0)).T
    cm.to_csv(out/f'{prefix}_confusion_matrix.csv'); cr.to_csv(out/f'{prefix}_classification_report.csv')
    print(f'\n{out.name} | {prefix} | n={len(d)}'); print(cm.to_string())
    print(classification_report(y,pred,digits=4,zero_division=0))
    return calc_metrics(y,p)

def build_head(dim):
    reg=tf.keras.regularizers.l2(1e-3); x=tf.keras.Input((dim,))
    h=tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=reg)(x)
    h=tf.keras.layers.Dropout(0.5)(h); h=tf.keras.layers.Dense(32,activation='relu',kernel_regularizer=reg)(h)
    h=tf.keras.layers.Dropout(0.3)(h); y=tf.keras.layers.Dense(1,activation='sigmoid')(h)
    m=tf.keras.Model(x,y); m.compile(optimizer=tf.keras.optimizers.Adam(3e-4),loss='binary_crossentropy')
    return m

summary=[]
for modality in ['audio','text']:
    Xtr,Mtr=load_matrix(modality,'train'); Xdv,Mdv=load_matrix(modality,'dev')
    scaler=StandardScaler().fit(Xtr); Xtr=scaler.transform(Xtr); Xdv=scaler.transform(Xdv)
    ytr=Mtr.label.to_numpy(); ydv=Mdv.label.to_numpy()
    counts=np.bincount(ytr,minlength=2); assert (counts>0).all()
    cw={i:float(len(ytr)/(2*counts[i])) for i in [0,1]}
    out=RUN_ROOT/modality; out.mkdir(parents=True,exist_ok=True); joblib.dump(scaler,out/'scaler.joblib')
    model=build_head(Xtr.shape[1])
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=PATIENCE,restore_best_weights=True),
               tf.keras.callbacks.ModelCheckpoint(str(out/'teacher.keras'),monitor='val_loss',save_best_only=True)]
    hist=model.fit(Xtr,ytr,validation_data=(Xdv,ydv),epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,
                   class_weight=cw,callbacks=callbacks,verbose=2)
    pd.DataFrame(hist.history).to_csv(out/'history.csv',index=False)
    model=tf.keras.models.load_model(out/'teacher.keras')
    ptr=model.predict(Xtr,batch_size=256,verbose=0).reshape(-1)
    pdv=model.predict(Xdv,batch_size=256,verbose=0).reshape(-1)
    train_metrics=save_eval(Mtr,ptr,out,'train_segment')
    dev_metrics=save_eval(Mdv,pdv,out,'dev_segment')
    save_json(out/'metrics.json',{'train_segment':train_metrics,'dev_segment':dev_metrics,
                                  'class_weights':{str(k):v for k,v in cw.items()},
                                  'segment_counts':counts.astype(int).tolist()})
    summary.append({'modality':modality,**dev_metrics})
    tf.keras.backend.clear_session(); gc.collect()

pd.DataFrame(summary).to_csv(RUN_ROOT/'teacher_summary.csv',index=False)
print('\nRA-PDS-KD segment teachers complete:',RUN_ROOT)
print('TEST REMAINS CLOSED. Final test evaluation belongs to the frozen student only.')
