"""Participant-disjoint emotion-embedding teachers; TEST is never read or inferred.

New cached embeddings: SUPERB Wav2Vec2 emotion recognition for audio and
emotion-tuned DistilRoBERTa for text. The same original segment manifest is the
only segmentation source. Each candidate must Pareto-improve on original DEV
log loss, Brier score, and AUROC before it becomes active for KD.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,brier_score_loss,f1_score,log_loss,roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib, torch
from transformers import AutoModelForAudioClassification,AutoModelForSequenceClassification,AutoFeatureExtractor,AutoTokenizer
from huggingface_hub import model_info

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.students.data_utils import (index_sources,load_audio_segment,make_aligned_segments,read_split,read_turns)
from scripts.teachers.context_utils import digest,participant_weights,save_json

AUDIO_MODEL='superb/wav2vec2-base-superb-er'; TEXT_MODEL='j-hartmann/emotion-english-distilroberta-base'


def parse_args(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=103)
    p.add_argument('--data-root',type=Path,default=Path('/content/drive/MyDrive/DAIC_WOZ'))
    p.add_argument('--c',type=float,default=.1); return p.parse_args(argv)


def metric(y,p):
    q=np.clip(p,1e-7,1-1e-7); pred=q>=.5
    return {'n_segments':int(len(y)),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'depressed_f1':float(f1_score(y,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'auroc':float(roc_auc_score(y,q)),'average_precision':float(average_precision_score(y,q)),
        'brier':float(brier_score_loss(y,q)),'log_loss':float(log_loss(y,q,labels=[0,1])),'threshold':.5}


def predictions(meta,p,out,split):
    d=meta[['participant_id','segment_id','split','label']].copy(); d['probability']=p
    d['prediction']=(p>=.5).astype(int); q=np.clip(p,1e-7,1-1e-7); d['logit']=np.log(q/(1-q))
    d.to_csv(out/f'{split}_segment_predictions.csv',index=False); return metric(d.label.to_numpy(),p)


def load_original(base,modality,meta):
    d=pd.read_csv(base/modality/'dev_segment_predictions.csv').set_index('segment_id').loc[meta.segment_id]
    if not np.array_equal(d.label.to_numpy(),meta.label.to_numpy()): raise ValueError('Original teacher labels do not match manifest')
    return metric(meta.label.to_numpy(),d.probability.to_numpy(float))


def encode_audio(model,processor,path,start,stop,device):
    x=load_audio_segment(path,start,stop,16000); b=processor(x,sampling_rate=16000,return_tensors='pt')
    with torch.inference_mode():
        h=model.wav2vec2(input_values=b.input_values.to(device)).last_hidden_state.mean(1)
    return h.squeeze(0).cpu().numpy().astype(np.float32)


def encode_text(model,tokenizer,text,device):
    b=tokenizer(text,return_tensors='pt',truncation=True,max_length=256); b={k:v.to(device) for k,v in b.items()}
    with torch.inference_mode(): h=model.base_model(**b).last_hidden_state[:,0]
    return h.squeeze(0).cpu().numpy().astype(np.float32)


def main(argv=None):
    from google.colab import drive
    drive.mount('/content/drive')
    a=parse_args(argv); np.random.seed(a.seed); torch.manual_seed(a.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    base=a.data_root/'experiments/teachers/segment_level_v1'/f'seed_{a.seed}'; protocol=json.loads((base/'protocol.json').read_text())
    cache=Path(protocol['cache_root']); cfg=json.loads((cache/'config.json').read_text()); meta=pd.read_csv(cache/'segment_manifest_train_dev.csv')
    train=read_split(a.data_root,'train_split_Depression_AVEC2017.csv',True).assign(split='train')
    dev=read_split(a.data_root,'dev_split_Depression_AVEC2017.csv',True); dev=dev.loc[dev.participant_id.ne(440)].assign(split='dev')
    if (len(train),len(dev))!=(107,34) or set(meta.split)!={'train','dev'}: raise ValueError('Unexpected protocol split')
    revisions={AUDIO_MODEL:model_info(AUDIO_MODEL).sha,TEXT_MODEL:model_info(TEXT_MODEL).sha}
    key=hashlib.sha256(json.dumps({'models':revisions,'manifest':digest(cache/'segment_manifest_train_dev.csv')},sort_keys=True).encode()).hexdigest()[:16]
    root=a.data_root/'experiments/features'/f'rapdskd_emotion_{key}'; root.mkdir(parents=True,exist_ok=True)
    run=base/'emotion_runs'/key; run.mkdir(parents=True,exist_ok=True); save_json(root/'config.json',{'models':revisions,'source_manifest':str(cache/'segment_manifest_train_dev.csv')})
    audio_index,text_index=index_sources(a.data_root); original_tokenizer=AutoTokenizer.from_pretrained(cfg['text_model'],revision=cfg['revisions'][cfg['text_model']])
    records=[]
    for row in pd.concat([train,dev]).itertuples():
        pid=int(row.participant_id); ap=audio_index.get(pid,[]); tp=text_index.get(pid,[])
        if len(ap)!=1 or len(tp)!=1: raise ValueError(f'{pid}: source coverage')
        for s in make_aligned_segments(pid,read_turns(tp[0]),original_tokenizer,cfg):
            records.append({'segment_id':s['segment_id'],'participant_id':pid,'split':row.split,'label':int(row.label),'start':s['start'],'stop':s['stop'],'text':s['text'],'audio_path':str(ap[0])})
    rebuilt=pd.DataFrame(records).sort_values('segment_id').reset_index(drop=True); expected=meta.sort_values('segment_id').reset_index(drop=True)
    keys=['segment_id','participant_id','split','label']
    if not rebuilt[keys].equals(expected[keys]): raise ValueError('Rebuilt segment IDs, participants, splits, or labels differ from original manifest')
    bounds=np.allclose(rebuilt[['start','stop']].to_numpy(float),expected[['start','stop']].to_numpy(float),rtol=0,atol=1e-7)
    if not bounds: raise ValueError('Rebuilt segment boundaries differ from original manifest beyond 1e-7 seconds')
    rebuilt=rebuilt.set_index('segment_id').loc[meta.segment_id].reset_index()
    ids=np.asarray(meta.segment_id,dtype='U32')
    def cached(name):
        path=root/f'{name}_embeddings.npz'
        if not path.exists(): return None
        with np.load(path,allow_pickle=False) as z:
            x=z['embeddings'].astype(np.float32); cached_ids=z['segment_ids'].astype(str)
        if cached_ids.tolist()!=ids.astype(str).tolist() or x.ndim!=2 or len(x)!=len(ids) or not np.isfinite(x).all():
            raise ValueError(f'{name}: invalid cached emotion embeddings')
        print('Reusing cached',name,'emotion embeddings:',len(x)); return x
    audio=cached('audio')
    if audio is None:
        am=AutoModelForAudioClassification.from_pretrained(AUDIO_MODEL,revision=revisions[AUDIO_MODEL]).to(device).eval(); ap=AutoFeatureExtractor.from_pretrained(AUDIO_MODEL,revision=revisions[AUDIO_MODEL])
        audio=np.stack([encode_audio(am,ap,r.audio_path,r.start,r.stop,device) for r in rebuilt.itertuples()]); del am,ap
        np.savez_compressed(root/'audio_embeddings.npz',segment_ids=ids,embeddings=audio); print('Saved audio emotion embeddings:',len(audio))
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    text=cached('text')
    if text is None:
        tm=AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL,revision=revisions[TEXT_MODEL]).to(device).eval(); tt=AutoTokenizer.from_pretrained(TEXT_MODEL,revision=revisions[TEXT_MODEL])
        text=np.stack([encode_text(tm,tt,r.text,device) for r in rebuilt.itertuples()]); del tm,tt
        np.savez_compressed(root/'text_embeddings.npz',segment_ids=ids,embeddings=text); print('Saved text emotion embeddings:',len(text))
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    rows=[]; selected={}
    for modality,x in [('audio',audio),('text',text)]:
        out=run/modality; out.mkdir(parents=True,exist_ok=True); tr=meta.split.eq('train').to_numpy(); dv=~tr
        scaler=StandardScaler().fit(x[tr]); xtr=scaler.transform(x[tr]); xdv=scaler.transform(x[dv]); joblib.dump(scaler,out/'scaler.joblib')
        model=LogisticRegression(C=a.c,class_weight=None,max_iter=1000,random_state=a.seed); model.fit(xtr,meta.loc[tr,'label'],sample_weight=participant_weights(meta.loc[tr]))
        train_m=predictions(meta.loc[tr],model.predict_proba(xtr)[:,1],out,'train'); dev_m=predictions(meta.loc[dv],model.predict_proba(xdv)[:,1],out,'dev')
        save_json(out/'metrics.json',{'train_segment':train_m,'dev_segment':dev_m})
        original=load_original(base,modality,meta.loc[dv]); good=dev_m['log_loss']<=original['log_loss'] and dev_m['brier']<=original['brier'] and dev_m['auroc']>=original['auroc']
        chosen=out if good else base/modality; selected[modality]={'candidate':'emotion_embedding_logreg' if good else 'original','root':str(chosen),'prediction_hashes':{s:digest(chosen/f'{s}_segment_predictions.csv') for s in ['train','dev']}}
        rows.extend([{'modality':modality,'candidate':'original','selected':not good,**original},{'modality':modality,'candidate':'emotion_embedding_logreg','selected':good,**dev_m}])
    table=pd.DataFrame(rows); table.to_csv(run/'teacher_comparison.csv',index=False)
    selection={'run_id':key,'selection_metric':'DEV-only Pareto: log loss, Brier, AUROC','manifest_sha256':digest(cache/'segment_manifest_train_dev.csv'),'cache_root':str(cache),'test_used':False,'modalities':selected}
    save_json(run/'selection.json',selection); save_json(base/'active_teacher_selection.json',selection); print(table.to_string(index=False)); print('TEST REMAINS CLOSED. Rebuild KD targets only after reviewing this table.')
    return selection


if __name__=='__main__': main()
