"""Improve teachers using existing TRAIN/DEV caches. No encoder or TEST inference.

Compares saved original heads, participant-balanced MLPs, and contextual GRUs.
Outputs preserve one prediction per segment. Context is offline privileged teacher
information, not participant-level prediction aggregation. Completed runs are reused.
"""
from pathlib import Path
import argparse, gc, hashlib, json, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (log_loss,roc_auc_score,average_precision_score,
    brier_score_loss,accuracy_score,f1_score,balanced_accuracy_score,confusion_matrix,classification_report)

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.teachers.context_utils import (digest,save_json,validate_manifest,participant_weights,
    window_indices,materialize_windows,align_predictions)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/content/drive/MyDrive/DAIC_WOZ'))
    p.add_argument('--seed',type=int,default=103); p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--patience',type=int,default=10); p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--window',type=int,default=5); p.add_argument('--max-gap',type=float,default=30.)
    p.add_argument('--lr',type=float,default=3e-4)
    a=p.parse_args(argv)
    if min(a.epochs,a.patience,a.batch_size)<1 or a.lr<=0 or a.window<1 or a.window%2!=1 or a.max_gap<0:
        p.error('Invalid training/window parameters')
    return a


def read_labels(path):
    d=pd.read_csv(path); d.columns=d.columns.str.strip().str.lower()
    d=d[['participant_id','phq8_binary']].rename(columns={'phq8_binary':'label'})
    if d.isna().any().any() or d.participant_id.duplicated().any() or not d.label.isin([0,1]).all():
        raise ValueError(f'Invalid split: {path}')
    if not (d.participant_id%1==0).all(): raise ValueError('Noninteger participant ID')
    return d.astype(int)


def load_inputs(args):
    base=args.data_root/'experiments/teachers/segment_level_v1'/f'seed_{args.seed}'
    protocol=json.loads((base/'protocol.json').read_text()); cache=Path(protocol['cache_root'])
    config=json.loads((cache/'config.json').read_text()); meta=pd.read_csv(cache/'segment_manifest_train_dev.csv')
    split_paths={s:args.data_root/'metadata'/f'{s}_split_Depression_AVEC2017.csv' for s in ['train','dev']}
    for s,path in split_paths.items():
        if digest(path)!=config['split_hashes'][s]: raise ValueError(f'{s} split changed since embedding extraction')
    tr=read_labels(split_paths['train']); dv=read_labels(split_paths['dev']); dv=dv.loc[dv.participant_id.ne(440)]
    # Deliberately read ONLY TEST IDs for the overlap guard.
    te=pd.read_csv(args.data_root/'metadata/test_split_Depression_AVEC2017.csv',usecols=lambda c:c.strip().lower()=='participant_id')
    test_ids=te.iloc[:,0]
    if test_ids.isna().any() or test_ids.duplicated().any() or not (test_ids%1==0).all(): raise ValueError('Invalid TEST IDs')
    if (len(tr),len(dv),len(te))!=(107,34,47): raise ValueError('Unexpected official split counts')
    validate_manifest(meta,tr,dv,test_ids)
    fingerprints={'manifest':digest(cache/'segment_manifest_train_dev.csv'),'config':digest(cache/'config.json')}
    for modality in ['audio','text']:
        for pid in meta.participant_id.unique():
            p=cache/modality/f'{pid}.npz'; fingerprints[f'{modality}/{pid}']=digest(p)
        for split in ['train','dev']:
            p=base/modality/f'{split}_segment_predictions.csv'; fingerprints[f'baseline/{modality}/{split}']=digest(p)
            align_predictions(p,meta.loc[meta.split.eq(split)])
    for pid in meta.participant_id.unique():
        with np.load(cache/'audio'/f'{pid}.npz',allow_pickle=False) as a, np.load(cache/'text'/f'{pid}.npz',allow_pickle=False) as t:
            sa=json.loads(str(a['signature'].item())); st=json.loads(str(t['signature'].item()))
            if sa['transcript_sha256']!=st['transcript_sha256']: raise ValueError(f'{pid}: audio/text source signature mismatch')
    return base,cache,meta,fingerprints


def load_embeddings(cache,modality,meta):
    chunks=[]
    for pid,group in meta.groupby('participant_id',sort=False):
        with np.load(cache/modality/f'{pid}.npz',allow_pickle=False) as z:
            ids=z['segment_ids'].astype(str); x=z['embeddings'].astype(np.float32)
            if int(z['participant_id'])!=pid or len(set(ids))!=len(ids) or set(ids)!=set(group.segment_id):
                raise ValueError(f'{modality}/{pid}: cache IDs disagree with manifest')
            if x.ndim!=2 or len(x)!=len(ids) or not np.isfinite(x).all(): raise ValueError('Invalid embeddings')
            chunks.append(pd.DataFrame(x,index=ids))
    all_x=pd.concat(chunks).loc[meta.segment_id].to_numpy(np.float32)
    if not np.isfinite(all_x).all(): raise ValueError('Inconsistent embedding dimensions')
    return all_x


def metrics(y,p):
    pred=p>=.5
    return dict(n_segments=len(y),accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        depressed_f1=float(f1_score(y,pred,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        auroc=float(roc_auc_score(y,p)),average_precision=float(average_precision_score(y,p)),
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),threshold=.5)


def export(meta,p,out,split):
    d=meta[['participant_id','segment_id','split','label']].copy(); d['probability']=p; d['prediction']=(p>=.5).astype(int)
    q=np.clip(p,1e-7,1-1e-7); d['logit']=np.log(q/(1-q)); d.to_csv(out/f'{split}_segment_predictions.csv',index=False)
    pd.DataFrame(confusion_matrix(d.label,d.prediction,labels=[0,1])).to_csv(out/f'{split}_segment_confusion_matrix.csv',index=False)
    report=classification_report(d.label,d.prediction,labels=[0,1],output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(out/f'{split}_segment_classification_report.csv')
    return metrics(d.label,p)


def build_model(tf,dim,kind,window,lr):
    if kind=='balanced_mlp':
        inputs=tf.keras.Input((dim,),name='target'); reg=tf.keras.regularizers.l2(1e-3)
        h=tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=reg)(inputs)
        h=tf.keras.layers.Dropout(.5)(h); h=tf.keras.layers.Dense(32,activation='relu',kernel_regularizer=reg)(h)
        h=tf.keras.layers.Dropout(.3)(h)
    else:
        context=tf.keras.Input((window,dim),name='context'); mask=tf.keras.Input((window,),dtype='bool',name='mask')
        target=tf.keras.Input((dim,),name='target'); projection=tf.keras.layers.Dense(64,activation='relu')
        seq=projection(context); h=tf.keras.layers.GRU(64)(seq,mask=mask)
        h=tf.keras.layers.Concatenate()([h,projection(target)])
        h=tf.keras.layers.Dropout(.3)(h); h=tf.keras.layers.Dense(32,activation='relu')(h)
        inputs={'context':context,'mask':mask,'target':target}
    output=tf.keras.layers.Dense(1,activation='sigmoid')(h); model=tf.keras.Model(inputs,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr,clipnorm=1.),loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryCrossentropy(name='bce')])
    return model


def fit_candidate(tf,args,kind,out,xtr,xdv,tr,dv,indices):
    import joblib
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        stored=json.loads((out/'complete.json').read_text())
        if all((out/name).is_file() and digest(out/name)==sha for name,sha in stored['artifacts'].items()):
            print('Reusing completed',out.name); return stored['metrics']
        raise ValueError(f'Completed artifacts changed: {out}; use a new run configuration')
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(args.seed)
    scaler=StandardScaler().fit(xtr); a=scaler.transform(xtr).astype(np.float32); b=scaler.transform(xdv).astype(np.float32)
    joblib.dump(scaler,out/'scaler.joblib')
    train_input,dev_input=a,b
    if kind=='context_gru':
        av,am=materialize_windows(a,indices['train']); bv,bm=materialize_windows(b,indices['dev'])
        train_input={'target':a,'context':av,'mask':am}; dev_input={'target':b,'context':bv,'mask':bm}
    model=build_model(tf,a.shape[1],kind,args.window,args.lr)
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_bce',mode='min',patience=args.patience,restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out/'teacher.keras'),monitor='val_bce',mode='min',save_best_only=True)]
    hist=model.fit(train_input,tr.label.to_numpy(),sample_weight=participant_weights(tr),validation_data=(dev_input,dv.label.to_numpy()),
                   batch_size=args.batch_size,epochs=args.epochs,callbacks=callbacks,verbose=2)
    pd.DataFrame(hist.history).to_csv(out/'history.csv',index=False)
    model=tf.keras.models.load_model(out/'teacher.keras')
    scores={}
    for split,inputs,meta in [('train',train_input,tr),('dev',dev_input,dv)]:
        p=model.predict(inputs,batch_size=args.batch_size,verbose=0).reshape(-1)
        scores[f'{split}_segment']=export(meta,p,out,split)
    save_json(out/'metrics.json',scores)
    artifacts={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'}
    save_json(out/'complete.json',{'metrics':scores,'artifacts':artifacts})
    del model; tf.keras.backend.clear_session(); gc.collect()
    return scores


def main(argv=None):
    args=parse_args(argv); base,cache,meta,fingerprints=load_inputs(args)
    import tensorflow as tf
    for device in tf.config.list_physical_devices('GPU'):
        try: tf.config.experimental.set_memory_growth(device,True)
        except RuntimeError: pass  # Already initialized in this Colab runtime.
    config={**vars(args),'data_root':str(args.data_root),'tensorflow':tf.__version__,
            'code':digest(__file__),'utils':digest(Path(__file__).with_name('context_utils.py')),'inputs':fingerprints}
    run_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    run=base/'context_runs'/run_id; save_json(run/'config.json',config)
    tr=meta.loc[meta.split.eq('train')].reset_index(drop=True); dv=meta.loc[meta.split.eq('dev')].reset_index(drop=True)
    indices={s:window_indices(m,args.window,args.max_gap)[0] for s,m in [('train',tr),('dev',dv)]}
    for s,idx in indices.items(): print(s,'mean context length:',round(float((idx>=0).sum(1).mean()),2))
    rows=[]; selection={'run_id':run_id,'selection_metric':'DEV segment BCE, threshold fixed at 0.5',
        'manifest_sha256':fingerprints['manifest'],'cache_root':str(cache),'test_used':False,'modalities':{}}
    for modality in ['audio','text']:
        all_x=load_embeddings(cache,modality,meta); xtr=all_x[meta.split.eq('train')]; xdv=all_x[meta.split.eq('dev')]
        original=base/modality; p=align_predictions(original/'dev_segment_predictions.csv',dv)
        candidates=[('original',original,metrics(dv.label,p))]
        for kind in ['balanced_mlp','context_gru']:
            out=run/modality/kind
            scores=fit_candidate(tf,args,kind,out,xtr,xdv,tr,dv,indices)
            candidates.append((kind,out,scores['dev_segment']))
        # Original wins ties; never force a new teacher if its predictive BCE is worse.
        chosen=min(candidates,key=lambda c:c[2]['log_loss'])
        for kind,path,score in candidates: rows.append({'modality':modality,'candidate':kind,'selected':kind==chosen[0],**score})
        kind,path,score=chosen
        selection['modalities'][modality]={'candidate':kind,'root':str(path),
            'prediction_hashes':{s:digest(path/f'{s}_segment_predictions.csv') for s in ['train','dev']}}
        print(modality,'selected:',kind,'DEV BCE:',round(score['log_loss'],4))
        del all_x,xtr,xdv; gc.collect()
    table=pd.DataFrame(rows); table.to_csv(run/'teacher_comparison.csv',index=False); print(table.to_string(index=False))
    save_json(run/'selection.json',selection)
    save_json(base/'active_teacher_selection.json',selection)
    print('Selected teacher predictions ready. Rebuild KD targets next. TEST REMAINS CLOSED.')
    return selection


if __name__=='__main__': main()
