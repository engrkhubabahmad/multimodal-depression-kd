from __future__ import annotations
from pathlib import Path
import argparse,gc,hashlib,json,random,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor,AutoTokenizer,get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,roc_auc_score,average_precision_score,confusion_matrix,classification_report

from scripts.students.data_utils import load_audio_segment
from scripts.teachers.sbt_unimodal.data import KD_CFG,build_manifests,save_json
from scripts.teachers.sbt_unimodal.models import AudioSBTTeacher,TextSBTTeacher
from scripts.teachers.sbt_unimodal.cache import (atomic_torch,atomic_json,digest_file,digest_json,
    frame_digest,stage_sources,build_features)

CODE_VERSION='sbt-unimodal-v2-disk'

def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=103); p.add_argument('--epochs',type=int,default=85)
    p.add_argument('--freeze-epochs',type=int,default=25); p.add_argument('--patience',type=int,default=10)
    p.add_argument('--audio-batch',type=int,default=16); p.add_argument('--text-batch',type=int,default=16)
    p.add_argument('--audio-accum',type=int,default=1); p.add_argument('--text-accum',type=int,default=1)
    p.add_argument('--head-lr',type=float,default=2e-5); p.add_argument('--encoder-lr',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=1e-2); p.add_argument('--force-retrain',action='store_true')
    p.add_argument('--s1-patience',type=int,default=5)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--local-cache',default='/content/rapdskd_teacher_cache')
    p.add_argument('--data-root',default='/content/drive/MyDrive/DAIC_WOZ')
    args=p.parse_args(argv)
    if args.workers<0: p.error('workers must be nonnegative')
    if not 1<=args.freeze_epochs<args.epochs: p.error('Require 1 <= freeze-epochs < epochs')
    if min(args.audio_batch,args.text_batch,args.audio_accum,args.text_accum,args.patience,args.s1_patience)<1:
        p.error('Batches, accumulation and patience must be positive')
    return args

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def participant_sampler(df):
    pc=df.participant_id.value_counts().to_dict()
    py=df[['participant_id','label']].drop_duplicates().label.value_counts().to_dict()
    n_part=df.participant_id.nunique()
    w=[(n_part/(2*py[int(y)]))/pc[int(pid)] for pid,y in zip(df.participant_id,df.label)]
    return WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),num_samples=len(df),replacement=True,
                                 generator=torch.Generator().manual_seed(103))

class AudioDataset(Dataset):
    def __init__(self,df,id_col): self.df=df.reset_index(drop=True); self.id_col=id_col
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; x=load_audio_segment(r.audio_path,float(r.start),float(r.stop),KD_CFG['sample_rate'])
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
        # Fixed padding is deliberate: Wav2Vec2 base group norm can otherwise make
        # a frozen embedding depend on the lengths of its batch companions.
        limit=int(KD_CFG['teacher_audio_seconds']*KD_CFG['sample_rate'])
        if any(len(v)>limit for v in x): raise ValueError('Audio exceeds configured bound')
        z=self.extractor(list(x),sampling_rate=KD_CFG['sample_rate'],padding='max_length',max_length=limit,
                         return_attention_mask=True,return_tensors='pt')
        return dict(input_values=z.input_values,attention_mask=z.attention_mask,label=torch.tensor(y,dtype=torch.float32),
                    participant_id=pid,sample_id=sid,split=split)

class TextCollator:
    def __init__(self,tokenizer,max_len=128): self.tokenizer=tokenizer; self.max_len=max_len
    def __call__(self,batch):
        x,y,pid,sid,split=zip(*batch)
        z=self.tokenizer(list(x),padding='max_length',truncation=True,max_length=self.max_len,
                         return_overflowing_tokens=True,stride=0,return_tensors='pt')
        return dict(input_ids=z.input_ids,attention_mask=z.attention_mask,label=torch.tensor(y,dtype=torch.float32),
                    window_owner=z.overflow_to_sample_mapping,n_samples=len(y),
                    participant_id=pid,sample_id=sid,split=split)

def metrics(y,p,th=.5):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=(p>=th).astype(int)
    out=dict(n=int(len(y)),accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
             recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
             macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
             balanced_accuracy=float(balanced_accuracy_score(y,pred)),threshold=float(th))
    out['auroc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')
    out['average_precision']=float(average_precision_score(y,p)) if len(np.unique(y))==2 else float('nan')
    q=np.clip(p,1e-7,1-1e-7)
    out['log_loss']=float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)))
    out['brier']=float(np.mean((p-y)**2))
    return out

def freeze_stage(model,stage):
    model.freeze_encoder()
    if stage==2: model.unfreeze_top()

def make_optimizer(model,args,steps):
    enc_ids={id(p) for p in model.encoder.parameters()}
    head=[p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    enc=[p for p in model.encoder.parameters() if p.requires_grad]
    groups=[]
    if head: groups.append({'params':head,'lr':args.head_lr})
    if enc: groups.append({'params':enc,'lr':args.encoder_lr})
    opt=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    sched=get_cosine_schedule_with_warmup(opt,num_warmup_steps=max(1,int(.05*steps)),num_training_steps=max(1,steps))
    return opt,sched

def forward_batch(model,b,modality,device):
    if 'features' in b: return model.classifier(b['features'].to(device)).squeeze(-1)
    if modality=='audio':
        return model(b['input_values'].to(device),b['attention_mask'].to(device))
    return model(b['input_ids'].to(device),b['attention_mask'].to(device),b['window_owner'].to(device),b['n_samples'])

def encode_batch(model,b,modality,device):
    if modality=='audio': return model.encode(b['input_values'].to(device),b['attention_mask'].to(device))
    return model.encode(b['input_ids'].to(device),b['attention_mask'].to(device),b['window_owner'].to(device),b['n_samples'])

def loader_for(ds,batch,args,collate=None,sampler=None):
    return DataLoader(ds,batch_size=batch,sampler=sampler,shuffle=False,collate_fn=collate,
                      num_workers=args.workers,pin_memory=torch.cuda.is_available(),
                      persistent_workers=args.workers>0,generator=torch.Generator().manual_seed(args.seed))

@torch.no_grad()
def predict(model,loader,modality,device,desc):
    model.eval(); rows=[]
    for b in tqdm(loader,desc=desc,leave=False,colour='green'):
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.startswith('cuda')):
            z=forward_batch(model,b,modality,device)
        z=z.float().cpu(); p=torch.sigmoid(z).numpy(); y=b['label'].numpy()
        for i in range(len(y)):
            rows.append(dict(participant_id=int(b['participant_id'][i]),sample_id=str(b['sample_id'][i]),split=str(b['split'][i]),
                             label=int(y[i]),logit=float(z[i].detach().cpu()),probability=float(p[i])))
    return pd.DataFrame(rows)

def score_key(m):
    return (m['macro_f1'],-m['log_loss'])

def snapshot(model,stage):
    # S1 needs only the small head: the frozen encoder is recoverable by revision.
    return {'model':(model.classifier if stage==1 else model).state_dict(),'head_only':stage==1}

def restore_weights(model,checkpoint):
    target=model.classifier if checkpoint.get('head_only') else model
    target.load_state_dict(checkpoint['model'])

def train_one(modality,train_df,dev_df,run_dir,args,device,revisions,source_identity):
    if not train_df.split.eq('train').all() or not dev_df.split.eq('dev').all():
        raise ValueError('Teacher train/dev split violation')
    if set(train_df.participant_id)&set(dev_df.participant_id): raise ValueError('Participant overlap')
    if modality=='audio':
        name=KD_CFG['teacher_audio_model']; model=AudioSBTTeacher(name,revision=revisions[name])
        prep=AutoFeatureExtractor.from_pretrained(name,revision=revisions[name])
        collate=AudioCollator(prep); batch=args.audio_batch; accum=args.audio_accum
        tr_ds=AudioDataset(train_df,'sample_id'); dv_ds=AudioDataset(dev_df,'sample_id')
    else:
        name=KD_CFG['teacher_text_model']; model=TextSBTTeacher(name,revision=revisions[name])
        prep=AutoTokenizer.from_pretrained(name,revision=revisions[name],use_fast=True)
        collate=TextCollator(prep,KD_CFG['teacher_text_tokens']); batch=args.text_batch; accum=args.text_accum
        tr_ds=TextDataset(train_df,'sample_id'); dv_ds=TextDataset(dev_df,'sample_id')
    import transformers
    implementation={p.name:digest_file(p) for p in Path(__file__).parent.glob('*.py')}
    implementation['student_data_utils']=digest_file(Path(__file__).parents[2]/'students'/'data_utils.py')
    identity=dict(code=CODE_VERSION,implementation=implementation,modality=modality,model=name,
        revision=revisions[name],sources=source_identity,config=KD_CFG,torch=torch.__version__,
        transformers=transformers.__version__,precision='fp16-autocast-to-fp32' if device.startswith('cuda') else 'fp32',
        pooling='masked-mean-audio/full-text-window-mean-cls-v1',padding='fixed')
    settings={k:v for k,v in vars(args).items() if k not in {'force_retrain','local_cache','data_root','workers'}}
    signature=digest_json(dict(identity=identity,settings=settings,train=frame_digest(train_df),dev=frame_digest(dev_df)))
    attempt=signature if not args.force_retrain else signature+'-forced-'+str(time.time_ns())
    out=run_dir/modality/'runs'/attempt; out.mkdir(parents=True,exist_ok=True)
    final=out/'best.pt'; last=out/'last.pt'; model=model.to(device)
    if final.exists():
        ck=torch.load(final,map_location='cpu',weights_only=False)
        if ck.get('signature')!=signature or not ck.get('complete'): raise ValueError('Invalid completed checkpoint')
        restore_weights(model,ck); print(f'{modality}: completed checkpoint reused: {final}')
        ck['checkpoint_path']=str(final); return model,prep,ck
    resume=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
    if resume and resume.get('signature')!=signature: raise ValueError('Resume signature mismatch')
    history=[] if resume is None else resume['history']
    best_key=(-1.,-float('inf')) if resume is None else tuple(resume['best_key'])
    best_path=None if resume is None else Path(resume['best_path'])
    criterion=torch.nn.BCEWithLogitsLoss(); amp=device.startswith('cuda')
    raw_tr=loader_for(tr_ds,batch,args,collate,participant_sampler(train_df))
    raw_dv=loader_for(dv_ds,batch,args,collate)
    persistent=run_dir.parent.parent/'sbt_frozen_features'
    for stage,limit in [(1,args.freeze_epochs),(2,args.epochs-args.freeze_epochs)]:
        if resume and (resume['stage']>stage or (resume['stage']==stage and resume['stage_finished'])): continue
        freeze_stage(model,stage)
        if stage==1:
            cached_tr=build_features(model,tr_ds,collate,train_df,identity,persistent,args.local_cache,batch,device,encode_batch)
            cached_dv=build_features(model,dv_ds,collate,dev_df,identity,persistent,args.local_cache,batch,device,encode_batch)
            tr_loader=loader_for(cached_tr,batch,args,sampler=participant_sampler(train_df))
            dv_loader=loader_for(cached_dv,batch,args)
        else:
            tr_loader,dv_loader=raw_tr,raw_dv
            if not (resume and resume['stage']==2):
                restore_weights(model,torch.load(best_path,map_location='cpu',weights_only=False))
                print(f'{modality}: S2 starts from best S1 head; frozen embeddings bypassed')
        steps=max(1,int(np.ceil(len(tr_loader)/accum)))
        opt,sched=make_optimizer(model,args,steps*limit)
        scaler=torch.amp.GradScaler('cuda',enabled=amp)
        start=1; stale=0; stage_best=(-1.,-float('inf'))
        if resume and resume['stage']==stage:
            restore_weights(model,resume); opt.load_state_dict(resume['optimizer'])
            sched.load_state_dict(resume['scheduler']); scaler.load_state_dict(resume['scaler'])
            start=resume['stage_epoch']+1; stale=resume['stale']; stage_best=tuple(resume['stage_best'])
            print(f'{modality}: resume S{stage} at epoch {start}')
        for stage_epoch in range(start,limit+1):
            epoch=len(history)+1
            # Per-epoch seeds restore sampler/dropout reproducibility after an epoch-boundary resume.
            seed_all(args.seed+(0 if modality=='audio' else 100000)+stage*1000+stage_epoch)
            tr_loader.sampler.generator.manual_seed(args.seed+stage*1000+stage_epoch)
            model.train(); model.encoder.eval()
            if stage==2:
                layers=(model.encoder.encoder.layers[-2:] if modality=='audio'
                        else model.encoder.encoder.albert_layer_groups[-1:])
                for layer in layers: layer.train()
            opt.zero_grad(set_to_none=True); running=0.; seen=0; started=time.perf_counter()
            bar=tqdm(tr_loader,desc=f'{modality} S{stage} epoch {stage_epoch:02d}/{limit}',colour='green')
            for step,b in enumerate(bar,1):
                group_start=((step-1)//accum)*accum
                group_size=min(accum,len(tr_loader)-group_start)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                    z=forward_batch(model,b,modality,device); y=b['label'].to(device)
                    raw_loss=criterion(z.float(),y.float()); loss=raw_loss/group_size
                scaler.scale(loss).backward(); running+=float(raw_loss.detach().cpu())*len(y); seen+=len(y)
                if step%accum==0 or step==len(tr_loader):
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    old_scale=scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    if scaler.get_scale()>=old_scale: sched.step()
                bar.set_postfix(loss=f'{running/max(1,seen):.4f}')
            train_seconds=time.perf_counter()-started; started=time.perf_counter()
            dev_pred=predict(model,dv_loader,modality,device,f'{modality} aligned DEV')
            m=metrics(dev_pred.label,dev_pred.probability); key=score_key(m)
            history.append(dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch,train_loss=running/max(1,seen),
                train_seconds=train_seconds,dev_seconds=time.perf_counter()-started,**m))
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(f"{modality} S{stage}/{stage_epoch} aligned DEV macro-F1={m['macro_f1']:.4f} BCE={m['log_loss']:.4f}")
            if key>stage_best: stage_best=key; stale=0
            else: stale+=1
            previous_best=None
            if key>best_key:
                previous_best=best_path
                best_key=key; best_path=out/f'best_epoch_{epoch:03d}.pt'
                ck=dict(**snapshot(model,stage),signature=signature,code_version=CODE_VERSION,model_name=name,
                    revision=revisions[name],best_epoch=epoch,stage=stage,dev_aligned_metrics=m,
                    config=vars(args),complete=False,identity=identity)
                atomic_torch(best_path,ck); dev_pred.to_csv(out/'best_dev_aligned_predictions.csv',index=False)
            patience=args.s1_patience if stage==1 else args.patience
            finished=stage_epoch==limit or stale>=patience
            atomic_torch(last,dict(signature=signature,**snapshot(model,stage),optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),scaler=scaler.state_dict(),history=history,
                stage=stage,stage_epoch=stage_epoch,stage_finished=finished,stale=stale,
                stage_best=stage_best,best_key=best_key,best_path=str(best_path)))
            # Remove only this run's superseded generated best after last.pt points to the new one.
            if previous_best is not None and previous_best!=best_path: previous_best.unlink(missing_ok=True)
            if finished: break
        resume=None
        del tr_loader,dv_loader
    ck=torch.load(best_path,map_location='cpu',weights_only=False)
    ck['complete']=True; ck['checkpoint_path']=str(final); atomic_torch(final,ck)
    restore_weights(model,ck); model.to(device)
    return model,prep,ck

def aligned_predictions(modality,model,prep,kd,run_dir,args,device,checkpoint):
    out=run_dir/modality
    export_key=digest_json({'checkpoint_sha256':digest_file(checkpoint['checkpoint_path']),
                           'manifest':frame_digest(kd),'code':CODE_VERSION})
    marker=out/'exports.json'
    try:
        saved=json.loads(marker.read_text())
        if saved['key']==export_key and all(digest_file(out/p)==sha for p,sha in saved['files'].items()):
            print(f'{modality}: verified aligned prediction cache reused')
            return saved['metrics']
    except (OSError,ValueError,KeyError): pass
    if modality=='audio':
        ds=AudioDataset(kd.rename(columns={'segment_id':'sample_id'}),'sample_id')
        loader=loader_for(ds,args.audio_batch,args,AudioCollator(prep))
    else:
        ds=TextDataset(kd.rename(columns={'segment_id':'sample_id'}),'sample_id')
        loader=loader_for(ds,args.text_batch,args,TextCollator(prep,KD_CFG['teacher_text_tokens']))
    d=predict(model,loader,modality,device,f'{modality} aligned KD outputs')
    d=d.rename(columns={'sample_id':'segment_id'}); d['prediction']=(d.probability>=.5).astype(int)
    keys=['participant_id','segment_id','split','label']
    if d.segment_id.duplicated().any() or not d[keys].reset_index(drop=True).equals(kd[keys].reset_index(drop=True)):
        raise ValueError('Aligned prediction identities/order differ from manifest')
    summary={}
    for split in ['train','dev']:
        q=d.loc[d.split.eq(split)].copy(); q.to_csv(out/f'{split}_segment_predictions.csv',index=False)
        m=metrics(q.label,q.probability); summary[split]=m
        cm=pd.DataFrame(confusion_matrix(q.label,q.prediction,labels=[0,1]),index=['actual_0','actual_1'],columns=['pred_0','pred_1'])
        cr=pd.DataFrame(classification_report(q.label,q.prediction,labels=[0,1],target_names=['non_depressed','depressed'],output_dict=True,zero_division=0)).T
        cm.to_csv(out/f'{split}_segment_confusion_matrix.csv'); cr.to_csv(out/f'{split}_segment_classification_report.csv')
    save_json(out/'metrics.json',{'aligned_segment':summary})
    files={p.name:digest_file(p) for p in out.glob('*_segment_*.csv')}
    atomic_json(marker,{'key':export_key,'files':files,'metrics':summary})
    return summary

def main(argv=None):
    args=parse_args(argv)
    from google.colab import drive
    drive.mount('/content/drive')
    seed_all(args.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    data_root=Path(args.data_root); exp_root=data_root/'experiments'
    manifest,kd,aud,txt,cache_root,revisions,source_map,source_identity=build_manifests(data_root,exp_root,args.local_cache)
    run_dir=exp_root/'teachers'/'segment_level_v1'/f'seed_{args.seed}'; run_dir.mkdir(parents=True,exist_ok=True)
    protocol=dict(name='RA-PDS-KD',teacher_family='SBT-Net unimodal adaptation',code_version=CODE_VERSION,
                  train_participants=107,dev_participants=34,test_participants_reserved=47,test_opened=False,
                  teacher_training_units={'audio':'participant speech chunks <=15 s','text':'participant text chunks <=128 ALBERT tokens'},
                  kd_output_units='existing aligned RA-PDS-KD segments <=10 s / <=254 segmentation tokens',
                  cache_root=str(cache_root),config=KD_CFG,revisions=revisions,exports_ready=False,
                  selection='aligned DEV macro-F1 at threshold 0.5; BCE tie-break',
                  cache_policy='disk shards only for S1; raw local-disk input for S2',
                  source_identity=source_identity,
                  provenance={'paper_audio':'wav2vec2.0','paper_text':'ALBERT-large',
                              'released_demo_audio':'facebook/wav2vec2-base','released_demo_text':'albert-base-v2',
                              'implemented_text':'albert-large-v2 per paper',
                              'exact_published_unimodal_heads_released':False,
                              'author_depression_checkpoint_found':False})
    save_json(run_dir/'protocol.json',protocol)
    print('Device:',device)
    print(f'Batch sizes | audio={args.audio_batch} text={args.text_batch} | grad accumulation audio={args.audio_accum} text={args.text_accum}')
    print('Embedding matrix in RAM: DISABLED. S1: disk features; S2: raw local-disk batches.')
    print('S1 resumes feature shards; training resumes at completed epoch boundaries.')
    if device=='cuda': print('GPU:',torch.cuda.get_device_name(0))
    print('Teacher samples:',{'audio_train':int((aud.split=='train').sum()),'audio_dev':int((aud.split=='dev').sum()),
          'text_train':int((txt.split=='train').sum()),'text_dev':int((txt.split=='dev').sum())})
    print('Aligned KD segments:',kd.groupby('split').size().to_dict()); print('TEST CLOSED.')

    # Original Drive paths remain in the durable manifest for downstream scripts.
    kd=kd.copy(); aud=aud.copy()
    kd['audio_path']=kd.audio_path.map(source_map); aud['audio_path']=aud.audio_path.map(source_map)
    if kd.audio_path.isna().any() or aud.audio_path.isna().any(): raise ValueError('Missing staged audio')
    summary=[]; checkpoints={}
    for modality,df in [('audio',aud),('text',txt)]:
        seed_all(args.seed+(0 if modality=='audio' else 100000))
        tr=df.loc[df.split.eq('train')].reset_index(drop=True)
        dv=kd.loc[kd.split.eq('dev')].rename(columns={'segment_id':'sample_id'}).reset_index(drop=True)
        model,prep,ck=train_one(modality,tr,dv,run_dir,args,device,revisions,source_identity)
        seg=aligned_predictions(modality,model,prep,kd,run_dir,args,device,ck)
        checkpoints[modality]={'path':ck['checkpoint_path'],'signature':ck['signature'],
                              'sha256':digest_file(ck['checkpoint_path'])}
        summary.append({'modality':modality,'best_epoch':int(ck['best_epoch']),**{f'dev_{k}':v for k,v in seg['dev'].items()}})
        del model,prep; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(summary).to_csv(run_dir/'teacher_summary.csv',index=False)
    protocol['checkpoints']=checkpoints
    protocol['prediction_sha256']={f'{modality}/{split}_segment_predictions.csv':
        digest_file(run_dir/modality/f'{split}_segment_predictions.csv')
        for modality in ('audio','text') for split in ('train','dev')}
    protocol['exports_ready']=True; atomic_json(run_dir/'protocol.json',protocol)
    print(pd.DataFrame(summary).to_string(index=False))
    print('SBT unimodal teachers ready. Next: build_reliability_targets.py. TEST REMAINS CLOSED.')

if __name__=='__main__':
    import sys
    main([] if 'ipykernel' in sys.modules else None)
