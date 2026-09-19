from __future__ import annotations
from pathlib import Path
import argparse,gc,hashlib,json,random
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

CODE_VERSION='sbt-unimodal-v1.3'

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=103); p.add_argument('--epochs',type=int,default=85)
    p.add_argument('--freeze-epochs',type=int,default=25); p.add_argument('--patience',type=int,default=10)
    p.add_argument('--audio-batch',type=int,default=16); p.add_argument('--text-batch',type=int,default=16)
    p.add_argument('--audio-accum',type=int,default=1); p.add_argument('--text-accum',type=int,default=1)
    p.add_argument('--head-lr',type=float,default=2e-5); p.add_argument('--encoder-lr',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=1e-2); p.add_argument('--force-retrain',action='store_true')
    return p.parse_known_args()[0]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def participant_sampler(df):
    pc=df.participant_id.value_counts().to_dict()
    py=df[['participant_id','label']].drop_duplicates().label.value_counts().to_dict()
    n_part=df.participant_id.nunique()
    w=[(n_part/(2*py[int(y)]))/pc[int(pid)] for pid,y in zip(df.participant_id,df.label)]
    return WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),num_samples=len(df),replacement=True)

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
        z=self.extractor(list(x),sampling_rate=KD_CFG['sample_rate'],padding=True,return_attention_mask=True,return_tensors='pt')
        return dict(input_values=z.input_values,attention_mask=z.attention_mask,label=torch.tensor(y,dtype=torch.float32),
                    participant_id=pid,sample_id=sid,split=split)

class TextCollator:
    def __init__(self,tokenizer,max_len=128): self.tokenizer=tokenizer; self.max_len=max_len
    def __call__(self,batch):
        x,y,pid,sid,split=zip(*batch)
        z=self.tokenizer(list(x),padding=True,truncation=True,max_length=self.max_len,return_tensors='pt')
        return dict(input_ids=z.input_ids,attention_mask=z.attention_mask,label=torch.tensor(y,dtype=torch.float32),
                    participant_id=pid,sample_id=sid,split=split)

def metrics(y,p,th=.5):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=(p>=th).astype(int)
    out=dict(n=int(len(y)),accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
             recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
             macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
             balanced_accuracy=float(balanced_accuracy_score(y,pred)),threshold=float(th))
    out['auroc']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')
    out['average_precision']=float(average_precision_score(y,p)) if len(np.unique(y))==2 else float('nan')
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
    if modality=='audio':
        return model(b['input_values'].to(device),b['attention_mask'].to(device))
    return model(b['input_ids'].to(device),b['attention_mask'].to(device))

@torch.no_grad()
def predict(model,loader,modality,device,desc):
    model.eval(); rows=[]
    for b in tqdm(loader,desc=desc,leave=False):
        z=forward_batch(model,b,modality,device); p=torch.sigmoid(z).cpu().numpy(); y=b['label'].numpy()
        for i in range(len(y)):
            rows.append(dict(participant_id=int(b['participant_id'][i]),sample_id=str(b['sample_id'][i]),split=str(b['split'][i]),
                             label=int(y[i]),logit=float(z[i].detach().cpu()),probability=float(p[i])))
    return pd.DataFrame(rows)

def train_one(modality,train_df,dev_df,run_dir,args,device,revisions):
    out=run_dir/modality; out.mkdir(parents=True,exist_ok=True)
    if modality=='audio':
        name=KD_CFG['teacher_audio_model']; model=AudioSBTTeacher(name)
        prep=AutoFeatureExtractor.from_pretrained(name,revision=revisions[name])
        collate=AudioCollator(prep); batch=args.audio_batch; accum=args.audio_accum
        tr_ds=AudioDataset(train_df,'sample_id'); dv_ds=AudioDataset(dev_df,'sample_id')
    else:
        name=KD_CFG['teacher_text_model']; model=TextSBTTeacher(name)
        prep=AutoTokenizer.from_pretrained(name,revision=revisions[name])
        collate=TextCollator(prep,KD_CFG['teacher_text_tokens']); batch=args.text_batch; accum=args.text_accum
        tr_ds=TextDataset(train_df,'sample_id'); dv_ds=TextDataset(dev_df,'sample_id')

    signature=hashlib.sha256(json.dumps(dict(code=CODE_VERSION,modality=modality,model=name,revision=revisions[name],
        epochs=args.epochs,freeze_epochs=args.freeze_epochs,head_lr=args.head_lr,encoder_lr=args.encoder_lr,
        audio_batch=args.audio_batch,text_batch=args.text_batch,audio_accum=args.audio_accum,text_accum=args.text_accum,
        train_ids=sorted(train_df.participant_id.unique().tolist()),dev_ids=sorted(dev_df.participant_id.unique().tolist())),sort_keys=True).encode()).hexdigest()
    best=out/'best.pt'; model=model.to(device)
    if best.exists() and not args.force_retrain:
        ck=torch.load(best,map_location='cpu',weights_only=False)
        if ck.get('signature')==signature and ck.get('complete') is True:
            model.load_state_dict(ck['model']); model.to(device)
            print(f'{modality}: valid checkpoint cache -> {best}')
            return model,prep,ck

    tr_loader=DataLoader(tr_ds,batch_size=batch,sampler=participant_sampler(train_df),collate_fn=collate,num_workers=0)
    dv_loader=DataLoader(dv_ds,batch_size=batch,shuffle=False,collate_fn=collate,num_workers=0)
    criterion=torch.nn.BCEWithLogitsLoss(); amp=device.startswith('cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    freeze_stage(model,1)
    steps_per_epoch=max(1,int(np.ceil(len(tr_loader)/accum)))
    opt,sched=make_optimizer(model,args,steps_per_epoch*max(1,args.freeze_epochs))
    best_score=-1.; stale=0; history=[]; stage=1

    for epoch in range(1,args.epochs+1):
        if epoch==args.freeze_epochs+1:
            stage=2; freeze_stage(model,2); remain=max(1,args.epochs-args.freeze_epochs)
            opt,sched=make_optimizer(model,args,steps_per_epoch*remain); stale=0
            print(f'{modality}: stage 2 -> top encoder parameters unfrozen')
        model.train();
        if stage==1: model.encoder.eval()
        opt.zero_grad(set_to_none=True); running=0.; seen=0
        bar=tqdm(tr_loader,desc=f'{modality} epoch {epoch:02d}/{args.epochs} S{stage}')
        for step,b in enumerate(bar,1):
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                z=forward_batch(model,b,modality,device); y=b['label'].to(device)
                loss=criterion(z.float(),y.float())/accum
            scaler.scale(loss).backward(); running+=float(loss.detach().cpu())*accum*len(y); seen+=len(y)
            if step%accum==0 or step==len(tr_loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            bar.set_postfix(loss=f'{running/max(1,seen):.4f}')

        dev_pred=predict(model,dv_loader,modality,device,f'{modality} DEV')
        m=metrics(dev_pred.label,dev_pred.probability)
        history.append(dict(epoch=epoch,stage=stage,train_loss=running/max(1,seen),**m))
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(f"{modality} epoch={epoch:02d} DEV macro-F1={m['macro_f1']:.4f} F1={m['f1']:.4f} AUROC={m['auroc']:.4f}")

        if m['macro_f1']>best_score+1e-8:
            best_score=m['macro_f1']; stale=0
            ck=dict(model=model.state_dict(),signature=signature,code_version=CODE_VERSION,model_name=name,
                    revision=revisions[name],best_epoch=epoch,stage=stage,dev_teacher_sample_metrics=m,config=vars(args),complete=False)
            torch.save(ck,best); dev_pred.to_csv(out/'best_dev_teacher_sample_predictions.csv',index=False)
        elif stage==2:
            stale+=1
            if stale>=args.patience:
                print(f'{modality}: early stop after stage-2 patience={args.patience}')
                break

    ck=torch.load(best,map_location='cpu',weights_only=False); ck['complete']=True; torch.save(ck,best)
    model.load_state_dict(ck['model']); model.to(device)
    return model,prep,ck

def aligned_predictions(modality,model,prep,kd,run_dir,args,device):
    out=run_dir/modality
    if modality=='audio':
        ds=AudioDataset(kd.rename(columns={'segment_id':'sample_id'}),'sample_id')
        loader=DataLoader(ds,batch_size=args.audio_batch,shuffle=False,collate_fn=AudioCollator(prep),num_workers=0)
    else:
        ds=TextDataset(kd.rename(columns={'segment_id':'sample_id'}),'sample_id')
        loader=DataLoader(ds,batch_size=args.text_batch,shuffle=False,collate_fn=TextCollator(prep,KD_CFG['teacher_text_tokens']),num_workers=0)
    d=predict(model,loader,modality,device,f'{modality} aligned KD outputs')
    d=d.rename(columns={'sample_id':'segment_id'}); d['prediction']=(d.probability>=.5).astype(int)
    summary={}
    for split in ['train','dev']:
        q=d.loc[d.split.eq(split)].copy(); q.to_csv(out/f'{split}_segment_predictions.csv',index=False)
        m=metrics(q.label,q.probability); summary[split]=m
        cm=pd.DataFrame(confusion_matrix(q.label,q.prediction,labels=[0,1]),index=['actual_0','actual_1'],columns=['pred_0','pred_1'])
        cr=pd.DataFrame(classification_report(q.label,q.prediction,labels=[0,1],target_names=['non_depressed','depressed'],output_dict=True,zero_division=0)).T
        cm.to_csv(out/f'{split}_segment_confusion_matrix.csv'); cr.to_csv(out/f'{split}_segment_classification_report.csv')
    save_json(out/'metrics.json',{'aligned_segment':summary})
    return summary

def main():
    from google.colab import drive
    drive.mount('/content/drive')
    args=parse_args(); seed_all(args.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    data_root=Path('/content/drive/MyDrive/DAIC_WOZ'); exp_root=data_root/'experiments'
    manifest,kd,aud,txt,cache_root,revisions=build_manifests(data_root,exp_root)
    run_dir=exp_root/'teachers'/'segment_level_v1'/f'seed_{args.seed}'; run_dir.mkdir(parents=True,exist_ok=True)
    protocol=dict(name='RA-PDS-KD',teacher_family='SBT-Net unimodal adaptation',code_version=CODE_VERSION,
                  train_participants=107,dev_participants=34,test_participants_reserved=47,test_opened=False,
                  teacher_training_units={'audio':'participant speech chunks <=15 s','text':'participant text chunks <=128 ALBERT tokens'},
                  kd_output_units='existing aligned RA-PDS-KD segments <=10 s / <=254 segmentation tokens',
                  cache_root=str(cache_root),config=KD_CFG,revisions=revisions,
                  provenance={'paper_audio':'wav2vec2.0','paper_text':'ALBERT-large',
                              'released_demo_audio':'facebook/wav2vec2-base','released_demo_text':'albert-base-v2',
                              'implemented_text':'albert-large-v2 per paper',
                              'exact_published_unimodal_heads_released':False,
                              'author_depression_checkpoint_found':False})
    save_json(run_dir/'protocol.json',protocol)
    print('Device:',device)
    print(f'Batch sizes | audio={args.audio_batch} text={args.text_batch} | grad accumulation audio={args.audio_accum} text={args.text_accum}')
    print('Embedding cache in Colab RAM: DISABLED; samples are loaded batch-by-batch from source files.')
    print('Teacher samples:',{'audio_train':int((aud.split=='train').sum()),'audio_dev':int((aud.split=='dev').sum()),
          'text_train':int((txt.split=='train').sum()),'text_dev':int((txt.split=='dev').sum())})
    print('Aligned KD segments:',kd.groupby('split').size().to_dict()); print('TEST CLOSED.')

    summary=[]
    for modality,df in [('audio',aud),('text',txt)]:
        tr=df.loc[df.split.eq('train')].reset_index(drop=True); dv=df.loc[df.split.eq('dev')].reset_index(drop=True)
        model,prep,ck=train_one(modality,tr,dv,run_dir,args,device,revisions)
        seg=aligned_predictions(modality,model,prep,kd,run_dir,args,device)
        summary.append({'modality':modality,'best_epoch':int(ck['best_epoch']),**{f'dev_{k}':v for k,v in seg['dev'].items()}})
        del model,prep; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    pd.DataFrame(summary).to_csv(run_dir/'teacher_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print('SBT unimodal teachers ready. Next: build_reliability_targets.py. TEST REMAINS CLOSED.')

if __name__=='__main__': main()
