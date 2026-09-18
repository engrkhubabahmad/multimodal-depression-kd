"""Train the three RA-PDS-KD segment-level ReLiMP-Net student variants.

1) no_kd: hard labels, clean input only
2) standard_kd: hard labels + equal dual-teacher KD, clean input only
3) ra_robust_kd: reliability-aware KD + clean/missing/noisy TRAIN conditions

All three are evaluated on DEV under clean + four robustness conditions.
Checkpoint and threshold selection use CLEAN DEV only. TEST is never opened here.
"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import argparse,json,random,sys,gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,
                             roc_auc_score,average_precision_score,confusion_matrix,classification_report)

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.students.relimpnet_segment import ReLiMPNetSegment,parameter_count
from scripts.kd.reliability_loss import standard_kd_loss,reliability_aware_kd_loss

CONDITIONS=['clean','audio_missing','text_missing','audio_noisy','text_noisy']
MODES=['no_kd','standard_kd','ra_robust_kd']


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=103); p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--patience',type=int,default=20); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--temperature',type=float,default=2.0); p.add_argument('--lambda-kd',type=float,default=0.5)
    p.add_argument('--final-mode',choices=MODES,default='ra_robust_kd')
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_features(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}


def load_targets(path,segment_ids):
    d=pd.read_csv(path); assert not d.segment_id.duplicated().any()
    d=d.set_index('segment_id').loc[list(segment_ids)].reset_index()
    assert d.segment_id.astype(str).tolist()==[str(x) for x in segment_ids]
    return d


class SegmentDataset(Dataset):
    def __init__(self,features,targets,conditions,audio_noisy_quality,text_noisy_quality):
        self.f=features; self.t=targets; self.conditions=list(conditions); self.aq=float(audio_noisy_quality); self.tq=float(text_noisy_quality)
    def __len__(self): return len(self.f['labels'])*len(self.conditions)
    def __getitem__(self,index):
        i=index//len(self.conditions); cond=self.conditions[index%len(self.conditions)]
        a=self.f['audio_clean'][i].copy(); t=self.f['text_clean'][i].copy(); aq=tq=1.0
        if cond=='audio_missing': a[:]=0; aq=0.0
        elif cond=='text_missing': t[:]=0; tq=0.0
        elif cond=='audio_noisy': a=self.f['audio_noisy'][i].copy(); aq=self.aq
        elif cond=='text_noisy': t=self.f['text_noisy'][i].copy(); tq=self.tq
        r=self.t.iloc[i]
        return {'audio':torch.from_numpy(a).float(),'text':torch.from_numpy(t).long(),'label':torch.tensor(float(self.f['labels'][i])),
                'audio_probability':torch.tensor(float(r.audio_probability)),'text_probability':torch.tensor(float(r.text_probability)),
                'audio_confidence':torch.tensor(float(r.audio_confidence)),'text_confidence':torch.tensor(float(r.text_confidence)),
                'audio_quality':torch.tensor(aq),'text_quality':torch.tensor(tq),'segment_id':str(self.f['segment_ids'][i]),
                'participant_id':int(self.f['participant_ids'][i]),'condition':cond}


def sampler_weights(features,conditions):
    pid=features['participant_ids'].astype(int); y=features['labels'].astype(int)
    pcounts=pd.Series(pid).value_counts().to_dict(); ccounts=np.bincount(y,minlength=2)
    cw={c:len(y)/(2*max(1,ccounts[c])) for c in [0,1]}
    base=np.array([cw[int(label)]/pcounts[int(p)] for p,label in zip(pid,y)],dtype=np.float64)
    return np.repeat(base,len(conditions))


def make_loader(features,targets,conditions,batch_size,audio_q,text_q,train=False):
    ds=SegmentDataset(features,targets,conditions,audio_q,text_q)
    if train:
        w=sampler_weights(features,conditions); sampler=WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(w),replacement=True)
        return DataLoader(ds,batch_size=batch_size,sampler=sampler,num_workers=0)
    return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0)


def predict(model,loader,device):
    model.eval(); rows=[]
    with torch.inference_mode():
        for b in loader:
            aq=b['audio_quality'].to(device); tq=b['text_quality'].to(device)
            z=model(b['audio'].to(device),b['text'].to(device),aq,tq); p=torch.sigmoid(z).cpu().numpy()
            for i,prob in enumerate(p):
                rows.append({'segment_id':b['segment_id'][i],'participant_id':int(b['participant_id'][i]),'label':int(b['label'][i]),
                             'condition':b['condition'][i],'probability':float(prob)})
    return pd.DataFrame(rows)


def metrics(df,threshold):
    y=df.label.to_numpy(int); p=df.probability.to_numpy(float); pred=(p>=threshold).astype(int)
    return {'n_segments':int(len(df)),'accuracy':float(accuracy_score(y,pred)),
            'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'depressed_f1':float(f1_score(y,pred,zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),'threshold':float(threshold)}


def best_threshold(df):
    best=None
    for th in np.arange(0.10,0.901,0.01):
        m=metrics(df,float(th)); key=(m['macro_f1'],m['depressed_f1'],m['balanced_accuracy'])
        if best is None or key>best[0]: best=(key,float(th),m)
    return best[1],best[2]


def save_eval(df,threshold,out,condition):
    d=df.loc[df.condition.eq(condition)].copy(); d['prediction']=(d.probability>=threshold).astype(int)
    m=metrics(d,threshold); d.to_csv(out/f'dev_{condition}_predictions.csv',index=False)
    cm=pd.DataFrame(confusion_matrix(d.label,d.prediction,labels=[0,1]),index=['actual_0','actual_1'],columns=['pred_0','pred_1'])
    cr=pd.DataFrame(classification_report(d.label,d.prediction,labels=[0,1],target_names=['non_depressed','depressed'],output_dict=True,zero_division=0)).T
    cm.to_csv(out/f'dev_{condition}_confusion_matrix.csv'); cr.to_csv(out/f'dev_{condition}_classification_report.csv')
    return m


def main():
    args=parse_args(); seed_all(args.seed); device='cuda' if torch.cuda.is_available() else 'cpu'
    data_root=Path('/content/drive/MyDrive/DAIC_WOZ'); feat_root=data_root/'processed'/'rapdskd_student'/f'seed_{args.seed}'
    kd_root=data_root/'processed'/'rapdskd_kd'/f'seed_{args.seed}'; out_root=data_root/'experiments'/'students'/'segment_level_v1'/f'seed_{args.seed}'
    out_root.mkdir(parents=True,exist_ok=True)
    summary=json.loads((feat_root/'feature_summary.json').read_text()); vocab=json.loads((feat_root/'vocab.json').read_text())
    train=load_features(feat_root/'train_features.npz'); dev=load_features(feat_root/'dev_features.npz')
    tr_targets=load_targets(kd_root/'train_teacher_targets.csv',train['segment_ids']); dv_targets=load_targets(kd_root/'dev_teacher_targets.csv',dev['segment_ids'])
    audio_q=summary['audio_noisy_quality']; text_q=summary['text_noisy_quality']
    comparison=[]
    for mode in MODES:
        print('\n'+'='*90+f'\nTRAINING {mode}\n'+'='*90); seed_all(args.seed)
        robust=mode=='ra_robust_kd'; model=ReLiMPNetSegment(len(vocab),use_reliability_fusion=robust).to(device)
        print('Device:',device,'| Params:',parameter_count(model))
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        train_conditions=CONDITIONS if robust else ['clean']
        train_loader=make_loader(train,tr_targets,train_conditions,args.batch_size,audio_q,text_q,True)
        clean_dev_loader=make_loader(dev,dv_targets,['clean'],args.batch_size,audio_q,text_q,False)
        out=out_root/mode; out.mkdir(parents=True,exist_ok=True)
        best_score=-1; best_epoch=0; stale=0; history=[]; best_path=out/'best.pt'
        for epoch in range(1,args.epochs+1):
            model.train(); total=0.0; n=0
            for b in train_loader:
                a=b['audio'].to(device); t=b['text'].to(device); y=b['label'].to(device); aq=b['audio_quality'].to(device); tq=b['text_quality'].to(device)
                z=model(a,t,aq,tq)
                if mode=='no_kd': loss=torch.nn.functional.binary_cross_entropy_with_logits(z.float(),y.float())
                elif mode=='standard_kd':
                    loss,_=standard_kd_loss(z,y,b['audio_probability'].to(device),b['text_probability'].to(device),args.temperature,args.lambda_kd)
                else:
                    loss,_=reliability_aware_kd_loss(z,y,b['audio_probability'].to(device),b['text_probability'].to(device),aq,tq,
                                                      b['audio_confidence'].to(device),b['text_confidence'].to(device),args.temperature,args.lambda_kd)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                total+=float(loss.detach().cpu())*len(y); n+=len(y)
            clean_df=predict(model,clean_dev_loader,device); th,m=best_threshold(clean_df); score=m['macro_f1']
            row={'epoch':epoch,'train_loss':total/max(1,n),'dev_clean_threshold':th,**{f'dev_clean_{k}':v for k,v in m.items() if k!='threshold'}}
            history.append(row); pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} clean_macroF1={score:.4f} depF1={m['depressed_f1']:.4f} th={th:.2f}")
            if score>best_score+1e-8:
                best_score=score; best_epoch=epoch; stale=0
                torch.save({'model':model.state_dict(),'mode':mode,'epoch':epoch,'threshold':th,'vocab_size':len(vocab),
                            'use_reliability_fusion':robust,'student_config':{'audio_dim':128,'d_model':128},
                            'training_config':vars(args),'audio_noisy_quality':audio_q,'text_noisy_quality':text_q},best_path)
            else:
                stale+=1
                if stale>=args.patience: print('Early stopping; best epoch:',best_epoch); break
        state=torch.load(best_path,map_location='cpu',weights_only=False); model.load_state_dict(state['model']); threshold=float(state['threshold'])
        all_dev_loader=make_loader(dev,dv_targets,CONDITIONS,args.batch_size,audio_q,text_q,False); all_df=predict(model,all_dev_loader,device)
        robustness=[]
        for condition in CONDITIONS:
            m=save_eval(all_df,threshold,out,condition); robustness.append({'mode':mode,'condition':condition,**m})
        pd.DataFrame(robustness).to_csv(out/'dev_robustness_metrics.csv',index=False)
        clean_m=next(x for x in robustness if x['condition']=='clean')
        comparison.append({'mode':mode,'best_epoch':best_epoch,'threshold':threshold,**{f'clean_{k}':v for k,v in clean_m.items() if k not in {'mode','condition'}}})
        (out/'metrics.json').write_text(json.dumps({'best_epoch':best_epoch,'threshold':threshold,'dev_robustness':robustness},indent=2))
        del model,opt; gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(comparison).to_csv(out_root/'student_comparison_clean_dev.csv',index=False)
    chosen=out_root/args.final_mode/'best.pt'; assert chosen.exists()
    state=torch.load(chosen,map_location='cpu',weights_only=False)
    selection={'protocol':'RA-PDS-KD','selected_mode':args.final_mode,'checkpoint':str(chosen),'threshold':float(state['threshold']),
               'selection_data':'DEV only','checkpoint_selection_condition':'clean DEV only',
               'robustness_conditions':'DEV only: clean/audio_missing/text_missing/audio_noisy/text_noisy',
               'test_opened':False,'final_test_condition':'clean only'}
    (out_root/'final_selection.json').write_text(json.dumps(selection,indent=2))
    print('\n',json.dumps(selection,indent=2)); print('TEST REMAINS CLOSED. Run final clean TEST evaluator only after accepting this frozen selection.')

if __name__=='__main__': main()
