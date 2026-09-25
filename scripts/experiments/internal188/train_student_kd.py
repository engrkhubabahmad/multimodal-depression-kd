"""Matched multimodal student comparison on canonical TRAIN-107/DEV-34."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    z=(np.asarray(p)>=.5).astype(int)
    return {'n':len(y),'accuracy':float(accuracy_score(y,z)),
            'macro_f1':float(f1_score(y,z,average='macro',zero_division=0)),
            'depressed_f1':float(f1_score(y,z,pos_label=1,zero_division=0)),
            'auroc':float(roc_auc_score(y,p)),
            'confusion_matrix':confusion_matrix(y,z,labels=[0,1]).tolist()}


def load_inputs(exp):
    split=pd.read_csv(exp/'split/manifest.csv')
    if split.split.value_counts().to_dict()!={'train':107,'val':34,'student_test':47}:
        raise ValueError('Wrong canonical split')
    manifest=pd.read_csv(exp/'students/canonical_v1/inputs/participant_manifest.csv')
    if len(manifest)!=141 or manifest.participant_id.duplicated().any():raise ValueError('Invalid participant features')
    expected=split[split.split.isin(['train','val'])].sort_values('participant_id')
    actual=manifest.sort_values('participant_id')
    if not np.array_equal(expected.participant_id.to_numpy(),actual.participant_id.to_numpy()):
        raise ValueError('Student input IDs differ from split')
    if not np.array_equal(expected.label.to_numpy(),actual.label.to_numpy()):raise ValueError('Label mismatch')
    textdir=exp/'students/canonical_v1/plain_text'
    arrays={}
    for name,file in [('train','train_text_embeddings.npz'),('val','dev_text_embeddings.npz')]:
        with np.load(textdir/file,allow_pickle=False) as z:
            ids=z['participant_ids'].astype(int); labels=z['labels'].astype(int); emb=z['embedding'].astype('float32')
        wanted=split[split.split.eq(name)].sort_values('participant_id')
        if set(ids)!=set(wanted.participant_id) or len(ids)!=len(wanted):raise ValueError(f'{name}: text ID coverage')
        order=pd.Series(np.arange(len(ids)),index=ids).loc[wanted.participant_id].to_numpy()
        if not np.array_equal(labels[order],wanted.label.to_numpy()):raise ValueError('Text label mismatch')
        frame=manifest.set_index('participant_id').loc[wanted.participant_id]
        audio=[]
        for path in tqdm(frame.feature_path,desc=f'{name} audio summaries',unit='participant'):
            z=np.load(path,mmap_mode='r',allow_pickle=False)
            if z.ndim!=2 or z.shape[0]!=130 or z.shape[1]<1:raise ValueError(f'Unexpected ComParE16 array: {path}')
            audio.append(np.concatenate([np.mean(z,axis=1),np.std(z,axis=1)]))
        arrays[name]=(wanted.participant_id.to_numpy(int),wanted.label.to_numpy(int),emb[order],np.asarray(audio,dtype='float32'))
    return arrays, split, textdir


class Student(nn.Module):
    def __init__(self,t,a):
        super().__init__()
        self.text=nn.Linear(t,16);self.audio=nn.Linear(a,16)
        self.head=nn.Sequential(nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
    def forward(self,t,a):
        return self.head(torch.cat([self.text(t),self.audio(a)],dim=1)).squeeze(1)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seed',type=int,default=103);p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=25);p.add_argument('--lr',type=float,default=1e-3)
    a=p.parse_args(argv);exp=a.experiment
    if a.epochs<1 or a.patience<1:raise ValueError('Invalid epochs/patience')
    arr,split,textdir=load_inputs(exp)
    (ids,y,xt,xa),(dids,dy,dt,da)=arr['train'],arr['val']
    selection=exp/'teachers/nusd_ecapa_frozen/evaluation/all_five/selection.json'
    selected=json.loads(selection.read_text())['selected']['run']
    audit=json.loads((exp/'teachers/nusd_ecapa_frozen/evaluation/train_audio_audit.json').read_text())
    if selected!=audit.get('selected_run'):raise ValueError('Audio TRAIN targets do not match selected DEV checkpoint')
    tp=exp/'teachers/idiap_text_frozen/train_text_kd_targets.csv'
    ap=exp/'teachers/nusd_ecapa_frozen/evaluation/train_audio_targets.csv'
    teachers=pd.read_csv(tp).merge(pd.read_csv(ap),on=['participant_id','label'],validate='one_to_one')
    if len(teachers)!=107 or teachers.participant_id.duplicated().any():raise ValueError('Invalid teacher join')
    teachers=teachers.set_index('participant_id').loc[ids]
    if not np.array_equal(y,teachers.label.to_numpy(int)):raise ValueError('KD labels differ')
    teacher_logits=teachers[['text_logit','audio_logit']].to_numpy('float32')
    if not np.isfinite(teacher_logits).all():raise ValueError('Nonfinite teacher logits')
    st,sa=StandardScaler().fit(xt),StandardScaler().fit(xa)
    xt=np.clip(st.transform(xt),-5,5).astype('float32');dt=np.clip(st.transform(dt),-5,5).astype('float32')
    xa=np.clip(sa.transform(xa),-5,5).astype('float32');da=np.clip(sa.transform(da),-5,5).astype('float32')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tr=[torch.tensor(v,device=device) for v in (xt,xa,y.astype('float32'),teacher_logits)]
    dv=[torch.tensor(v,device=device) for v in (dt,da)]
    out=exp/'students/canonical_v1/matched_kd_v1';out.mkdir(parents=True,exist_ok=True)
    rows=[]; T=2.0
    # The same architecture, feature scaling, seed and DEV selection apply in all three modes.
    for mode in ('plain','standard_kd','ra_kd'):
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        model=Student(xt.shape[1],xa.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-2)
        yz=tr[2].unsqueeze(1)
        soft=torch.sigmoid(tr[3]/T)
        if mode=='standard_kd':weights=torch.full_like(soft,.5)
        else:
            # TRAIN-only reliability: teacher probability assigned to the true class.
            reliability=torch.sigmoid(tr[3])*yz+(1-torch.sigmoid(tr[3]))*(1-yz)
            weights=reliability/(reliability.sum(1,keepdim=True)+1e-8)
        target=(weights*soft).sum(1).detach()
        best=(-1.,-1.,-1.);bad=0;history=[];folder=out/mode;folder.mkdir(parents=True,exist_ok=True)
        bar=tqdm(range(1,a.epochs+1),desc=mode,unit='epoch')
        for epoch in bar:
            model.train();opt.zero_grad(set_to_none=True)
            z=model(tr[0],tr[1]);hard=nn.functional.binary_cross_entropy_with_logits(z,tr[2])
            loss=hard if mode=='plain' else .5*hard+.5*T*T*nn.functional.binary_cross_entropy_with_logits(z/T,target)
            loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                train_p=torch.sigmoid(model(tr[0],tr[1])).cpu().numpy()
                dev_p=torch.sigmoid(model(*dv)).cpu().numpy()
            train_m=metrics(y,train_p);dev_m=metrics(dy,dev_p)
            key=(dev_m['macro_f1'],dev_m['depressed_f1'],dev_m['auroc'])
            history.append({'epoch':epoch,'loss':float(loss.item()),'train_macro_f1':train_m['macro_f1'],
                            'dev_macro_f1':dev_m['macro_f1'],'dev_depressed_f1':dev_m['depressed_f1'],'dev_auroc':dev_m['auroc']})
            bar.set_postfix(train_f1=f"{train_m['macro_f1']:.3f}",dev_f1=f"{dev_m['macro_f1']:.3f}")
            if key>best:
                best=key;bad=0
                torch.save({'state_dict':model.state_dict(),'epoch':epoch},folder/'best.pt')
            else:bad+=1
            if bad>=a.patience:break
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        state=torch.load(folder/'best.pt',map_location=device,weights_only=True);model.load_state_dict(state['state_dict']);model.eval()
        with torch.no_grad():
            train_p=torch.sigmoid(model(tr[0],tr[1])).cpu().numpy();dev_p=torch.sigmoid(model(*dv)).cpu().numpy()
        for name,pid,labels,probs in [('train',ids,y,train_p),('dev',dids,dy,dev_p)]:
            pd.DataFrame({'participant_id':pid,'label':labels,'probability':probs,
                          'prediction':(probs>=.5).astype(int)}).to_csv(folder/f'{name}_predictions.csv',index=False)
        info={'mode':mode,'best_epoch':state['epoch'],'train':metrics(y,train_p),'dev':metrics(dy,dev_p)}
        (folder/'metrics.json').write_text(json.dumps(info,indent=2)+'\n')
        rows.append({'mode':mode,'epoch':state['epoch'],**{f'dev_{k}':v for k,v in info['dev'].items() if k not in ('confusion_matrix',)}})
        print(mode,'TRAIN',info['train'],'DEV',info['dev'],flush=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'split':'canonical TRAIN-107/DEV-34; TEST unopened',
        'student_features':'cached plain-text branch 64d participant embeddings and ComParE16 130-channel mean/std',
        'teacher_text':str(tp),'teacher_audio':str(ap),'selected_audio_run':selected,
        'source_sha256':{str(v):digest(v) for v in [tp,ap,selection,textdir/'best.pt']},
        'scalers_fit':'TRAIN only','temperature':T,'kd_loss_weight':.5,
        'ra_reliability':'TRAIN label-conditioned true-class teacher probability normalized across two modalities',
        'selection':'max DEV macro F1; tie depressed F1, AUROC; same seed and architecture',
        'limitation':'DEV model selection is exploratory; no held-out TEST evaluation'},indent=2)+'\n')
    print('Comparison saved:',out/'comparison.csv',flush=True)

if __name__=='__main__':main()
