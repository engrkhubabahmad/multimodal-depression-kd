"""Fresh transcript and temporal-audio students on canonical TRAIN/DEV; no TEST access."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from .train_student_kd import digest
from .train_student_kd_v2 import metrics
from .train_student_kd_v3 import teacher_targets
from scripts.students.participant_student_v3.pretrain_text import participant_doc


def temporal_array(path,bins=32):
    x=np.load(path,mmap_mode='r',allow_pickle=False)
    if x.ndim!=2 or x.shape[0]!=130 or x.shape[1]<bins:raise ValueError(f'Invalid 130-channel audio: {path} {x.shape}')
    edges=np.linspace(0,x.shape[1],bins+1,dtype=int)
    pooled=np.stack([np.asarray(x[:,edges[i]:edges[i+1]],dtype=np.float32).mean(axis=1)
                     for i in range(bins)],axis=1)
    if not np.isfinite(pooled).all():raise ValueError(f'Nonfinite audio features: {path}')
    return pooled


def inputs(exp,seed):
    split=pd.read_csv(exp/'split/manifest.csv')
    if split.split.value_counts().to_dict()!={'train':107,'val':34,'student_test':47}:
        raise ValueError('Expected canonical TRAIN-107/DEV-34/TEST-47')
    coverage=pd.read_csv(exp/'coverage/participant_manifest.csv')
    if len(coverage)!=141 or coverage.participant_id.duplicated().any():raise ValueError('Coverage mismatch')
    data={}; paths={}
    for name in ('train','val'):
        frame=split[split.split.eq(name)].sort_values('participant_id')[['participant_id','label']]
        rows=frame.merge(coverage[['participant_id','label','transcript_path','feature_path']],
                         on=['participant_id','label'],validate='one_to_one')
        if len(rows)!=len(frame) or not np.array_equal(rows.participant_id,frame.participant_id):
            raise ValueError(f'{name} participant/label coverage mismatch')
        if 440 in set(rows.participant_id):raise ValueError('Excluded participant present')
        docs=[participant_doc(Path(p)) for p in rows.transcript_path]
        if not all(docs):raise ValueError(f'Empty {name} transcript')
        audio=[]
        for i,path in enumerate(rows.feature_path,1):
            audio.append(temporal_array(path))
            if i%20==0 or i==len(rows):print(f'{name} audio preprocessing {i}/{len(rows)}',flush=True)
        data[name]={'ids':rows.participant_id.to_numpy(int),'y':rows.label.to_numpy(int),
                    'docs':docs,'audio':np.stack(audio).astype('float32')}
        paths[name]=rows[['participant_id','transcript_path','feature_path']].to_dict('records')
    vectorizer=TfidfVectorizer(stop_words='english',min_df=2,max_features=1000,sublinear_tf=True)
    train_sparse=vectorizer.fit_transform(data['train']['docs'])
    svd=TruncatedSVD(n_components=min(32,train_sparse.shape[1]-1,train_sparse.shape[0]-1),random_state=seed)
    text_scaler=StandardScaler().fit(svd.fit_transform(train_sparse))
    audio_scaler=StandardScaler().fit(data['train']['audio'].transpose(0,2,1).reshape(-1,130))
    for name in ('train','val'):
        z=svd.transform(vectorizer.transform(data[name]['docs']))
        data[name]['text']=np.clip(text_scaler.transform(z),-5,5).astype('float32')
        a=data[name]['audio'];b=a.transpose(0,2,1).reshape(-1,130)
        data[name]['audio']=np.clip(audio_scaler.transform(b).reshape(len(a),32,130).transpose(0,2,1),-5,5).astype('float32')
    provenance={'train_vocab_size':len(vectorizer.vocabulary_),'text_dim':data['train']['text'].shape[1],
        'audio_shape':list(data['train']['audio'].shape[1:]),'fit':'TRAIN only',
        'coverage_sha256':digest(exp/'coverage/participant_manifest.csv'),
        'split_sha256':digest(exp/'split/manifest.csv'),
        'source_files':paths,'test_opened':False}
    return data,provenance


class Student(nn.Module):
    def __init__(self,text_dim):
        super().__init__()
        self.text=nn.Sequential(nn.Linear(text_dim,16),nn.ReLU(),nn.Dropout(.25))
        self.audio=nn.Sequential(nn.Conv1d(130,24,5,padding=2),nn.ReLU(),
            nn.Conv1d(24,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(34,16),nn.ReLU(),nn.Dropout(.25),nn.Linear(16,1))
    def forward(self,t,a,mask):
        # Mask the raw input before each branch. Bias terms cannot leak a missing view.
        h=torch.cat([self.text(t*mask[:,0:1])*mask[:,0:1],
                     self.audio(a*mask[:,1:2,None])*mask[:,1:2],mask],dim=1)
        return self.head(h).squeeze(1)


class ResidualAudio(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv1d(24,24,3,padding=dilation,dilation=dilation),
            nn.GroupNorm(4,24),nn.ReLU(),nn.Dropout(.3),
            nn.Conv1d(24,24,3,padding=1),nn.GroupNorm(4,24))
    def forward(self,x):
        return torch.relu(x+self.layers(x))


class DeepStudent(nn.Module):
    def __init__(self,text_dim):
        super().__init__()
        self.text=nn.Sequential(nn.Linear(text_dim,48),nn.LayerNorm(48),nn.ReLU(),
            nn.Dropout(.35),nn.Linear(48,24),nn.ReLU())
        self.audio=nn.Sequential(nn.Conv1d(130,24,5,padding=2),nn.GroupNorm(4,24),
            nn.ReLU(),ResidualAudio(2),ResidualAudio(4))
        self.audio_out=nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Dropout(.35))
        self.head=nn.Sequential(nn.Linear(50,32),nn.ReLU(),nn.Dropout(.4),
            nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,t,a,mask):
        text=self.text(t*mask[:,0:1])*mask[:,0:1]
        h=self.audio(a*mask[:,1:2,None])
        audio=self.audio_out(torch.cat([h.mean(-1),h.amax(-1)],dim=1))*mask[:,1:2]
        return self.head(torch.cat([text,audio,mask],dim=1)).squeeze(1)


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True,type=Path)
    ap.add_argument('--seed',type=int,default=103);ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--patience',type=int,default=20)
    ap.add_argument('--architecture',choices=['compact','deep'],default='compact');a=ap.parse_args(argv)
    if a.epochs<1 or a.patience<1:raise ValueError('Invalid training length')
    exp=a.experiment;data,provenance=inputs(exp,a.seed)
    tr,dv=data['train'],data['val'];y,vy=tr['y'],dv['y']
    selected=json.loads((exp/'teachers/nusd_ecapa_frozen/evaluation/all_five/selection.json').read_text())['selected']['run']
    audio_audit=json.loads((exp/'teachers/nusd_ecapa_frozen/evaluation/train_audio_audit.json').read_text())
    if selected!=audio_audit.get('selected_run'):raise ValueError('Audio target checkpoint mismatch')
    soft,confidence,tinfo=teacher_targets(exp,tr['ids'],y)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tt,ta,yy,ss,cc=[torch.tensor(z,device=device) for z in (tr['text'],tr['audio'],y.astype('float32'),soft,confidence)]
    vt,va=[torch.tensor(z,device=device) for z in (dv['text'],dv['audio'])]
    positive_weight=torch.tensor([(y==0).sum()/max(1,(y==1).sum())],device=device,dtype=torch.float32)
    output=exp/'students/canonical_v1'/('temporal_text_deep_v1' if a.architecture=='deep' else 'temporal_text_v1')
    output.mkdir(parents=True,exist_ok=True)
    summary=[];modes=('text_only','audio_only','plain','standard_kd','ra_kd')
    for mode in modes:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        model=(DeepStudent(tt.shape[1]) if a.architecture=='deep' else Student(tt.shape[1])).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4 if a.architecture=='deep' else 5e-4,
                              weight_decay=.05 if a.architecture=='deep' else .02)
        folder=output/mode;folder.mkdir(parents=True,exist_ok=True)
        best=(-1.,-1.,-1.);stale=0;history=[]
        for epoch in range(1,a.epochs+1):
            # Separate generator ensures identical sample masks/noise across modes.
            generator=torch.Generator(device=device).manual_seed(a.seed*10000+epoch)
            mask=torch.ones((len(y),2),device=device)
            if mode=='text_only':mask[:,1]=0
            elif mode=='audio_only':mask[:,0]=0
            else:
                draw=torch.rand(len(y),device=device,generator=generator)
                mask[draw<.25,0]=0;mask[(draw>=.25)&(draw<.5),1]=0
            noisy_t=tt+.05*torch.randn(tt.shape,device=device,generator=generator)
            noisy_a=ta+.05*torch.randn(ta.shape,device=device,generator=generator)
            model.train();opt.zero_grad(set_to_none=True)
            z=model(noisy_t,noisy_a,mask)
            hard=nn.functional.binary_cross_entropy_with_logits(z,yy,pos_weight=positive_weight)
            if mode in ('standard_kd','ra_kd'):
                weights=torch.ones_like(cc) if mode=='standard_kd' else cc*mask
                target=(weights*ss).sum(1)/weights.sum(1).clamp_min(1e-8)
                loss=.7*hard+.3*nn.functional.binary_cross_entropy_with_logits(z,target.detach())
            else:loss=hard
            loss.backward()
            if a.architecture=='deep':nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            model.eval()
            valmask=torch.ones((len(vy),2),device=device)
            if mode=='text_only':valmask[:,1]=0
            if mode=='audio_only':valmask[:,0]=0
            with torch.no_grad():
                pt=torch.sigmoid(model(tt,ta,torch.ones_like(mask) if mode not in ('text_only','audio_only') else mask)).cpu().numpy()
                pv=torch.sigmoid(model(vt,va,valmask)).cpu().numpy()
            train_m,val_m=metrics(y,pt),metrics(vy,pv)
            key=(val_m['macro_f1'],val_m['depressed_f1'],val_m['auroc'])
            history.append({'epoch':epoch,'loss':float(loss.item()),'train_macro_f1':train_m['macro_f1'],
                            'val_macro_f1':val_m['macro_f1'],'val_depressed_f1':val_m['depressed_f1'],
                            'val_auroc':val_m['auroc'],'val_predicted_positive':val_m['predicted_positive']})
            print(f'{mode} Epoch {epoch:03d}/{a.epochs} - loss: {loss.item():.4f} - train_macro_f1: {train_m["macro_f1"]:.4f} - val_macro_f1: {val_m["macro_f1"]:.4f} - val_depressed_f1: {val_m["depressed_f1"]:.4f} - val_auroc: {val_m["auroc"]:.4f} - val_positive: {val_m["predicted_positive"]}',flush=True)
            if key>best:
                best=key;stale=0;torch.save({'state_dict':model.state_dict(),'epoch':epoch},folder/'best.pt')
            else:stale+=1
            if stale>=a.patience:
                print(f'{mode} early stopping at epoch {epoch}; best checkpoint saved',flush=True);break
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        state=torch.load(folder/'best.pt',map_location=device,weights_only=True);model.load_state_dict(state['state_dict']);model.eval()
        results={};scenarios={'clean':(1.,1.),'missing_text':(0.,1.),'missing_audio':(1.,0.)}
        if mode in ('text_only','audio_only'):scenarios={'clean':(1.,0.) if mode=='text_only' else (0.,1.)}
        for name,(tm,am) in scenarios.items():
            mask=torch.tensor([[tm,am]],device=device).repeat(len(vy),1)
            with torch.no_grad():pval=torch.sigmoid(model(vt,va,mask)).cpu().numpy()
            results[name]=metrics(vy,pval)
            pd.DataFrame({'participant_id':dv['ids'],'label':vy,'probability':pval,
                 'prediction':(pval>=.5).astype(int)}).to_csv(folder/f'dev_{name}_predictions.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps({'best_epoch':state['epoch'],'scenarios':results},indent=2)+'\n')
        summary.append({'mode':mode,'epoch':state['epoch'],**{f'val_{k}':v for k,v in results['clean'].items() if k!='confusion_matrix'},
            'missing_text_macro_f1':results.get('missing_text',{}).get('macro_f1'),
            'missing_audio_macro_f1':results.get('missing_audio',{}).get('macro_f1')})
        print(f'{mode} BEST epoch={state["epoch"]} VAL {results["clean"]}',flush=True)
    pd.DataFrame(summary).to_csv(output/'comparison.csv',index=False)
    provenance.update({'seed':a.seed,'selected_audio_run':selected,'teacher':tinfo,
        'student_architecture':a.architecture,
        'architecture':'TRAIN transcript TF-IDF/SVD32; temporal Conv1d over 32 bins of ComParE16',
        'modes':modes,'parameters':sum(p.numel() for p in model.parameters()),
        'mask_schedule':'multimodal: 50% both, 25% missing text, 25% missing audio',
        'training_noise_sd':.05,'selection':'clean DEV macro F1, tie depressed F1 then AUROC',
        'kd_loss_weight':.3,'ra_reliability':'label-free normalized confidence and availability',
        'warning':'DEV selection exploratory; teacher logit normalization is rank preserving, not calibration',
        'test_opened':False})
    (output/'protocol.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Comparison saved:',output/'comparison.csv',flush=True)

if __name__=='__main__':main()
