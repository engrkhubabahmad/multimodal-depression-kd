"""Pre-registered experimental rank-preserving KD with label-free modality reliability."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from .train_student_kd import load_inputs,digest
from .train_student_kd_v2 import metrics

class MaskedStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.text=nn.Linear(16,8);self.audio=nn.Linear(16,8)
        self.head=nn.Sequential(nn.ReLU(),nn.Dropout(.2),nn.Linear(18,1))
    def forward(self,t,a,mask):
        z=torch.cat([self.text(t*mask[:,0:1]),self.audio(a*mask[:,1:2]),mask],dim=1)
        return self.head(z).squeeze(1)

def teacher_targets(exp,ids,y):
    t=exp/'teachers/idiap_text_frozen/train_text_kd_targets.csv'
    a=exp/'teachers/nusd_ecapa_frozen/evaluation/train_audio_targets.csv'
    z=pd.read_csv(t).merge(pd.read_csv(a),on=['participant_id','label'],validate='one_to_one')
    if len(z)!=107 or z.participant_id.duplicated().any():raise ValueError('Bad KD join')
    z=z.set_index('participant_id').loc[ids]
    if not np.array_equal(z.label.to_numpy(int),y):raise ValueError('KD label mismatch')
    raw=z[['text_logit','audio_logit']].to_numpy(float)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite teacher logits')
    center=np.median(raw,axis=0);spread=raw.std(axis=0)
    if np.any(spread<1e-4):raise ValueError('Teacher logit has no meaningful variation')
    scaled=np.clip((raw-center)/spread,-3,3)
    soft=1/(1+np.exp(-scaled))
    confidence=np.maximum(abs(soft-.5),.05)
    standard=soft.mean(axis=1)
    ra=(confidence*soft).sum(axis=1)/confidence.sum(axis=1)
    return soft.astype('float32'),confidence.astype('float32'),{
        'text_file':str(t),'audio_file':str(a),'source_sha256':{str(t):digest(t),str(a):digest(a)},
        'logit_medians':center.tolist(),'logit_train_std':spread.tolist(),
        'soft_target_absolute_difference_mean':float(abs(standard-ra).mean()),
        'teacher_disagreement':int(np.sum((soft[:,0]>=.5)!=(soft[:,1]>=.5)))}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True,type=Path)
    p.add_argument('--seed',type=int,default=103);p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--patience',type=int,default=25);a=p.parse_args(argv)
    exp=a.experiment;arr,_,textdir=load_inputs(exp)
    (ids,y,xt,xa),(dids,dy,dt,da)=arr['train'],arr['val']
    selection=exp/'teachers/nusd_ecapa_frozen/evaluation/all_five/selection.json'
    selected=json.loads(selection.read_text())['selected']['run']
    audit=json.loads((exp/'teachers/nusd_ecapa_frozen/evaluation/train_audio_audit.json').read_text())
    if selected!=audit.get('selected_run'):raise ValueError('Audio TRAIN targets do not match selected checkpoint')
    teacher,confidence,tinfo=teacher_targets(exp,ids,y)
    for x,d,name in ((xt,dt,'text'),(xa,da,'audio')):
        if not np.isfinite(x).all() or not np.isfinite(d).all():raise ValueError(f'{name} feature nonfinite')
    def compact(x,d):
        scaler=StandardScaler().fit(x);x=np.clip(scaler.transform(x),-5,5);d=np.clip(scaler.transform(d),-5,5)
        pca=PCA(n_components=16,random_state=a.seed).fit(x)
        return pca.transform(x).astype('float32'),pca.transform(d).astype('float32')
    xt,dt=compact(xt,dt);xa,da=compact(xa,da)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tx,ax,yy,soft,conf=[torch.tensor(v,device=device) for v in
        (xt,xa,y.astype('float32'),teacher,confidence)]
    td,ad=[torch.tensor(v,device=device) for v in (dt,da)]
    pos_weight=torch.tensor([(y==0).sum()/max(1,(y==1).sum())],dtype=torch.float32,device=device)
    out=exp/'students/canonical_v1/matched_kd_v3';out.mkdir(parents=True,exist_ok=True)
    rows=[];full=torch.ones((len(dy),2),device=device)
    for mode in ('plain','standard_kd','ra_kd'):
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        model=MaskedStudent().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
        folder=out/mode;folder.mkdir(parents=True,exist_ok=True)
        best=(-1.,-1.,-1.);bad=0;history=[]
        bar=tqdm(range(1,a.epochs+1),desc=f'v3 {mode}',unit='epoch')
        for epoch in bar:
            # Shared random stream: each mode sees the same mask/noise schedule.
            draw=torch.rand(len(y),device=device)
            mask=torch.ones((len(y),2),device=device)
            mask[draw<.25,0]=0;mask[(draw>=.25)&(draw<.5),1]=0
            nt=tx+.1*torch.randn_like(tx);na=ax+.1*torch.randn_like(ax)
            model.train();opt.zero_grad(set_to_none=True)
            logits=model(nt,na,mask)
            hard=nn.functional.binary_cross_entropy_with_logits(logits,yy,pos_weight=pos_weight)
            if mode=='plain':loss=hard
            else:
                weights=torch.ones_like(conf) if mode=='standard_kd' else conf*mask
                target=(weights*soft).sum(1)/weights.sum(1).clamp_min(1e-8)
                loss=.7*hard+.3*nn.functional.binary_cross_entropy_with_logits(logits,target.detach())
            loss.backward();opt.step()
            model.eval()
            with torch.no_grad():pval=torch.sigmoid(model(td,ad,full)).cpu().numpy()
            m=metrics(dy,pval);key=(m['macro_f1'],m['depressed_f1'],m['auroc'])
            history.append({'epoch':epoch,'loss':float(loss.item()),'dev_macro_f1':m['macro_f1'],
                            'dev_depressed_f1':m['depressed_f1'],'dev_auroc':m['auroc'],
                            'dev_predicted_positive':m['predicted_positive']})
            bar.set_postfix(dev_f1=f"{m['macro_f1']:.3f}",positives=m['predicted_positive'])
            if key>best:
                best=key;bad=0;torch.save({'state_dict':model.state_dict(),'epoch':epoch},folder/'best.pt')
            else:bad+=1
            if bad>=a.patience:break
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        state=torch.load(folder/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(state['state_dict']);model.eval()
        scenarios={'clean':(torch.ones_like(full),td,ad),
            'missing_text':(torch.tensor([[0.,1.]],device=device).repeat(len(dy),1),td,ad),
            'missing_audio':(torch.tensor([[1.,0.]],device=device).repeat(len(dy),1),td,ad)}
        results={}
        for name,(mask,t,aa) in scenarios.items():
            with torch.no_grad():prob=torch.sigmoid(model(t,aa,mask)).cpu().numpy()
            result=metrics(dy,prob);results[name]=result
            pd.DataFrame({'participant_id':dids,'label':dy,'probability':prob,
                'prediction':(prob>=.5).astype(int)}).to_csv(folder/f'dev_{name}_predictions.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps({'best_epoch':state['epoch'],'scenarios':results},indent=2)+'\n')
        rows.append({'mode':mode,'epoch':state['epoch'],**{f'clean_{k}':v for k,v in results['clean'].items() if k!='confusion_matrix'},
             'missing_text_macro_f1':results['missing_text']['macro_f1'],
             'missing_audio_macro_f1':results['missing_audio']['macro_f1']})
        print(mode,'clean',results['clean'],'missing text',results['missing_text'],
              'missing audio',results['missing_audio'],flush=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'seed':a.seed,'split':'TRAIN-107/DEV-34; TEST closed',
        'teacher':tinfo,'audio_run':selected,'text_student_checkpoint_sha256':digest(textdir/'best.pt'),
        'feature_preprocessing':'TRAIN-only StandardScaler and PCA-16 for each modality',
        'kd':'TRAIN-only median-centering and standard-deviation scaling of each teacher logit; T=1; KD loss weight 0.3',
        'ra_reliability':'label-free normalized teacher confidence with modality availability',
        'training_masks':'50% both, 25% missing text, 25% missing audio; Gaussian sigma .1, same schedule across modes',
        'selection':'best clean DEV macro F1; tie depressed F1 then AUROC',
        'warning':'exploratory DEV selection; logit normalization is not probability calibration',
        'test_opened':False},indent=2)+'\n')
    print('Saved:',out/'comparison.csv')

if __name__=='__main__':main()
