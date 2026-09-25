"""Frozen unimodal students plus a tiny train-only fusion gate; no teacher fine-tuning."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from torch import nn
from scipy.special import expit
from sklearn.metrics import classification_report
from .student_temporal_v1 import inputs,Student,DeepStudent
from .train_student_kd import digest
from .train_student_kd_v2 import metrics
from .expert_gate import gate_features,fuse,fit


class TextExpert(nn.Module):
    def __init__(self,model):
        super().__init__();self.encoder=copy.deepcopy(model.text);self.head=copy.deepcopy(model.head)
    def forward(self,x):
        h=self.encoder(x);mask=x.new_tensor([[1.,0.]]).expand(len(x),-1)
        return self.head(torch.cat([h,torch.zeros_like(h),mask],1)).squeeze(1)


class AudioExpert(nn.Module):
    def __init__(self,model):
        super().__init__();self.encoder=copy.deepcopy(model.audio);self.head=copy.deepcopy(model.head)
    def forward(self,x):
        h=self.encoder(x);mask=x.new_tensor([[0.,1.]]).expand(len(x),-1)
        return self.head(torch.cat([torch.zeros_like(h),h,mask],1)).squeeze(1)


def infer(experts,text,audio,device):
    with torch.inference_mode():
        t=experts[0](torch.as_tensor(text,dtype=torch.float32,device=device)).cpu().numpy()
        a=experts[1](torch.as_tensor(audio,dtype=torch.float32,device=device)).cpu().numpy()
    return np.column_stack([t,a]).astype(float)


def stability(experts,text,audio,device,scale,seed,repeats=5):
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repeats):
        nt=(text+rng.normal(0,.05,text.shape)).astype('float32')
        na=(audio+rng.normal(0,.05,audio.shape)).astype('float32')
        samples.append(infer(experts,nt,na,device))
    return 1/(1+np.std(samples,axis=0)/np.maximum(scale,1e-4))


def verify_predictions(directory,ids,y,p):
    f=pd.read_csv(directory/'dev_clean_predictions.csv').set_index('participant_id')
    if len(f)!=len(ids) or f.index.duplicated().any() or set(f.index)!=set(ids):raise ValueError('Saved DEV IDs mismatch')
    f=f.loc[ids]
    if not np.array_equal(y,f.label.to_numpy(int)):raise ValueError('Saved DEV labels mismatch')
    delta=float(np.max(abs(p-f.probability.to_numpy())))
    if delta>2e-5:raise ValueError(f'{directory}: checkpoint/preprocessing reproduction failed: max probability delta={delta}')
    return delta


def evaluate(y,p):
    m=metrics(y,p)
    m['classification_report']=classification_report(y,np.asarray(p)>=.5,labels=[0,1],
        target_names=['Non-depressed','Depressed'],output_dict=True,zero_division=0)
    return m


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seed',type=int,default=103)
    p.add_argument('--text-target',choices=['inductive','author_docs'],default='inductive')
    a=p.parse_args(argv);exp=a.experiment
    base=exp/'students/canonical_v1';textdir=base/'temporal_text_deep_v1/text_only'
    audiodir=base/'temporal_text_v1/audio_only'
    teacher_dir=exp/'teachers'/('idiap_text_author_docs_v1' if a.text_target=='author_docs' else 'idiap_text_inductive_v1')
    for file in [textdir/'best.pt',audiodir/'best.pt',teacher_dir/'train_targets.csv',teacher_dir/'audit.json']:
        if not file.is_file():raise FileNotFoundError(f'Required completed stage missing: {file}')
    for parent in [textdir.parent,audiodir.parent]:
        protocol=json.loads((parent/'protocol.json').read_text())
        if protocol['seed']!=a.seed or protocol['split_sha256']!=digest(exp/'split/manifest.csv'):
            raise ValueError('Existing expert checkpoint belongs to a different seed/split')
    data,provenance,preprocessor=inputs(exp,a.seed,return_preprocessor=True)
    tr,va=data['train'],data['val'];dim=tr['text'].shape[1]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    deep=DeepStudent(dim);compact=Student(dim)
    deep.load_state_dict(torch.load(textdir/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    compact.load_state_dict(torch.load(audiodir/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    experts=[TextExpert(deep).to(device).eval(),AudioExpert(compact).to(device).eval()]
    for expert in experts:
        for param in expert.parameters():param.requires_grad_(False)
    lt=infer(experts,tr['text'],tr['audio'],device);lv=infer(experts,va['text'],va['audio'],device)
    pt,pv=expit(lt),expit(lv);scale=np.maximum(lt.std(axis=0),1e-4)
    reproduction={'text_probability_max_difference':verify_predictions(textdir,va['ids'],va['y'],pv[:,0]),
                  'audio_probability_max_difference':verify_predictions(audiodir,va['ids'],va['y'],pv[:,1])}
    print('Restored expert predictions verified:',reproduction,flush=True)
    train_stability=stability(experts,tr['text'],tr['audio'],device,scale,a.seed)
    val_stability=stability(experts,va['text'],va['audio'],device,scale,a.seed+1)
    ft,fv=gate_features(lt,train_stability,scale),gate_features(lv,val_stability,scale)
    audio_target=exp/'teachers/nusd_ecapa_frozen/evaluation/train_audio_targets.csv'
    targets=pd.read_csv(teacher_dir/'train_targets.csv').merge(pd.read_csv(audio_target),on=['participant_id','label'],validate='one_to_one')
    if len(targets)!=107 or targets.participant_id.duplicated().any():raise ValueError('Invalid teacher targets')
    targets=targets.set_index('participant_id').loc[tr['ids']]
    if not np.array_equal(targets.label.to_numpy(int),tr['y']):raise ValueError('Teacher label mismatch')
    teacher=targets[['text_probability','audio_probability']].to_numpy(float)
    if not np.isfinite(teacher).all() or np.any((teacher<0)|(teacher>1)):raise ValueError('Invalid teacher probabilities')
    selected=json.loads((exp/'teachers/nusd_ecapa_frozen/evaluation/all_five/selection.json').read_text())['selected']['run']
    if json.loads((audio_target.parent/'train_audio_audit.json').read_text())['selected_run']!=selected:
        raise ValueError('Audio teacher checkpoint mismatch')
    teacher_audit=json.loads((teacher_dir/'audit.json').read_text())
    if teacher_audit['split_sha256']!=digest(exp/'split/manifest.csv') or teacher_audit['test_opened']:
        raise ValueError('Text teacher target provenance does not match split')
    if a.text_target=='author_docs' and teacher_audit['dev_checkpoint_tfidf_max_abs_difference']>1e-6:
        raise ValueError('Author DEV features did not reproduce')
    output=base/('preserved_experts_author_text_v1' if a.text_target=='author_docs' else 'preserved_experts_v1')
    output.mkdir(parents=True,exist_ok=True)
    # Persist preprocessing and pruned expert weights for later reproducible deployment/evaluation.
    joblib.dump(preprocessor,output/'preprocessing.joblib')
    for name,expert in zip(('text','audio'),experts):
        torch.save({'state_dict':expert.cpu().state_dict(),'text_dim':dim},output/f'{name}_expert.pt');expert.to(device)
    models={'text_only':None,'audio_only':None,'equal_fusion':np.zeros(3)};fits={}
    for mode in ('plain','standard_kd','ra_kd'):
        history=[]
        def log(step,theta,loss):
            tm=metrics(tr['y'],fuse(theta,ft,pt));vm=metrics(va['y'],fuse(theta,fv,pv))
            history.append({'step':step,'train_loss':loss,'train_macro_f1':tm['macro_f1'],'val_macro_f1':vm['macro_f1']})
            print(f'{mode} Step {step:03d}/100 - loss: {loss:.6f} - train_macro_f1: {tm["macro_f1"]:.4f} - val_macro_f1: {vm["macro_f1"]:.4f}',flush=True)
        models[mode],fits[mode]=fit(ft,pt,tr['y'],teacher,mode,callback=log)
        pd.DataFrame(history).to_csv(output/f'{mode}_history.csv',index=False)
    prior=float(tr['y'].mean());rows=[]
    # Noise is feature-space Gaussian perturbation, not raw-waveform or transcript corruption.
    noise_inputs={}
    for sd in (.1,.3):
        for modality in ('text','audio','both'):
            print(f'Feature-noise evaluation: {modality}, sigma={sd}, 5 repeats',flush=True)
            for repeat in range(5):
                rng=np.random.default_rng(a.seed+repeat+int(sd*1000))
                t=va['text'].copy();au=va['audio'].copy()
                if modality in ('text','both'):t+=(rng.normal(0,sd,t.shape)).astype('float32')
                if modality in ('audio','both'):au+=(rng.normal(0,sd,au.shape)).astype('float32')
                l=infer(experts,t,au,device)
                stab=stability(experts,t,au,device,scale,a.seed+repeat+2,repeats=3)
                noise_inputs[(sd,modality,repeat)]=(expit(l),gate_features(l,stab,scale))
    for name,theta in models.items():
        folder=output/name;folder.mkdir(parents=True,exist_ok=True)
        def predict(prob,features,mask=None):
            if name=='text_only':return prob[:,0]
            if name=='audio_only':return prob[:,1]
            return fuse(theta,features,prob,mask,prior)
        train_prob=predict(pt,ft);val_prob=predict(pv,fv)
        report={'train':evaluate(tr['y'],train_prob),'dev':evaluate(va['y'],val_prob)}
        for split,d,q in [('train',tr,train_prob),('dev',va,val_prob)]:
            pd.DataFrame({'participant_id':d['ids'],'label':d['y'],'probability':q,'prediction':q>=.5}).to_csv(folder/f'{split}_predictions.csv',index=False)
        if theta is not None:
            for scenario,mask in [('missing_text',[0,1]),('missing_audio',[1,0]),('missing_both',[0,0])]:
                prob=predict(pv,fv,np.tile(mask,(len(pv),1)))
                report[scenario]=evaluate(va['y'],prob)
                pd.DataFrame({'participant_id':va['ids'],'label':va['y'],'probability':prob,'prediction':prob>=.5}).to_csv(folder/f'dev_{scenario}_predictions.csv',index=False)
            if not np.allclose(predict(pv,fv,np.tile([1,0],(len(pv),1))),pv[:,0],atol=1e-12):raise AssertionError('Text fallback broken')
            if not np.allclose(predict(pv,fv,np.tile([0,1],(len(pv),1))),pv[:,1],atol=1e-12):raise AssertionError('Audio fallback broken')
            (folder/'gate.json').write_text(json.dumps({'theta':theta.tolist(),'logit_scale':scale.tolist(),'prior':prior,'fit':fits.get(name)},indent=2)+'\n')
        noise=[]
        for (sd,modality,repeat),(prob,features) in noise_inputs.items():
            m=metrics(va['y'],predict(prob,features))
            noise.append({'sd':sd,'modality':modality,'repeat':repeat,**{k:m[k] for k in ['macro_f1','depressed_f1','auroc']}})
        pd.DataFrame(noise).to_csv(folder/'dev_feature_noise.csv',index=False)
        (folder/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        rows.append({'mode':name,'train_macro_f1':report['train']['macro_f1'],
            **{f'dev_{k}':report['dev'][k] for k in ['accuracy','macro_f1','depressed_f1','auroc','predicted_positive']},
            'missing_audio_macro_f1':report.get('missing_audio',{}).get('macro_f1'),
            'missing_text_macro_f1':report.get('missing_text',{}).get('macro_f1')})
        print(name,'DEV:',report['dev'],flush=True)
    pd.DataFrame(rows).to_csv(output/'comparison.csv',index=False)
    provenance.update({'architecture':'pruned deep text expert and compact temporal audio expert, frozen; three-parameter probability fusion gate',
        'source_checkpoint_sha256':{str(d/'best.pt'):digest(d/'best.pt') for d in [textdir,audiodir]},
        'teacher_targets_sha256':{str(f):digest(f) for f in [teacher_dir/'train_targets.csv',audio_target]},
        'teacher_inference_audit':teacher_audit,'text_target_source':a.text_target,
        'reproduction':reproduction,'gate_parameters':3,'gate_selection':'TRAIN objective only',
        'teacher_weights_updated':False,'expert_weights_updated':False,'out_of_fold':False,
        'limitation':'Expert checkpoints were selected using DEV in earlier experiments. Overall DEV results remain exploratory. Gate fits TRAIN in-sample predictions; no generalization guarantee.',
        'noise_protocol':'Gaussian noise in preprocessed feature space; sigma .1/.3; 5 repeats; no raw-input robustness claim',
        'test_opened':False})
    (output/'protocol.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False));print('Saved:',output/'comparison.csv')

if __name__=='__main__':main()
