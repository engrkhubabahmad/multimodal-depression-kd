"""Export frozen Idiap targets through the same inductive path for TRAIN and DEV."""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np,pandas as pd,torch
from .idiap_source import ensure_frozen_export_source
from .train_student_kd import digest
from .train_student_kd_v2 import metrics
from scripts.teachers.idiap_text.export_targets import load_author
from scripts.students.participant_student_v3.pretrain_text import participant_doc


def predict(model,x):
    """Exact author inductive algebra: B@H0 = 2X; B@H1 = X@word_state + h_doc."""
    with torch.no_grad():
        x=torch.as_tensor(x,dtype=torch.float32)
        h=model.get_H_1(2*x)
        z=model.node_emb2out(x@model.H_1_words[:x.shape[1]].cpu()+h)
        index=list(model.classes_).index('positive')
        other=1-index
        return (z[:,index]-z[:,other]).numpy(),torch.softmax(z,dim=1)[:,index].numpy()


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True,type=Path);a=p.parse_args(argv)
    exp=a.experiment;out=exp/'teachers/idiap_text_inductive_v1';out.mkdir(parents=True,exist_ok=True)
    src=ensure_frozen_export_source();ckpt=src/'model/Participant/model_inductgcn[250].pkl'
    vtfile=src/'model/Participant/vtzer_inductgcn[250].pkl'
    with vtfile.open('rb') as f:vt=pickle.load(f)
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    author=load_author(src/'main.py');author.DEVICE=torch.device('cpu')
    model=author.InducTGCN(state['embedding_dim'],state['classes_'],0,vt)
    model.load_state_dict(dict(state['model_state_dict']));model.classes_=state['classes_'];model.cpu().eval()
    split=pd.read_csv(exp/'split/manifest.csv');coverage=pd.read_csv(exp/'coverage/participant_manifest.csv')
    if split.split.value_counts().to_dict()!={'train':107,'val':34,'student_test':47}:raise ValueError('Wrong canonical split')
    if len(coverage)!=141 or coverage.participant_id.duplicated().any():raise ValueError('Wrong TRAIN/DEV coverage')
    audit={'teacher_checkpoint_sha256':digest(ckpt),'vectorizer_sha256':digest(vtfile),
           'train_and_dev_inference':'author inductive document prediction using frozen saved word state',
           'teacher_weights_updated':False,'train_in_sample':True,'test_opened':False,
           'split_sha256':digest(exp/'split/manifest.csv'),'source_transcript_sha256':{}}
    for name in ('train','val'):
        wanted=split.loc[split.split.eq(name),['participant_id','label']].sort_values('participant_id')
        frame=wanted.merge(coverage[['participant_id','label','transcript_path']],on=['participant_id','label'],validate='one_to_one')
        if len(frame)!=len(wanted):raise ValueError('Missing transcript coverage')
        docs=[participant_doc(Path(path)) for path in frame.transcript_path]
        if not all(docs):raise ValueError('Empty participant transcript')
        x=vt.transform(docs).toarray().astype('float32');logits,prob=predict(model,x)
        # Verify against the author's forward method, with its batch cache cleared.
        model.Conv_0_Test=None
        with torch.no_grad():reference=model.forward(docs).cpu().numpy()[:,list(model.classes_).index('positive')]
        if not np.allclose(prob,reference,atol=1e-6,rtol=1e-5):raise ValueError('Inductive algebra differs from author forward')
        frame[['participant_id','label']].assign(text_probability=prob,text_logit=logits).to_csv(out/f'{name}_targets.csv',index=False)
        audit[name]=metrics(frame.label.to_numpy(),prob)
        audit[name]['probability_std']=float(prob.std())
        audit['source_transcript_sha256'].update({str(pid):digest(path) for pid,path in zip(frame.participant_id,frame.transcript_path)})
        if name=='val':
            mapping=pd.read_csv(src/'data/AVEC_16_data/dev_IDS.txt',sep='\t').original_ID.astype(int).tolist()
            indices=[mapping.index(int(pid)) for pid in frame.participant_id]
            original_x=state['A_dev'].cpu()[indices,:x.shape[1]].numpy()
            _,oldprob=predict(model,original_x)
            audit['dev_reconstruction']={'tfidf_max_abs_difference':float(abs(x-original_x).max()),
                'probability_max_abs_difference':float(abs(prob-oldprob).max()),
                'matches_saved_dev_inputs':bool(np.allclose(x,original_x,atol=1e-6,rtol=1e-5)),
                'saved_dev_input_metrics':metrics(frame.label.to_numpy(),oldprob)}
        print(f'Frozen Idiap {name}:',audit[name],flush=True)
    audit['note']='Old TRAIN export used normalized graph training scores. This export uses the same author inference formula on TRAIN and DEV. A preprocessing mismatch, if reported, prevents claiming exact reproduction of the old DEV score.'
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('DEV reconstruction:',audit['dev_reconstruction'],flush=True)
    print('New inference-consistent targets:',out,flush=True)

if __name__=='__main__':main()
