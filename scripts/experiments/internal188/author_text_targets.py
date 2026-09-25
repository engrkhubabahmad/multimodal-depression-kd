"""Reproduce pinned Idiap documents and export matched TRAIN/DEV teacher targets."""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np,pandas as pd,torch
from .audit_idiap_inference import predict
from .idiap_source import ensure_participant_documents
from .train_student_kd import digest
from .train_student_kd_v2 import metrics
from scripts.teachers.idiap_text.export_targets import load_author


def author_rows(root,split):
    base=root/'data/AVEC_16_data'
    ids=pd.read_csv(base/f'{split}_IDS.txt',sep='\t')
    lines=(base/f'{split}_Participant.txt').read_text().splitlines()
    if len(ids)!=len(lines) or ids.original_ID.duplicated().any():
        raise ValueError(f'Author {split} document/ID row count mismatch')
    labels=[];docs=[]
    for line in lines:
        label,separator,doc=line.partition('\t')
        if not separator or label not in ('positive','negative') or not doc.strip():
            raise ValueError(f'Invalid author {split} document')
        labels.append(int(label=='positive'));docs.append(doc)
    return ids.original_ID.astype(int).to_numpy(),np.asarray(labels),docs,ids


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True,type=Path)
    a=ap.parse_args(argv);exp=a.experiment;src=ensure_participant_documents()
    ckpt=src/'model/Participant/model_inductgcn[250].pkl'
    vec=src/'model/Participant/vtzer_inductgcn[250].pkl'
    with vec.open('rb') as file:vectorizer=pickle.load(file)
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    author=load_author(src/'main.py');author.DEVICE=torch.device('cpu')
    model=author.InducTGCN(state['embedding_dim'],state['classes_'],0,vectorizer)
    model.load_state_dict(dict(state['model_state_dict']));model.classes_=state['classes_'];model.cpu().eval()
    manifest=pd.read_csv(exp/'split/manifest.csv')
    if manifest.split.value_counts().to_dict()!={'train':107,'val':34,'student_test':47}:
        raise ValueError('Incorrect canonical split')
    result={};vectors={}
    for author_split,ours in [('train','train'),('dev','val')]:
        ids,labels,docs,_=author_rows(src,author_split)
        expected=manifest[manifest.split.eq(ours)][['participant_id','label']].sort_values('participant_id')
        lookup={int(pid):i for i,pid in enumerate(ids)}
        if set(expected.participant_id)-set(lookup):raise ValueError(f'Missing {ours} author document')
        ix=[lookup[int(pid)] for pid in expected.participant_id]
        if not np.array_equal(labels[ix],expected.label.to_numpy(int)):
            raise ValueError(f'{ours} author labels do not match canonical labels')
        x=vectorizer.transform([docs[i] for i in ix]).toarray().astype('float32')
        vectors[ours]=x
        logits,prob=predict(model,x)
        result[ours]=(expected.assign(text_probability=prob,text_logit=logits),ix)
    original=state['A_dev'].cpu().numpy()[result['val'][1],:vectors['val'].shape[1]]
    delta=np.max(np.abs(vectors['val']-original),axis=1)
    # Fail closed: vectorizer or corpus/version mismatch means this is not a
    # checkpoint-matched input. Never silently substitute the saved DEV matrix.
    if not np.allclose(vectors['val'],original,atol=1e-6,rtol=1e-5):
        detail={'max_abs_difference':float(delta.max()),'mismatch_count':int((delta>1e-6).sum()),
                'note':'Released author text did not reproduce checkpoint DEV features; no targets exported.'}
        raise ValueError(json.dumps(detail))
    target=exp/'teachers/idiap_text_author_docs_v1';target.mkdir(parents=True,exist_ok=True)
    audit={'source':'pinned Idiap train_Participant.txt and dev_Participant.txt, matched by original_ID',
        'checkpoint_sha256':digest(ckpt),'vectorizer_sha256':digest(vec),
        'source_sha256':{name:digest(src/'data/AVEC_16_data'/f'{name}_Participant.txt') for name in ['train','dev']},
        'split_sha256':digest(exp/'split/manifest.csv'),
        'dev_checkpoint_tfidf_max_abs_difference':float(delta.max()),
        'train_in_sample':True,'teacher_weights_updated':False,'test_opened':False}
    for ours in ('train','val'):
        frame=result[ours][0];frame.to_csv(target/f'{ours}_targets.csv',index=False)
        audit[ours]=metrics(frame.label.to_numpy(int),frame.text_probability.to_numpy(float))
        print('Author-doc text',ours,{k:audit[ours][k] for k in ('macro_f1','depressed_f1','auroc','confusion_matrix')},flush=True)
    (target/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Verified author DEV TF-IDF max difference:',float(delta.max()),'Saved:',target,flush=True)

if __name__=='__main__':main()
