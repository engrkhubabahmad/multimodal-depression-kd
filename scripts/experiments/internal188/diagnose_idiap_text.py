"""Read-only diagnosis of frozen Idiap text inputs and TRAIN-fitted threshold."""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import f1_score
from .audit_idiap_inference import predict
from .idiap_source import ensure_frozen_export_source
from .train_student_kd import digest
from .train_student_kd_v2 import metrics
from scripts.teachers.idiap_text.export_targets import load_author
from scripts.students.participant_student_v3.pretrain_text import participant_doc


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True,type=Path)
    args=parser.parse_args(argv);exp=args.experiment
    output=exp/'teachers/idiap_text_inductive_v1/diagnosis';output.mkdir(parents=True,exist_ok=True)
    previous=exp/'teachers/idiap_text_inductive_v1/audit.json'
    old=json.loads(previous.read_text())
    if old['split_sha256']!=digest(exp/'split/manifest.csv') or old['test_opened']:
        raise ValueError('Text export belongs to another split or opened TEST')
    train=pd.read_csv(previous.parent/'train_targets.csv')
    dev=pd.read_csv(previous.parent/'val_targets.csv')
    split=pd.read_csv(exp/'split/manifest.csv')
    for name,frame,count in [('train',train,107),('val',dev,34)]:
        wanted=split.loc[split.split.eq(name),['participant_id','label']].sort_values('participant_id')
        got=frame[['participant_id','label']].sort_values('participant_id')
        if len(frame)!=count or frame.participant_id.duplicated().any() or not got.reset_index(drop=True).equals(wanted.reset_index(drop=True)):
            raise ValueError(f'{name} participant/label mismatch')
    y=train.label.to_numpy(int);p=train.text_probability.to_numpy(float)
    if not np.isfinite(p).all():raise ValueError('Nonfinite TRAIN probabilities')
    # A small, fixed search on TRAIN. This is a diagnostic, not a clean
    # generalization estimate; the published teacher was trained on TRAIN.
    candidates=np.r_[0.5,np.quantile(p,np.linspace(.05,.95,19))]
    threshold=float(max(candidates,key=lambda t:(f1_score(y,p>=t,average='macro',zero_division=0),
                                                  f1_score(y,p>=t,pos_label=1,zero_division=0),-abs(t-.5))))
    def summarize(frame,t):
        probabilities=frame.text_probability.to_numpy(float)
        shifted=np.clip(probabilities+.5-t,0,1)
        return metrics(frame.label.to_numpy(int),shifted)
    diagnosis={'train_only_selected_threshold':threshold,
        'train_original':summarize(train,.5),'dev_original':summarize(dev,.5),
        'train_threshold_diagnostic':summarize(train,threshold),
        'dev_threshold_diagnostic':summarize(dev,threshold),
        'train_probability_quantiles':np.quantile(p,[0,.05,.25,.5,.75,.95,1]).tolist(),
        'dev_probability_quantiles':np.quantile(dev.text_probability,[0,.05,.25,.5,.75,.95,1]).tolist(),
        'threshold_source':'TRAIN labels only; evaluated once on DEV for diagnosis',
        'limitation':'TRAIN in-sample threshold may overfit; DEV is repeatedly used in this project; do not use this diagnostic to claim independent performance',
        'test_opened':False}
    src=ensure_frozen_export_source();checkpoint=src/'model/Participant/model_inductgcn[250].pkl'
    with (src/'model/Participant/vtzer_inductgcn[250].pkl').open('rb') as file:vectorizer=pickle.load(file)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    author=load_author(src/'main.py');author.DEVICE=torch.device('cpu')
    model=author.InducTGCN(state['embedding_dim'],state['classes_'],0,vectorizer)
    model.load_state_dict(dict(state['model_state_dict']));model.classes_=state['classes_'];model.cpu().eval()
    coverage=pd.read_csv(exp/'coverage/participant_manifest.csv')
    frame=dev.merge(coverage[['participant_id','label','transcript_path']],on=['participant_id','label'],validate='one_to_one')
    if len(frame)!=34:raise ValueError('Incomplete DEV transcript coverage')
    documents=[participant_doc(Path(path)) for path in frame.transcript_path]
    reconstructed=vectorizer.transform(documents).toarray().astype('float32')
    order=pd.read_csv(src/'data/AVEC_16_data/dev_IDS.txt',sep='\t').original_ID.astype(int).tolist()
    indices=[order.index(int(pid)) for pid in frame.participant_id]
    saved=state['A_dev'].cpu().numpy()[indices,:reconstructed.shape[1]]
    _,saved_p=predict(model,saved)
    _,new_p=predict(model,reconstructed)
    if not np.allclose(new_p,frame.text_probability,atol=1e-6):raise ValueError('Cached reconstructed probabilities differ')
    row_diff=np.max(np.abs(saved-reconstructed),axis=1)
    frame=frame[['participant_id','label','text_probability']].copy()
    frame['author_saved_input_probability']=saved_p
    frame['tfidf_max_abs_difference']=row_diff
    frame['probability_abs_difference']=np.abs(saved_p-new_p)
    frame['reconstructed_positive']=(new_p>=.5).astype(int)
    frame['author_saved_positive']=(saved_p>=.5).astype(int)
    frame.to_csv(output/'dev_input_discrepancies.csv',index=False)
    diagnosis['author_saved_input_dev']=metrics(frame.label.to_numpy(int),saved_p)
    diagnosis['participant_input_mismatches']=int(np.sum(row_diff>1e-6))
    diagnosis['participant_prediction_disagreements']=int(np.sum(frame.reconstructed_positive!=frame.author_saved_positive))
    (output/'summary.json').write_text(json.dumps(diagnosis,indent=2)+'\n')
    print('TRAIN threshold:',threshold)
    for key in ('train_original','dev_original','train_threshold_diagnostic','dev_threshold_diagnostic','author_saved_input_dev'):
        m=diagnosis[key];print(key,{k:m[k] for k in ('macro_f1','depressed_f1','auroc','confusion_matrix')})
    print('Input mismatches:',diagnosis['participant_input_mismatches'],
          'prediction disagreements:',diagnosis['participant_prediction_disagreements'])
    print('Saved:',output)

if __name__=='__main__':main()
