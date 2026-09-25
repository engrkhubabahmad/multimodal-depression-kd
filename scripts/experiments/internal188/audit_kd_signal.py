"""Read-only audit of KD targets and matched student predictions; TEST unopened."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd


def describe(v):
    q=np.asarray(v,dtype=float)
    return {'min':float(q.min()),'median':float(np.median(q)),
            'mean':float(q.mean()),'max':float(q.max()),'std':float(q.std())}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True,type=Path);a=p.parse_args(argv)
    e=a.experiment
    t=pd.read_csv(e/'teachers/idiap_text_frozen/train_text_kd_targets.csv')
    u=pd.read_csv(e/'teachers/nusd_ecapa_frozen/evaluation/train_audio_targets.csv')
    z=t.merge(u,on=['participant_id','label'],validate='one_to_one')
    if len(z)!=107 or z.participant_id.duplicated().any():raise ValueError('Invalid TRAIN targets')
    y=z.label.to_numpy(int);tp=z.text_probability.to_numpy(float);ap=z.audio_probability.to_numpy(float)
    tl=z.text_logit.to_numpy(float);al=z.audio_logit.to_numpy(float)
    if not np.isfinite([tp,ap,tl,al]).all():raise ValueError('Nonfinite targets')
    sig=lambda x:1/(1+np.exp(-x))
    soft=np.column_stack([sig(tl/2),sig(al/2)])
    rel=np.column_stack([np.where(y==1,tp,1-tp),np.where(y==1,ap,1-ap)])
    w=rel/rel.sum(axis=1,keepdims=True)
    standard=soft.mean(axis=1);ra=(w*soft).sum(axis=1)
    report={'n_train':107,'teacher_text_probability':describe(tp),'teacher_audio_probability':describe(ap),
        'teacher_disagreement_at_0.5':int(np.sum((tp>=.5)!=(ap>=.5))),
        'teacher_text_train_errors':int(np.sum((tp>=.5)!=y)),
        'teacher_audio_train_errors':int(np.sum((ap>=.5)!=y)),
        'standard_soft_target':describe(standard),'ra_soft_target':describe(ra),
        'absolute_target_difference':describe(np.abs(ra-standard)),
        'target_correlation':float(np.corrcoef(ra,standard)[0,1]),
        'ra_text_weight':describe(w[:,0]),
        'caveat':'Reliability is computed with TRAIN labels and is unavailable for unlabeled inference.'}
    base=e/'students/canonical_v1/matched_kd_v2';frames=[]
    for mode in ('plain','standard_kd','ra_kd'):
        f=pd.read_csv(base/mode/'dev_predictions.csv')
        if len(f)!=34 or f.participant_id.duplicated().any():raise ValueError(f'Invalid {mode} DEV predictions')
        frames.append(f.set_index('participant_id').sort_index())
    if not all(frames[0].index.equals(f.index) and frames[0].label.equals(f.label) for f in frames[1:]):
        raise ValueError('Student DEV predictions do not align')
    report['dev_prediction_differences']={
        'plain_vs_standard':int((frames[0].prediction!=frames[1].prediction).sum()),
        'standard_vs_ra':int((frames[1].prediction!=frames[2].prediction).sum()),
        'plain_vs_ra':int((frames[0].prediction!=frames[2].prediction).sum()),
        'standard_vs_ra_probability_abs':describe(abs(frames[1].probability-frames[2].probability))}
    print(json.dumps(report,indent=2))
    (base/'kd_signal_audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
