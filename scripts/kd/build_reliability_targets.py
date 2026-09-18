"""Build reliability-aware segment-level KD targets from frozen teacher outputs.

TRAIN and DEV only. TEST is never loaded or queried.
Default reliability: 1 - normalized binary entropy.
"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json, math
import numpy as np
import pandas as pd

DATA_ROOT=Path('/content/drive/MyDrive/DAIC_WOZ'); SEED=103
TEACHER_ROOT=DATA_ROOT/'experiments'/'teachers'/'segment_level_v1'/f'seed_{SEED}'
OUT=DATA_ROOT/'processed'/'rapdskd_kd'/f'seed_{SEED}'; OUT.mkdir(parents=True,exist_ok=True)

def entropy_reliability(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    h=-(p*np.log(p)+(1-p)*np.log(1-p))/math.log(2.0)
    return np.clip(1.0-h,1e-3,1.0)

def build(split):
    name=f'{split}_segment_predictions.csv'
    a=pd.read_csv(TEACHER_ROOT/'audio'/name); t=pd.read_csv(TEACHER_ROOT/'text'/name)
    keys=['participant_id','segment_id','split','label']
    for d in (a,t):
        assert set(keys+['probability'])<=set(d.columns)
        assert d['split'].eq(split).all() and not d.segment_id.duplicated().any()
        assert not d['split'].eq('test').any()
    d=a[keys+['probability']].rename(columns={'probability':'audio_probability'}).merge(
        t[keys+['probability']].rename(columns={'probability':'text_probability'}),
        on=keys,how='inner',validate='one_to_one')
    assert len(d)==len(a)==len(t), 'Audio/text aligned segment mismatch'
    d['audio_reliability']=entropy_reliability(d.audio_probability)
    d['text_reliability']=entropy_reliability(d.text_probability)
    z=d.audio_reliability+d.text_reliability
    d['audio_weight']=d.audio_reliability/z; d['text_weight']=d.text_reliability/z
    d['teacher_probability']=d.audio_weight*d.audio_probability+d.text_weight*d.text_probability
    d['teacher_disagreement']=(d.audio_probability-d.text_probability).abs()
    d['teacher_agreement']=1.0-d.teacher_disagreement
    path=OUT/f'{split}_reliability_targets.csv'; d.to_csv(path,index=False)
    return d,path

train,train_path=build('train'); dev,dev_path=build('dev')
qc={'protocol':'RA-PDS-KD','level':'segment',
    'reliability':'1 - normalized binary entropy; normalized across audio/text per segment',
    'train_segments':int(len(train)),'dev_segments':int(len(dev)),
    'test_teacher_queries':False,'test_targets_created':False,
    'teacher_root':str(TEACHER_ROOT),'train_output':str(train_path),'dev_output':str(dev_path)}
(OUT/'qc.json').write_text(json.dumps(qc,indent=2))
print(json.dumps(qc,indent=2)); print('TEST CLOSED: no teacher test targets were created.')
