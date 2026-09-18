"""Build aligned TRAIN/DEV teacher targets for all three student experiments.

No robustness corruption is created here. Missing/noisy input conditions belong to
Student 3 training and DEV robustness evaluation. TEST is never queried.
"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd

DATA_ROOT=Path('/content/drive/MyDrive/DAIC_WOZ'); SEED=103
TEACHER_ROOT=DATA_ROOT/'experiments'/'teachers'/'segment_level_v1'/f'seed_{SEED}'
OUT=DATA_ROOT/'processed'/'rapdskd_kd'/f'seed_{SEED}'; OUT.mkdir(parents=True,exist_ok=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


selection_path=TEACHER_ROOT/'active_teacher_selection.json'
selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
if selection is not None:
    protocol=json.loads((TEACHER_ROOT/'protocol.json').read_text())
    if selection['test_used'] or selection['cache_root']!=protocol['cache_root']:
        raise ValueError('Selected teachers use a different protocol/cache')
    manifest=Path(protocol['cache_root'])/'segment_manifest_train_dev.csv'
    if digest(manifest)!=selection['manifest_sha256']: raise ValueError('Teacher segment manifest changed; rerun teacher comparison')


def prediction_path(modality,split):
    if selection is None: return TEACHER_ROOT/modality/f'{split}_segment_predictions.csv'
    chosen=selection['modalities'][modality]; path=Path(chosen['root'])/f'{split}_segment_predictions.csv'
    if digest(path)!=chosen['prediction_hashes'][split]: raise ValueError('Selected teacher predictions changed; rerun comparison')
    return path


def entropy_confidence(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    h=-(p*np.log(p)+(1-p)*np.log(1-p))/math.log(2.0)
    return np.clip(1.0-h,1e-3,1.0)


def build(split):
    name=f'{split}_segment_predictions.csv'
    a=pd.read_csv(prediction_path('audio',split)); t=pd.read_csv(prediction_path('text',split))
    keys=['participant_id','segment_id','split','label']
    for d in (a,t):
        assert set(keys+['probability'])<=set(d.columns)
        assert d['split'].eq(split).all() and not d.segment_id.duplicated().any()
        assert not d['split'].eq('test').any()
    d=a[keys+['probability']].rename(columns={'probability':'audio_probability'}).merge(
        t[keys+['probability']].rename(columns={'probability':'text_probability'}),on=keys,how='inner',validate='one_to_one')
    assert len(d)==len(a)==len(t),'Audio/text aligned segment mismatch'
    d['audio_confidence']=entropy_confidence(d.audio_probability)
    d['text_confidence']=entropy_confidence(d.text_probability)
    d['standard_teacher_probability']=0.5*(d.audio_probability+d.text_probability)
    z=d.audio_confidence+d.text_confidence
    d['clean_audio_weight']=d.audio_confidence/z; d['clean_text_weight']=d.text_confidence/z
    d['clean_reliability_teacher_probability']=d.clean_audio_weight*d.audio_probability+d.clean_text_weight*d.text_probability
    d['teacher_disagreement']=(d.audio_probability-d.text_probability).abs()
    path=OUT/f'{split}_teacher_targets.csv'; d.to_csv(path,index=False)
    return d,path

train,train_path=build('train'); dev,dev_path=build('dev')
qc={'protocol':'RA-PDS-KD','level':'segment','train_segments':int(len(train)),'dev_segments':int(len(dev)),
    'teacher_selection':selection,
    'standard_kd':'equal 0.5/0.5 audio-text teacher target on clean TRAIN input',
    'reliability_prior':'teacher entropy confidence only; Student 3 multiplies it by actual modality availability/quality per input condition',
    'robust_conditions_built_here':False,'robust_conditions_stage':'Student 3 training + DEV robustness evaluation only',
    'test_teacher_queries':False,'test_targets_created':False,'train_output':str(train_path),'dev_output':str(dev_path)}
(OUT/'qc.json').write_text(json.dumps(qc,indent=2))
print(json.dumps(qc,indent=2)); print('TEST CLOSED: no teacher test targets were created.')
