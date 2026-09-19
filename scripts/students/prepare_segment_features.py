"""Prepare lightweight ReLiMP-Net TRAIN/DEV segment features for RA-PDS-KD.

TEST is intentionally not prepared here. The final test evaluator reconstructs the
same clean pipeline only after the final student is frozen.
"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json, sys, zlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.students.data_utils import (read_split,index_sources,read_turns,load_teacher_protocol,
    make_aligned_segments,build_vocab,encode_text,corrupt_text_ids,load_audio_segment,add_gaussian_noise_snr,
    logmel_summary,fit_standardizer,apply_standardizer,quality_from_snr,quality_from_text_noise,sha256_file)

DATA_ROOT=Path('/content/drive/MyDrive/DAIC_WOZ'); SEED=103
AUDIO_SNR_DB=10.0; TEXT_NOISE_RATE=0.30; STUDENT_TEXT_TOKENS=64; MAX_VOCAB=10000
OUT=DATA_ROOT/'processed'/'rapdskd_student'/f'seed_{SEED}'; OUT.mkdir(parents=True,exist_ok=True)

protocol,cache_root,cfg,cache_cfg=load_teacher_protocol(DATA_ROOT,SEED)
model_name=cfg['text_model']; revision=cache_cfg['revisions'][model_name]
tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision)

train=read_split(DATA_ROOT,'train_split_Depression_AVEC2017.csv',True).assign(split='train')
dev=read_split(DATA_ROOT,'dev_split_Depression_AVEC2017.csv',True)
dev=dev.loc[dev.participant_id.ne(440)].reset_index(drop=True).assign(split='dev')
assert (len(train),len(dev))==(107,34)
manifest=pd.concat([train,dev],ignore_index=True)
audio_index,text_index=index_sources(DATA_ROOT)

records=[]
for row in tqdm(list(manifest.itertuples()),desc='Reconstruct aligned TRAIN/DEV segments'):
    pid=int(row.participant_id); ap=audio_index.get(pid,[]); tp=text_index.get(pid,[])
    assert len(ap)==1 and len(tp)==1,f'{pid}: expected one audio/transcript file'
    turns=read_turns(tp[0]); segments=make_aligned_segments(pid,turns,tokenizer,cfg)
    for s in segments:
        records.append({'participant_id':pid,'segment_id':s['segment_id'],'split':row.split,'label':int(row.label),
                        'start':float(s['start']),'stop':float(s['stop']),'text':s['text'],
                        'audio_path':str(ap[0]),'transcript_path':str(tp[0])})
segments=pd.DataFrame(records); assert not segments.segment_id.duplicated().any()

teacher_manifest=pd.read_csv(cache_root/'segment_manifest_train_dev.csv')
assert set(teacher_manifest.segment_id)==set(segments.segment_id),'Student/teacher segment reconstruction mismatch'
expected=teacher_manifest.set_index('segment_id').loc[segments.segment_id].reset_index()
for c in ['participant_id','split','label','text']:
    assert expected[c].tolist()==segments[c].tolist(),f'Student/teacher {c} mismatch'
assert np.allclose(expected[['start','stop']],segments[['start','stop']],rtol=0,atol=1e-7),'Segment timing mismatch'
assert set(segments.loc[segments.split.eq('train'),'participant_id'])==set(train.participant_id)
assert set(segments.loc[segments.split.eq('dev'),'participant_id'])==set(dev.participant_id)

vocab=build_vocab(segments.loc[segments.split.eq('train'),'text'],MAX_VOCAB,2)
(OUT/'vocab.json').write_text(json.dumps(vocab,indent=2,ensure_ascii=False))
print('TRAIN-only vocabulary:',len(vocab))


def segment_seed(segment_id,offset=0): return (SEED+zlib.crc32(segment_id.encode())+offset)&0xffffffff

def build_raw(split):
    part=segments.loc[segments.split.eq(split)].reset_index(drop=True)
    audio_clean=[]; audio_noisy=[]; text_clean=[]; text_noisy=[]
    for r in tqdm(list(part.itertuples()),desc=f'{split} student features'):
        wav=load_audio_segment(r.audio_path,float(r.start),float(r.stop),cfg['sample_rate'])
        rng=np.random.default_rng(segment_seed(r.segment_id,11)); noisy=add_gaussian_noise_snr(wav,AUDIO_SNR_DB,rng)
        audio_clean.append(logmel_summary(wav,cfg['sample_rate'],64)); audio_noisy.append(logmel_summary(noisy,cfg['sample_rate'],64))
        ids=encode_text(r.text,vocab,STUDENT_TEXT_TOKENS); text_clean.append(ids)
        rng=np.random.default_rng(segment_seed(r.segment_id,29)); text_noisy.append(corrupt_text_ids(ids,len(vocab),TEXT_NOISE_RATE,rng))
    return part,np.asarray(audio_clean,np.float32),np.asarray(audio_noisy,np.float32),np.asarray(text_clean,np.int64),np.asarray(text_noisy,np.int64)

tr_meta,tr_a,tr_an,tr_t,tr_tn=build_raw('train')
mean,std=fit_standardizer(tr_a); np.savez_compressed(OUT/'audio_standardizer.npz',mean=mean,std=std)
tr_a=apply_standardizer(tr_a,mean,std); tr_an=apply_standardizer(tr_an,mean,std)

dv_meta,dv_a,dv_an,dv_t,dv_tn=build_raw('dev')
dv_a=apply_standardizer(dv_a,mean,std); dv_an=apply_standardizer(dv_an,mean,std)


def save_split(name,meta,a,an,t,tn):
    np.savez_compressed(OUT/f'{name}_features.npz',segment_ids=meta.segment_id.astype(str).to_numpy(),
                        participant_ids=meta.participant_id.astype(np.int64).to_numpy(),labels=meta.label.astype(np.int64).to_numpy(),
                        audio_clean=a,audio_noisy=an,text_clean=t,text_noisy=tn)
    meta[['participant_id','segment_id','split','label','start','stop']].to_csv(OUT/f'{name}_manifest.csv',index=False)

save_split('train',tr_meta,tr_a,tr_an,tr_t,tr_tn); save_split('dev',dv_meta,dv_a,dv_an,dv_t,dv_tn)
summary={'protocol':'RA-PDS-KD','student':'ReLiMP-Net segment-level','train_participants':107,'dev_participants':34,
         'teacher_manifest_sha256':sha256_file(cache_root/'segment_manifest_train_dev.csv'),
         'teacher_cache_config_sha256':sha256_file(cache_root/'config.json'),
         'train_segments':int(len(tr_meta)),'dev_segments':int(len(dv_meta)),'test_prepared':False,
         'audio_input':'64-bin log-Mel mean+std = 128-D per aligned segment','text_input':f'train-only vocabulary, {STUDENT_TEXT_TOKENS} IDs/segment',
         'audio_noise':f'Gaussian additive noise at {AUDIO_SNR_DB:g} dB SNR','text_noise':f'mask/delete/replace at rate {TEXT_NOISE_RATE:g}',
         'audio_noisy_quality':quality_from_snr(AUDIO_SNR_DB),'text_noisy_quality':quality_from_text_noise(TEXT_NOISE_RATE),
         'audio_normalization_source':'clean TRAIN segments only','vocabulary_source':'TRAIN segment text only',
         'teacher_segment_alignment_verified':True,'test_closed':True}
(OUT/'feature_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2)); print('TEST CLOSED: no test student features were created.')
