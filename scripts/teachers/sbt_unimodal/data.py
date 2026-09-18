from __future__ import annotations
from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import model_info

from scripts.students.data_utils import read_split,index_sources,read_turns,make_aligned_segments

SEED=103
KD_CFG=dict(
    protocol='RA-PDS-KD-sbt-unimodal-v1',
    segmenter='aligned_turn_partition_v1',
    max_segments_per_participant=128,
    max_audio_seconds=10.0,
    min_audio_seconds=0.5,
    sample_rate=16000,
    max_text_tokens=254,
    text_model='sentence-transformers/all-MiniLM-L6-v2',
    teacher_audio_model='facebook/wav2vec2-base',
    teacher_text_model='albert-large-v2',
    teacher_audio_seconds=15.0,
    teacher_text_tokens=128,
)

def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def hf_revision(name):
    try: return model_info(name).sha
    except Exception: return None

def build_manifests(data_root:Path,exp_root:Path):
    train=read_split(data_root,'train_split_Depression_AVEC2017.csv',True).assign(split='train')
    dev=read_split(data_root,'dev_split_Depression_AVEC2017.csv',True)
    dev=dev.loc[dev.participant_id.ne(440)].reset_index(drop=True).assign(split='dev')
    test=read_split(data_root,'test_split_Depression_AVEC2017.csv',False)
    assert (len(train),len(dev),len(test))==(107,34,47)
    assert not set(train.participant_id)&set(dev.participant_id)
    assert not set(train.participant_id)&set(test.participant_id)
    assert not set(dev.participant_id)&set(test.participant_id)
    manifest=pd.concat([train,dev],ignore_index=True)

    audio_idx,text_idx=index_sources(data_root)
    rows=[]
    for r in manifest.itertuples():
        ap=audio_idx.get(int(r.participant_id),[]); tp=text_idx.get(int(r.participant_id),[])
        assert len(ap)==1 and len(tp)==1,f'{r.participant_id}: expected one audio and one transcript'
        rows.append(dict(participant_id=int(r.participant_id),split=r.split,label=int(r.label),
                         audio_path=str(ap[0]),transcript_path=str(tp[0])))
    manifest=pd.DataFrame(rows)

    revisions={KD_CFG['text_model']:hf_revision(KD_CFG['text_model']),
               KD_CFG['teacher_audio_model']:hf_revision(KD_CFG['teacher_audio_model']),
               KD_CFG['teacher_text_model']:hf_revision(KD_CFG['teacher_text_model'])}
    seg_tok=AutoTokenizer.from_pretrained(KD_CFG['text_model'],revision=revisions[KD_CFG['text_model']])
    text_tok=AutoTokenizer.from_pretrained(KD_CFG['teacher_text_model'],revision=revisions[KD_CFG['teacher_text_model']])

    kd_rows=[]; audio_rows=[]; text_rows=[]
    for r in manifest.itertuples():
        turns=read_turns(r.transcript_path)
        for s in make_aligned_segments(r.participant_id,turns,seg_tok,KD_CFG):
            kd_rows.append(dict(participant_id=int(r.participant_id),split=r.split,label=int(r.label),
                                audio_path=r.audio_path,transcript_path=r.transcript_path,**s))

        ar=[]
        for ti,t in enumerate(turns.itertuples()):
            dur=float(t.stop_time-t.start_time); n=max(1,math.ceil(dur/KD_CFG['teacher_audio_seconds']))
            for k in range(n):
                start=float(t.start_time)+dur*k/n; stop=float(t.start_time)+dur*(k+1)/n
                if stop-start<KD_CFG['min_audio_seconds']: continue
                ar.append(dict(participant_id=int(r.participant_id),split=r.split,label=int(r.label),
                               sample_id=f'{int(r.participant_id)}_a_{ti:04d}_{k:02d}',
                               audio_path=r.audio_path,start=start,stop=stop))
        if len(ar)>KD_CFG['max_segments_per_participant']:
            idx=np.linspace(0,len(ar)-1,KD_CFG['max_segments_per_participant'],dtype=int); ar=[ar[i] for i in idx]
        audio_rows.extend(ar)

        tr=[]
        for ti,t in enumerate(turns.itertuples()):
            ids=text_tok.encode(t.value,add_special_tokens=False)
            if not ids: continue
            for k in range(0,len(ids),KD_CFG['teacher_text_tokens']):
                piece=ids[k:k+KD_CFG['teacher_text_tokens']]
                txt=text_tok.decode(piece,skip_special_tokens=True).strip()
                if txt:
                    tr.append(dict(participant_id=int(r.participant_id),split=r.split,label=int(r.label),
                                   sample_id=f'{int(r.participant_id)}_t_{ti:04d}_{k//KD_CFG["teacher_text_tokens"]:02d}',
                                   text=txt))
        if len(tr)>KD_CFG['max_segments_per_participant']:
            idx=np.linspace(0,len(tr)-1,KD_CFG['max_segments_per_participant'],dtype=int); tr=[tr[i] for i in idx]
        text_rows.extend(tr)

    kd=pd.DataFrame(kd_rows); aud=pd.DataFrame(audio_rows); txt=pd.DataFrame(text_rows)
    assert not kd.segment_id.duplicated().any()
    assert not aud.sample_id.duplicated().any()
    assert not txt.sample_id.duplicated().any()

    cache_payload=dict(KD_CFG,revisions=revisions)
    key=hashlib.sha256(json.dumps(cache_payload,sort_keys=True).encode()).hexdigest()[:16]
    cache_root=exp_root/'features'/f'rapdskd_segments_{key}'; cache_root.mkdir(parents=True,exist_ok=True)
    save_json(cache_root/'config.json',cache_payload)
    kd.to_csv(cache_root/'segment_manifest_train_dev.csv',index=False)
    aud.to_csv(cache_root/'sbt_audio_teacher_samples.csv',index=False)
    txt.to_csv(cache_root/'sbt_text_teacher_samples.csv',index=False)
    return manifest,kd,aud,txt,cache_root,revisions
