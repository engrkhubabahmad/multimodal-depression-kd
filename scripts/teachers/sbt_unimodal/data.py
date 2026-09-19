from __future__ import annotations
from pathlib import Path
import json, math, hashlib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import model_info
from tqdm.auto import tqdm

from scripts.students.data_utils import read_split,index_sources,read_turns,make_aligned_segments
from scripts.teachers.sbt_unimodal.cache import stage_sources,frame_digest

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
    teacher_text_model='albert-base-v2',
    teacher_audio_seconds=15.0,
    teacher_text_tokens=128,
)

def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def hf_revision(name):
    revision=model_info(name).sha
    if not revision: raise RuntimeError('Cannot pin model revision: '+name)
    return revision

def build_manifests(data_root:Path,exp_root:Path,local_root='/content/rapdskd_teacher_cache'):
    train=read_split(data_root,'train_split_Depression_AVEC2017.csv',True).assign(split='train')
    dev=read_split(data_root,'dev_split_Depression_AVEC2017.csv',True)
    dev=dev.loc[dev.participant_id.ne(440)].reset_index(drop=True).assign(split='dev')
    # Read IDs only: even if this file contains labels, do not load them.
    test=pd.read_csv(data_root/'metadata'/'test_split_Depression_AVEC2017.csv',
                     usecols=lambda c:c.strip().lower()=='participant_id')
    test.columns=['participant_id']; test['participant_id']=pd.to_numeric(test.participant_id,errors='raise')
    assert not test.participant_id.duplicated().any() and test.participant_id.notna().all()
    assert (len(train),len(dev),len(test))==(107,34,47)
    assert not set(train.participant_id)&set(dev.participant_id)
    assert not set(train.participant_id)&set(test.participant_id)
    assert not set(dev.participant_id)&set(test.participant_id)
    manifest=pd.concat([train,dev],ignore_index=True)

    print('Scanning DAIC-WOZ source paths...',flush=True)
    audio_idx,text_idx=index_sources(data_root)
    rows=[]
    for r in manifest.itertuples():
        ap=audio_idx.get(int(r.participant_id),[]); tp=text_idx.get(int(r.participant_id),[])
        assert len(ap)==1 and len(tp)==1,f'{r.participant_id}: expected one audio and one transcript'
        rows.append(dict(participant_id=int(r.participant_id),split=r.split,label=int(r.label),
                         audio_path=str(ap[0]),transcript_path=str(tp[0])))
    manifest=pd.DataFrame(rows)
    source_map,source_identity=stage_sources(manifest,local_root)

    revisions={KD_CFG['text_model']:hf_revision(KD_CFG['text_model']),
               KD_CFG['teacher_audio_model']:hf_revision(KD_CFG['teacher_audio_model']),
               KD_CFG['teacher_text_model']:hf_revision(KD_CFG['teacher_text_model'])}
    seg_tok=AutoTokenizer.from_pretrained(KD_CFG['text_model'],revision=revisions[KD_CFG['text_model']])


    kd_rows=[]; audio_rows=[]; text_rows=[]
    for r in tqdm(manifest.itertuples(),total=len(manifest),desc='Build participant and KD manifests',colour='green'):
        turns=read_turns(r.transcript_path)
        for s in make_aligned_segments(r.participant_id,turns,seg_tok,KD_CFG):
            kd_rows.append(dict(participant_id=int(r.participant_id),split=r.split,label=int(r.label),
                                audio_path=r.audio_path,transcript_path=r.transcript_path,**s))

        # Author loader expects one prepared audio/text document per CSV row.
        # Original conversion script is unavailable: explicitly use participant
        # speech intervals and joined participant text, without interviewer speech.
        common=dict(participant_id=int(r.participant_id),split=r.split,label=int(r.label),
                    sample_id=f'{int(r.participant_id)}_participant')
        intervals=[[float(t.start_time),float(t.stop_time)] for t in turns.itertuples()]
        audio_rows.append(dict(**common,audio_path=r.audio_path,audio_intervals=json.dumps(intervals)))
        text_rows.append(dict(**common,text=' '.join(turns.value)))

    kd=pd.DataFrame(kd_rows); aud=pd.DataFrame(audio_rows); txt=pd.DataFrame(text_rows)
    assert not kd.segment_id.duplicated().any()
    assert not aud.sample_id.duplicated().any()
    assert not txt.sample_id.duplicated().any()

    cache_payload=dict(KD_CFG,revisions=revisions,sources=source_identity,
                       manifest_hash=frame_digest(kd),audio_hash=frame_digest(aud),text_hash=frame_digest(txt))
    key=hashlib.sha256(json.dumps(cache_payload,sort_keys=True).encode()).hexdigest()[:16]
    cache_root=exp_root/'features'/f'rapdskd_segments_{key}'; cache_root.mkdir(parents=True,exist_ok=True)
    save_json(cache_root/'config.json',cache_payload)
    kd.to_csv(cache_root/'segment_manifest_train_dev.csv',index=False)
    aud.to_csv(cache_root/'sbt_audio_teacher_samples.csv',index=False)
    txt.to_csv(cache_root/'sbt_text_teacher_samples.csv',index=False)
    return manifest,kd,aud,txt,cache_root,revisions,source_map,source_identity
