"""Evaluate the pinned frozen USSD audio checkpoint on TRAIN/DEV only; do not train."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.teachers.ussd_audio.common import CustomComparE16, load_author_stats, AUTHOR_RUN4_CHECKPOINT_SHA256
from .split import digest
from .train_audio_teacher import evaluate, rows_for
from .train_text_teacher import verified_split, metric


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir',type=Path,required=True);p.add_argument('--coverage-dir',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--author-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=32);a=p.parse_args(argv)
    if digest(a.checkpoint)!=AUTHOR_RUN4_CHECKPOINT_SHA256: raise ValueError('Checkpoint does not match pinned USSD run #4')
    manifest=verified_split(a.split_dir,a.coverage_dir);indexed=rows_for(a.features,manifest)
    mean,std,normalizer_path=load_author_stats(a.author_root)
    signature={'split_sha256':digest(a.split_dir/'manifest.csv'),'coverage_sha256':digest(a.coverage_dir/'participant_manifest.csv'),
      'checkpoint_sha256':digest(a.checkpoint),'normalization_sha256':digest(normalizer_path),'batch_size':a.batch_size,'training_performed':False}
    if a.output.exists() and any(a.output.iterdir()):
        audit=a.output/'audit.json'
        if audit.is_file() and json.loads(audit.read_text()).get('signature')==signature:
            print('Verified frozen audio evaluation:',a.output);print(audit.read_text());return
        raise ValueError('Use a fresh frozen-evaluation output folder')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=CustomComparE16().to(device)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);weights=saved.get('state_dict',saved.get('model_state_dict',saved))
    if all(k.startswith('module.') for k in weights): weights={k[7:]:v for k,v in weights.items()}
    model.load_state_dict(weights,strict=True);model.eval()
    a.output.mkdir(parents=True,exist_ok=True);metrics={}
    for split in ('train','val'):
        frame=manifest.loc[manifest.split.eq(split)].sort_values('participant_id')
        predictions,embeddings=evaluate(model,frame,indexed,mean,std,device,a.batch_size)
        y=predictions.label.to_numpy(int);prob=predictions.audio_probability.to_numpy(float)
        predictions['prediction']=(prob>=.5).astype(int)
        predictions.to_csv(a.output/f'{split}_audio_targets.csv',index=False)
        np.savez_compressed(a.output/f'{split}_audio_embeddings.npz',participant_ids=predictions.participant_id.to_numpy(int),labels=y,embedding=embeddings,probability=prob)
        metrics[split]=metric(y,prob)
    audit={'signature':signature,'teacher':'USSD ComParE16+LSTM published run #4, frozen','frozen_checkpoint':True,
      'train_participants':int((manifest.split=='train').sum()),'val_participants':int((manifest.split=='val').sum()),
      'test_participants':int((manifest.split=='student_test').sum()),'test_media_opened':False,'metrics':metrics,
      'warning':'This checkpoint has previous DAIC-WOZ exposure; DEV scores are exploratory.'}
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2));return audit

if __name__=='__main__':main()
