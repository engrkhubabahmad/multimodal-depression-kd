"""Prepare one full original DAIC-WOZ audio interview per canonical TRAIN/DEV participant."""
import argparse,json
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from .split import digest
MODEL='stepfun-ai/Step-Audio-2-mini-Think'
ADAPTER='maimai11/woz'
PROMPT='Listen to the participant interview. Is the participant depressed? Answer Yes or No first, then briefly explain.'

def one(root,pattern):
 hits=list(Path(root).rglob(pattern))
 if len(hits)!=1:raise FileNotFoundError(f'Expected one {pattern}; found {len(hits)}')
 return hits[0]

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--daic-root',type=Path,required=True);p.add_argument('--split-dir',type=Path,required=True);p.add_argument('--adapter-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
 mark=json.loads((a.split_dir/'complete.json').read_text());manifest_path=a.split_dir/'manifest.csv'
 if digest(manifest_path)!=mark['manifest_sha256']:raise ValueError('Split manifest changed')
 manifest=pd.read_csv(manifest_path);counts=manifest.split.value_counts().to_dict()
 if counts!={'train':107,'val':34,'student_test':47}:raise ValueError(f'Expected canonical TRAIN/DEV/TEST=107/34/47, got {counts}')
 audit=json.loads(a.adapter_audit.read_text())
 if audit.get('adapter')!=ADAPTER or audit.get('split_sha256')!=digest(manifest_path):raise ValueError('Adapter audit/split mismatch')
 selected=manifest.loc[manifest.split.isin(['train','val'])].sort_values(['split','participant_id']);records=[];index=[]
 for r in tqdm(selected.itertuples(index=False),total=len(selected),desc='Step-Audio2 official TRAIN/DEV',colour='green'):
  pid=int(r.participant_id);audio=one(a.daic_root,f'{pid}_AUDIO.wav')
  records.append({'messages':[{'role':'user','content':'<audio>'+PROMPT}],'audios':[str(audio)]})
  index.append({'participant_id':pid,'label':int(r.label),'split':r.split,'source_split':r.source_split,'audio_path':str(audio),'audio_sha256':digest(audio)})
 a.output.mkdir(parents=True,exist_ok=True);data=a.output/'train_dev.jsonl';idx=a.output/'train_dev_index.csv'
 text=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records);frame=pd.DataFrame(index)
 if data.exists() and data.read_text()!=text:raise ValueError(f'Existing dataset differs: {data}')
 if idx.exists() and not pd.read_csv(idx).equals(frame):raise ValueError(f'Existing participant map differs: {idx}')
 data.write_text(text);frame.to_csv(idx,index=False)
 prov={'adapter':ADAPTER,'adapter_revision':audit['revision'],'split_sha256':digest(manifest_path),'input':'one unsegmented original DAIC participant interview WAV per participant','audio_segmentation':'none; raw audio passed directly to Step-Audio2','prompt':PROMPT,'author_config_match':{'max_length':8000,'response_prefix':None,'max_new_tokens':64,'temperature':0.0},'author_data_preprocessing_source':'not published in adapter repository; this reproduces disclosed inference settings, not unverifiable hidden JSONL preprocessing','n_train':107,'n_val':34,'test_opened':False}
 (a.output/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
 command=['swift','infer','--model',MODEL,'--model_type','step_audio2_mini','--adapters',ADAPTER,'--use_hf','true','--load_args','false','--val_dataset',str(data),'--result_path',str(a.output/'train_dev_predictions.jsonl'),'--infer_backend','transformers','--max_batch_size','1','--temperature','0','--max_new_tokens','64','--max_length','8000','--add_non_thinking_prefix','true']
 print(f'Prepared one full-audio example for each of {len(frame)} TRAIN/DEV participants; TEST untouched.')
 print('Author data preprocessing files are not public; see provenance.json.')
 return command
if __name__=='__main__':main()
