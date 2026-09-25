"""Score frozen Step-Audio2 answer-first generations for official TRAIN and DEV."""
import argparse,json,re
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,classification_report
from .split import digest

def explicit_yes_no(response):
 response=response or '';candidates=[]
 m=re.match(r'^\s*(?:<[^>\r\n]{1,32}>\s*)*(yes|no)\b',response,re.I)
 if m:candidates.append(m.group(1).lower())
 candidates.extend(x.lower() for x in re.findall(r'<英语>\s*(yes|no)\b',response,re.I))
 return candidates[0] if candidates and len(set(candidates))==1 else None

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split-dir',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--generations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
 manifest_path=a.split_dir/'manifest.csv';marker=json.loads((a.split_dir/'complete.json').read_text())
 if digest(manifest_path)!=marker['manifest_sha256']:raise ValueError('Split manifest changed')
 manifest=pd.read_csv(manifest_path);expected=manifest.loc[manifest.split.isin(['train','val'])].sort_values('participant_id')
 index=pd.read_csv(a.index).sort_values('participant_id').reset_index(drop=True)
 if len(index)!=141 or index.participant_id.duplicated().any():raise ValueError('Expected 141 unique official TRAIN/DEV rows')
 if not index[['participant_id','label','split']].reset_index(drop=True).equals(expected[['participant_id','label','split']].reset_index(drop=True)):raise ValueError('Prediction index differs from canonical TRAIN/DEV manifest')
 rows=[json.loads(x) for x in a.generations.read_text().splitlines() if x.strip()]
 if len(rows)!=len(index):raise ValueError(f'Expected {len(index)} generations, found {len(rows)}')
 expected_paths=dict(zip(index.audio_path,index.participant_id));seen={};unknown=[]
 for row in rows:
  audios=row.get('audios')
  if not isinstance(audios,list) or len(audios)!=1:raise ValueError('Generation lacks its single full-interview audio path')
  path=str(audios[0]);pid=expected_paths.get(path)
  if pid is None or pid in seen:raise ValueError(f'Unknown or duplicate generated audio path: {path}')
  response=row.get('response') or '';answer=explicit_yes_no(response)
  seen[pid]=(answer,response)
  if answer is None:unknown.append(pid)
 if len(seen)!=141:raise ValueError(f'Generation coverage incomplete: {len(seen)}/141')
 result=index.copy();result['response_parseable']=[seen[int(pid)][0] is not None for pid in result.participant_id]
 result['prediction']=[None if seen[int(pid)][0] is None else int(seen[int(pid)][0]=='yes') for pid in result.participant_id]
 result['response']=[seen[int(pid)][1] for pid in result.participant_id]
 metrics={}
 for split in ('train','val'):
  f=result.loc[result.split.eq(split)].copy();known=f.loc[f.prediction.notna()];y=known.label.to_numpy(int);pred=known.prediction.to_numpy(int)
  metrics[split]={'n_total':len(f),'n_parseable':len(known),'n_unparseable':int(len(f)-len(known)),
    'accuracy_parseable_only':float(accuracy_score(y,pred)) if len(known) else None,
    'macro_f1_parseable_only':float(f1_score(y,pred,average='macro',zero_division=0)) if len(known) else None,
    'depressed_f1_parseable_only':float(f1_score(y,pred,zero_division=0)) if len(known) else None,
    'confusion_matrix_parseable_only':confusion_matrix(y,pred,labels=[0,1]).tolist() if len(known) else None,
    'classification_report_parseable_only':classification_report(y,pred,labels=[0,1],target_names=['Non-depressed','Depressed'],output_dict=True,zero_division=0) if len(known) else None,
    'full_split_scored':bool(len(known)==len(f))}
 summary={'adapter':'maimai11/woz frozen; no training','n_train_dev':141,'unparseable_participant_ids':unknown,'results':metrics,'probabilities_available':False,'auroc_available':False,'test_opened':False,'status':'EXPLORATORY; the author training roster and preprocessing files are unavailable, and the public adapter reports metrics on an original 35-person DEV set. Check full_split_scored before interpreting metrics.'}
 a.output.mkdir(parents=True,exist_ok=True);result.to_csv(a.output/'participant_predictions.csv',index=False);(a.output/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return summary
if __name__=='__main__':main()
