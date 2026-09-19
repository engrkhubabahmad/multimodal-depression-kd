from pathlib import Path
import argparse,importlib.util,sys
import numpy as np,pandas as pd,torch
from tqdm.auto import tqdm
from .common import DEV_IDS,labels,metrics,save_json,transcript

def load_author_main(path):
    spec=importlib.util.spec_from_file_location('idiap_author_main',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--daic-root',required=True); p.add_argument('--dev-csv'); p.add_argument('--source-root',default='/content/solo_teacher_sources/bias_in_daic-woz'); p.add_argument('--output',required=True); p.add_argument('--exclude',type=int,nargs='*',default=[440]); a=p.parse_args(argv)
    root=Path(a.daic_root); src=Path(a.source_root); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    split=Path(a.dev_csv) if a.dev_csv else next(root.rglob('dev_split_Depression_AVEC2017.csv')); ymap=labels(split); excluded=set(a.exclude)
    ids=[i for i in DEV_IDS if i not in excluded]
    if set(ids)-set(ymap): raise ValueError(f'Missing labels: {sorted(set(ids)-set(ymap))}')
    docs=[]; rows=[]
    for pid in tqdm(ids,desc='Text raw preprocessing',colour='green'):
        path,df=transcript(root,pid); keep=df[(df.speaker.astype(str).str.casefold()=='participant') & ~df.value.astype(str).str.contains('scrubbed_entry',case=False,na=False)]; doc=' '.join(keep.value.astype(str)).strip();
        if not doc: raise ValueError(f'{pid}: empty participant transcript')
        docs.append(doc); rows.append({'participant_id':pid,'label':ymap[pid],'utterances':len(keep),'transcript_path':str(path)})
    author=load_author_main(src/'main.py'); author.DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=author.load_induct_gcn_model(src/'model/Participant/model_inductgcn[250].pkl',src/'model/Participant/vtzer_inductgcn[250].pkl',device=author.DEVICE); model.Conv_0_Test=None
    probs=model(docs).detach().cpu().numpy(); classes=list(model.classes_); pos=classes.index('positive'); prob=probs[:,pos]; pred=(prob>=.5).astype(int); y=np.array([r['label'] for r in rows])
    for r,q,z in zip(rows,prob,pred): r.update(prob_depressed=float(q),prediction=int(z))
    pd.DataFrame(rows).to_csv(out/'dev_predictions.csv',index=False); result=metrics(y,pred,prob); result.update(model='idiap/participant-induct-gcn-top250',official_dev_n=35,local_dev_n=len(ids),excluded_ids=sorted(excluded),test_opened=False); save_json(out/'metrics.json',result); print(pd.DataFrame([result]).drop(columns=['classification_report','confusion_matrix']).to_string(index=False)); print('CM [actual rows 0,1]:',result['confusion_matrix']); return result

if __name__=='__main__': main()

