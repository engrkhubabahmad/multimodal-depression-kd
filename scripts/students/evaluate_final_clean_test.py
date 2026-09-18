"""One-time clean TEST evaluation for the frozen RA-PDS-KD final student.

No teacher inference, no missing/noisy conditions, no threshold search, no fitting.
If the official test metadata has no labels, provide --test-labels-csv with
participant_id and phq8_binary columns.
"""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import argparse,json,os,sys
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,
                             roc_auc_score,average_precision_score,confusion_matrix,classification_report)

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.students.data_utils import (read_split,index_sources,read_turns,load_teacher_protocol,make_aligned_segments,
    encode_text,load_audio_segment,logmel_summary,apply_standardizer)
from scripts.students.relimpnet_segment import ReLiMPNetSegment


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=103); p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--test-labels-csv',type=str,default=(os.environ.get('TEST_LABELS_CSV') or None),help='Optional CSV with participant_id, phq8_binary if official TEST metadata is unlabeled')
    p.add_argument('--confirm-final-test',action='store_true',default=os.environ.get('RAPDSKD_OPEN_FINAL_TEST','').upper()=='YES',help='Explicitly confirm that the final student and DEV threshold are frozen and TEST may be opened')
    return p.parse_args()


def attach_labels(test,args):
    if 'label' in test.columns: return test
    if not args.test_labels_csv:
        raise RuntimeError('TEST labels are unavailable in test_split_Depression_AVEC2017.csv. Final metrics cannot be computed without labels. '
                           'Provide --test-labels-csv AFTER the final student is frozen; do not use it earlier.')
    lab=pd.read_csv(args.test_labels_csv); lab.columns=lab.columns.str.strip().str.lower()
    assert {'participant_id','phq8_binary'}<=set(lab.columns)
    lab=lab[['participant_id','phq8_binary']].rename(columns={'phq8_binary':'label'}); lab['participant_id']=lab.participant_id.astype(int)
    out=test.merge(lab,on='participant_id',how='left',validate='one_to_one'); assert out.label.notna().all() and out.label.isin([0,1]).all()
    out['label']=out.label.astype(int); return out


def metric_dict(y,p,th):
    pred=(p>=th).astype(int)
    return {'n_segments':int(len(y)),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'depressed_f1':float(f1_score(y,pred,zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'auroc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),'threshold':float(th)}


def main():
    args=parse_args()
    if not args.confirm_final_test:
        raise RuntimeError('FINAL TEST IS CLOSED. Re-run with --confirm-final-test or set RAPDSKD_OPEN_FINAL_TEST=YES only after the final student/checkpoint/DEV threshold are frozen.')
    data_root=Path('/content/drive/MyDrive/DAIC_WOZ'); feat_root=data_root/'processed'/'rapdskd_student'/f'seed_{args.seed}'
    student_root=data_root/'experiments'/'students'/'segment_level_v1'/f'seed_{args.seed}'
    selection=json.loads((student_root/'final_selection.json').read_text())
    assert selection['test_opened'] is False and selection['final_test_condition']=='clean only'
    checkpoint=Path(selection['checkpoint']); threshold=float(selection['threshold']); state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    vocab=json.loads((feat_root/'vocab.json').read_text()); scaler=np.load(feat_root/'audio_standardizer.npz'); mean=scaler['mean']; std=scaler['std']
    protocol,cache_root,cfg,cache_cfg=load_teacher_protocol(data_root,args.seed)
    model_name=cfg['text_model']; revision=cache_cfg['revisions'][model_name]; tokenizer=AutoTokenizer.from_pretrained(model_name,revision=revision)

    test=read_split(data_root,'test_split_Depression_AVEC2017.csv',False); assert len(test)==47
    test=attach_labels(test,args).assign(split='test')
    train_ids=set(read_split(data_root,'train_split_Depression_AVEC2017.csv',True).participant_id)
    dev_ids=set(read_split(data_root,'dev_split_Depression_AVEC2017.csv',True).participant_id)
    test_ids=set(test.participant_id); assert not train_ids&test_ids and not dev_ids&test_ids
    audio_index,text_index=index_sources(data_root)

    rows=[]; audio=[]; text=[]; labels=[]
    for row in tqdm(list(test.itertuples()),desc='FINAL clean TEST features'):
        pid=int(row.participant_id); ap=audio_index.get(pid,[]); tp=text_index.get(pid,[]); assert len(ap)==1 and len(tp)==1,f'{pid}: source coverage'
        turns=read_turns(tp[0]); segs=make_aligned_segments(pid,turns,tokenizer,cfg)
        for s in segs:
            wav=load_audio_segment(ap[0],s['start'],s['stop'],cfg['sample_rate']); feat=logmel_summary(wav,cfg['sample_rate'],64)
            audio.append(apply_standardizer(feat,mean,std)); text.append(encode_text(s['text'],vocab,64)); labels.append(int(row.label))
            rows.append({'participant_id':pid,'segment_id':s['segment_id'],'split':'test','label':int(row.label),'condition':'clean'})
    audio=np.asarray(audio,np.float32); text=np.asarray(text,np.int64); labels=np.asarray(labels,np.int64); meta=pd.DataFrame(rows)
    assert len(audio)==len(text)==len(meta) and not meta.segment_id.duplicated().any()
    np.savez_compressed(feat_root/'final_test_clean_features.npz',segment_ids=meta.segment_id.astype(str).to_numpy(),
                        participant_ids=meta.participant_id.to_numpy(np.int64),labels=labels,audio_clean=audio,text_clean=text)

    model=ReLiMPNetSegment(len(vocab),use_reliability_fusion=bool(state['use_reliability_fusion']))
    model.load_state_dict(state['model']); device='cuda' if torch.cuda.is_available() else 'cpu'; model=model.to(device).eval()
    probs=[]
    with torch.inference_mode():
        for i in range(0,len(meta),args.batch_size):
            a=torch.from_numpy(audio[i:i+args.batch_size]).to(device); t=torch.from_numpy(text[i:i+args.batch_size]).to(device)
            q=torch.ones(len(a),device=device); z=model(a,t,q,q); probs.extend(torch.sigmoid(z).cpu().numpy().tolist())
    probs=np.asarray(probs,float); meta['probability']=probs; meta['prediction']=(probs>=threshold).astype(int)
    out=student_root/'final_test_clean'; out.mkdir(parents=True,exist_ok=True); meta.to_csv(out/'predictions.csv',index=False)
    metrics=metric_dict(labels,probs,threshold); (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    cm=pd.DataFrame(confusion_matrix(labels,meta.prediction,labels=[0,1]),index=['actual_0','actual_1'],columns=['pred_0','pred_1'])
    cr=pd.DataFrame(classification_report(labels,meta.prediction,labels=[0,1],target_names=['non_depressed','depressed'],output_dict=True,zero_division=0)).T
    cm.to_csv(out/'confusion_matrix.csv'); cr.to_csv(out/'classification_report.csv')
    print('\nFINAL CLEAN TEST | mode:',selection['selected_mode'],'| threshold:',threshold); print(json.dumps(metrics,indent=2)); print(cm.to_string())
    print('\nNo robustness condition was evaluated on TEST. No test threshold search or teacher query was performed.')

if __name__=='__main__': main()
