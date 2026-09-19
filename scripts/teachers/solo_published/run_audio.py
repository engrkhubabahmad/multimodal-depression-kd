from pathlib import Path
from types import SimpleNamespace
import argparse,sys
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .common import metrics,save_json

EXPECTED='A+Conv1D-BiLSTM+PHQ-Subscores+Mel+NoGB_2022-03-25_103243_f1_score-0.5532.pt'

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--source-root',required=True); p.add_argument('--features',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); a=p.parse_args(argv); ck=Path(a.checkpoint)
    if not ck.exists(): raise FileNotFoundError(f'Author audio checkpoint missing: {ck}\nExpected file: {EXPECTED}')
    state=torch.load(ck,map_location='cpu',weights_only=False)
    missing={'audio_net','evaluator'}-set(state)
    if missing: raise ValueError(f'Not a compatible author audio checkpoint; missing keys: {sorted(missing)}')
    base=Path(a.source_root)/'models/Audio_ConvLSTM'; sys.path.insert(0,str(base))
    from dataset.dataset import DepressionDataset,ToTensor
    from torchvision import transforms
    import utils as author
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); first=next(iter(state['audio_net'])); args=SimpleNamespace(device=device,gpu='0,0' if first.startswith('module.') else '0')
    cfg={'PHQ_THRESHOLD':10,'WEIGHTS':{'TYPE':'absolute_path','CUSTOM_ABSOLUTE_PATH':str(ck.resolve()),'INCLUDED':['audio_net','evaluator']},'AUDIO_NET':{'INPUT_DIM':80,'CONV_HIDDEN':256,'LSTM_HIDDEN':256,'OUTPUT_DIM':256,'NUM_LAYERS':4,'ACTIVATION':'relu','NORM':'bn','DROPOUT':.5},'EVALUATOR':{'PREDICT_TYPE':'phq-subscores','INPUT_FEATURE_DIM':256,'CLASSES_RESOLUTION':4,'N_CLASSES':4,'N_SUBSCORES':8,'STD':5}}
    audio,evaluator=author.get_models(cfg,args); audio.eval(); evaluator.eval(); root=Path(a.features)/'validation/clipped_data'; ds=DepressionDataset(str(root),'test',True,False,transforms.Compose([ToTensor('test')])); dl=DataLoader(ds,batch_size=64,num_workers=0)
    rows=[]
    with torch.no_grad():
        for batch in tqdm(dl,desc='Audio frozen inference',colour='green'):
            feat=audio(batch['audio'].to(device)); heads=evaluator(feat); score=author.compute_score(heads,cfg['EVALUATOR'],args).cpu().numpy()
            for pid,y,s in zip(batch['ID'].numpy(),batch['phq_binary_gt'].numpy(),score): rows.append((int(pid),int(y),float(s),int(s>=10)))
    clips=pd.DataFrame(rows,columns=['participant_id','label','phq_score','prediction']); participants=clips.groupby('participant_id',as_index=False).agg(label=('label','first'),clips=('prediction','size'),positive_clips=('prediction','sum'),mean_phq_score=('phq_score','mean')); participants['prediction']=(participants.positive_clips>=participants.clips*.5).astype(int)
    y=participants.label.to_numpy(); pred=participants.prediction.to_numpy(); result=metrics(y,pred,pred.astype(float)); result.update(model='PingCheng-Wei/Audio_ConvLSTM',checkpoint=ck.name,soft_probability_available=False,test_opened=False)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); clips.to_csv(out/'dev_clip_predictions.csv',index=False); participants.to_csv(out/'dev_predictions.csv',index=False); save_json(out/'metrics.json',result); print(participants.to_string(index=False)); print('CM [actual rows 0,1]:',result['confusion_matrix']); return result

if __name__=='__main__': main()
