from pathlib import Path
import argparse
import librosa,numpy as np,pandas as pd,wave
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm
from .common import DEV_IDS,find_file,labels,transcript

def wav(path):
    with wave.open(str(path)) as f: return np.frombuffer(f.readframes(f.getnframes()),dtype=np.int16).astype(float),f.getframerate()

def participant_audio(x,sr,df):
    keep=df[(df.speaker.astype(str).str.casefold()=='participant') & ~df.value.astype(str).str.contains('scrubbed_entry',case=False,na=False)]
    parts=[x[int(r.start_time*sr):int(r.stop_time*sr)] for r in keep.itertuples()]
    if not parts: raise ValueError('No usable Participant speech')
    return np.hstack(parts)

def windows(mel,frame=1800,hop=1500):
    n=max(1,(mel.shape[1]-frame)//hop+2)
    for i in range(n):
        z=np.zeros((80,frame),dtype=np.float32); q=mel[:,i*hop:i*hop+frame]; z[:,:q.shape[1]]=q; yield z

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--daic-root',required=True); p.add_argument('--dev-csv'); p.add_argument('--output',required=True); p.add_argument('--exclude',type=int,nargs='*',default=[440]); a=p.parse_args(argv)
    root=Path(a.daic_root); out=Path(a.output); split=Path(a.dev_csv) if a.dev_csv else next(root.rglob('dev_split_Depression_AVEC2017.csv')); ymap=labels(split); meta=[]
    for pid in tqdm([i for i in DEV_IDS if i not in set(a.exclude)],desc='Audio author preprocessing',colour='green'):
        _,df=transcript(root,pid); x,sr=wav(find_file(root,pid,'AUDIO.wav')); x=participant_audio(x,sr,df); mel=librosa.power_to_db(librosa.feature.melspectrogram(y=x,sr=sr,n_fft=2048,hop_length=533,n_mels=80)); mel=normalize(mel)
        d=out/'validation/clipped_data/audio/mel-spectrogram'; d.mkdir(parents=True,exist_ok=True)
        for j,z in enumerate(windows(mel)): np.save(d/f'{pid}-{j:02}_audio.npy',z); meta.append((pid,ymap[pid],j))
    m=pd.DataFrame(meta,columns=['participant_id','label','clip']); base=out/'validation/clipped_data'; np.save(base/'ID_gt.npy',m.participant_id); np.save(base/'phq_binary_gt.npy',m.label); np.save(base/'phq_score_gt.npy',m.label*10); np.save(base/'gender_gt.npy',np.zeros(len(m))); np.save(base/'phq_subscores_gt.npy',np.zeros((len(m),8))); m.to_csv(out/'manifest.csv',index=False); print(m.groupby(['label']).size())

if __name__=='__main__': main()
