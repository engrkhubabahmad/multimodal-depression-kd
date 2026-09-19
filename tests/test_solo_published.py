import numpy as np,pandas as pd
from scripts.teachers.solo_published.prepare_audio import participant_audio,windows

def test_participant_audio_keeps_only_participant():
    x=np.arange(10,dtype=float); df=pd.DataFrame({'start_time':[0,0.5],'stop_time':[0.5,1.0],'speaker':['Ellie','Participant'],'value':['q','a']}); y=participant_audio(x,10,df); assert np.array_equal(y,np.array([5,6,7,8,9],dtype=float))

def test_windows_shape_and_padding():
    z=list(windows(np.ones((80,2000),dtype=np.float32))); assert len(z)==2; assert z[0].shape==(80,1800); assert np.all(z[1][:,500:]==0)
