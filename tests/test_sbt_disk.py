"""Synthetic CPU checks; no DAIC data or pretrained downloads required."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from scripts.teachers.sbt_unimodal import cache, train
from scripts.teachers.sbt_unimodal.models import TextSBTTeacher,AudioSBTTeacher

def frame():
    return pd.DataFrame(dict(sample_id=['a','b','c','d'],participant_id=[1,1,2,2],
                             split=['train']*4,label=[0,0,1,1],text=['one']*4))

class Tiny(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__(); self.encoder=nn.Module()
        self.encoder.config=SimpleNamespace(hidden_size=4)
        self.encoder.encoder=nn.Module(); self.encoder.encoder.layers=nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)])
        self.classifier=nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Dropout(.3),nn.Linear(4,1))
    def encode(self,input_values,attention_mask=None):
        z=input_values[:,:4]
        for layer in self.encoder.encoder.layers: z=layer(z)
        return z
    def forward(self,input_values,attention_mask=None): return self.classifier(self.encode(input_values)).squeeze(-1)
    def freeze_encoder(self): self.encoder.requires_grad_(False)
    def unfreeze_top(self): self.encoder.requires_grad_(True)

class Samples:
    def __init__(self,df,*args): self.df=df; self.reads=0
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        self.reads+=1; r=self.df.iloc[i]
        return np.arange(8,dtype=np.float32)+i,float(r.label),int(r.participant_id),str(r.sample_id),r.split

def collate(raw):
    x,y,p,s,split=zip(*raw)
    return dict(input_values=torch.tensor(np.array(x)),attention_mask=torch.ones(len(x),8),
                label=torch.tensor(y),participant_id=p,sample_id=s,split=split)

def test_shards_equivalence_reuse_and_corruption(tmp_path):
    torch.manual_seed(9); model=Tiny(); model.freeze_encoder(); ds=Samples(frame())
    args=(model,ds,collate,frame(),{'modality':'audio','revision':'fixed'},tmp_path/'drive',tmp_path/'local',2,'cpu',train.encode_batch)
    cached=cache.build_features(*args)
    direct=model.encode(collate([ds[i] for i in range(4)])['input_values']).detach()
    assert torch.allclose(torch.stack([cached[i]['features'] for i in range(4)]),direct)
    ds.reads=0; cache.build_features(*args); assert ds.reads==0
    shard=next((tmp_path/'drive').glob('*/000000.npy')); shard.write_bytes(b'corrupt')
    cache.build_features(*args); assert ds.reads==2
    model.unfreeze_top()
    with pytest.raises(ValueError,match='frozen'): cache.build_features(*args)

def test_cache_invalidates_content_order_and_revision(tmp_path):
    a=frame(); b=a.iloc[::-1].reset_index(drop=True)
    assert cache.frame_digest(a)!=cache.frame_digest(b)
    b=a.copy(); b.loc[0,'label']=1; assert cache.frame_digest(a)!=cache.frame_digest(b)
    b=a.copy(); b.loc[0,'text']='changed'; assert cache.frame_digest(a)!=cache.frame_digest(b)
    assert cache.digest_json({'revision':'a'})!=cache.digest_json({'revision':'b'})

def test_atomic_failure_preserves_previous_file(tmp_path):
    p=tmp_path/'state'; p.write_bytes(b'old')
    def fail(f): f.write(b'partial'); raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError): cache.atomic_write(p,fail)
    assert p.read_bytes()==b'old'

def test_source_staging_rejects_test_before_reading(tmp_path):
    manifest=pd.DataFrame([dict(participant_id=999,split='test',audio_path='missing',transcript_path='missing')])
    with pytest.raises(ValueError,match='TEST'): cache.stage_sources(manifest,tmp_path)

def test_student_sampling_budget_and_participant_balance(monkeypatch):
    import sys,types,importlib
    google=types.ModuleType('google'); colab=types.ModuleType('google.colab')
    colab.drive=SimpleNamespace(mount=lambda *a:None); google.colab=colab
    monkeypatch.setitem(sys.modules,'google',google); monkeypatch.setitem(sys.modules,'google.colab',colab)
    students=importlib.import_module('scripts.students.train_segment_students')
    f={'participant_ids':np.array([1,1,1,2,3]),'labels':np.array([0,0,0,0,1])}
    w=students.sampler_weights(f,['clean'])
    assert np.isclose(w[:3].sum(),w[3])
    assert np.isclose(w[:4].sum(),w[4])
    for conditions in [['clean'],students.CONDITIONS]:
        loader=students.make_loader(f,pd.DataFrame(),conditions,2,.5,.5,True)
        assert loader.sampler.num_samples==5 and len(loader)==3

def test_text_windows_cover_every_token():
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,pre_tokenizers,processors
    tok=Tokenizer(models.WordLevel({'[UNK]':0,'[PAD]':1,'[CLS]':2,'[SEP]':3,'a':4},unk_token='[UNK]'))
    tok.pre_tokenizer=pre_tokenizers.Whitespace()
    tok.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',2),('[SEP]',3)])
    fast=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='[UNK]',pad_token='[PAD]',cls_token='[CLS]',sep_token='[SEP]')
    b=train.TextCollator(fast,8)([(' '.join(['a']*19),1.,1,'id','dev')])
    assert int((b['input_ids']==4).sum())==19
    assert len(b['input_ids'])==4 and b['window_owner'].tolist()==[0]*4

def test_real_tiny_encoders_pooling_and_freezing():
    from transformers import Wav2Vec2Config,Wav2Vec2Model,AlbertConfig,AlbertModel
    audio=AudioSBTTeacher.__new__(AudioSBTTeacher); nn.Module.__init__(audio)
    cfg=Wav2Vec2Config(hidden_size=8,num_hidden_layers=3,num_attention_heads=2,intermediate_size=16,
        conv_dim=(8,8,8),conv_stride=(2,2,2),conv_kernel=(3,3,3),num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,mask_time_prob=0.,hidden_dropout=0.,attention_dropout=0.)
    audio.encoder=Wav2Vec2Model(cfg); audio.classifier=nn.Linear(8,1); audio.eval(); audio.freeze_encoder()
    x=torch.randn(2,128); mask=torch.ones(2,128,dtype=torch.long); mask[0,80:]=0; x[0,80:]=0
    assert torch.allclose(audio.encode(x,mask)[0],audio.encode(x[:1],mask[:1])[0],atol=1e-5)
    audio.unfreeze_top()
    assert not any(p.requires_grad for p in audio.encoder.encoder.layers[0].parameters())
    assert all(p.requires_grad for p in audio.encoder.encoder.layers[-1].parameters())
    text=TextSBTTeacher.__new__(TextSBTTeacher); nn.Module.__init__(text)
    text.encoder=AlbertModel(AlbertConfig(vocab_size=20,embedding_size=4,hidden_size=8,num_hidden_layers=2,
        num_hidden_groups=1,num_attention_heads=2,intermediate_size=16,max_position_embeddings=16))
    text.classifier=nn.Linear(8,1); text.eval()
    ids=torch.randint(0,20,(3,8)); masks=torch.ones_like(ids); owner=torch.tensor([0,0,1])
    raw=text.encode(ids,masks); pooled=text.encode(ids,masks,owner,2)
    assert torch.allclose(pooled[0],raw[:2].mean(0),atol=1e-6)
    text.unfreeze_top(); assert not any(p.requires_grad for p in text.encoder.pooler.parameters())

@pytest.mark.parametrize('stop_stage,stop_epoch',[(1,1),(1,2),(2,1),(2,2)])
def test_epoch_resume_matches_uninterrupted(tmp_path,monkeypatch,stop_stage,stop_epoch):
    monkeypatch.setattr(train,'AudioSBTTeacher',Tiny)
    monkeypatch.setattr(train,'AudioDataset',Samples)
    monkeypatch.setattr(train,'AudioCollator',lambda prep:collate)
    monkeypatch.setattr(train.AutoFeatureExtractor,'from_pretrained',lambda *a,**kw:None)
    args=train.parse_args(['--epochs','4','--freeze-epochs','2','--audio-batch','2','--workers','0',
                           '--local-cache',str(tmp_path/'local')])
    tr=frame(); dv=tr.copy(); dv['split']='dev'; dv['participant_id']+=10
    revisions={train.KD_CFG['teacher_audio_model']:'pinned'}
    def run(root):
        train.seed_all(42)
        return train.train_one('audio',tr,dv,root,args,'cpu',revisions,[])[2]
    full=run(tmp_path/'full')
    original=train.atomic_torch
    def interrupted(path,value):
        original(path,value)
        if str(path).endswith('last.pt') and value['stage']==stop_stage and value['stage_epoch']==stop_epoch:
            raise RuntimeError('simulated disconnect')
    monkeypatch.setattr(train,'atomic_torch',interrupted)
    with pytest.raises(RuntimeError,match='disconnect'): run(tmp_path/'resume')
    monkeypatch.setattr(train,'atomic_torch',original)
    resumed=run(tmp_path/'resume')
    assert resumed['best_epoch']==full['best_epoch']
    for name,tensor in full['model'].items(): assert torch.equal(tensor,resumed['model'][name]),name
    # A completed checkpoint bypasses training and cache extraction.
    monkeypatch.setattr(train,'build_features',lambda *a,**kw:pytest.fail('cache rebuilt'))
    assert run(tmp_path/'resume')['best_epoch']==resumed['best_epoch']
