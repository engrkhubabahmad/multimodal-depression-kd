from __future__ import annotations
import math,torch
from torch import nn

class AudioSegmentEncoder(nn.Module):
    """Compact temporal encoder for one 130x384 ComParE16 segment."""
    def __init__(self,d_model=96,gru_hidden=64,dropout=.25):
        super().__init__()
        self.front=nn.Sequential(
            nn.Conv1d(130,d_model,5,stride=2,padding=2,bias=False),
            nn.BatchNorm1d(d_model),nn.GELU(),
            nn.Conv1d(d_model,d_model,5,stride=2,padding=2,groups=d_model,bias=False),
            nn.Conv1d(d_model,d_model,1,bias=False),
            nn.BatchNorm1d(d_model),nn.GELU(),nn.Dropout(dropout))
        self.gru=nn.GRU(d_model,gru_hidden,batch_first=True)
        self.proj=nn.Sequential(nn.Linear(gru_hidden,d_model),nn.GELU(),nn.LayerNorm(d_model))
    def forward(self,x):
        b,s,c,t=x.shape
        z=self.front(x.reshape(b*s,c,t)).transpose(1,2)
        _,h=self.gru(z)
        return self.proj(h[-1]).reshape(b,s,-1)

class TextDocumentEncoder(nn.Module):
    def __init__(self,input_dim=250,d_model=96,dropout=.30):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,128),nn.GELU(),nn.Dropout(dropout),
                               nn.Linear(128,d_model),nn.GELU(),nn.LayerNorm(d_model))
    def forward(self,x): return self.net(x)

class RichParticipantStudent(nn.Module):
    """227k-class multimodal student with text-conditioned acoustic attention."""
    def __init__(self,text_dim=250,d_model=96,dropout=.35):
        super().__init__(); self.d_model=d_model
        self.audio=AudioSegmentEncoder(d_model=d_model); self.text=TextDocumentEncoder(text_dim,d_model)
        self.query=nn.Linear(d_model,d_model,bias=False); self.key=nn.Linear(d_model,d_model,bias=False)
        self.salience=nn.Linear(d_model,1)
        self.classifier=nn.Sequential(nn.Linear(4*d_model,128),nn.GELU(),nn.Dropout(dropout),
                                      nn.Linear(128,32),nn.GELU(),nn.Dropout(dropout/2),nn.Linear(32,1))
    def forward(self,audio,text_tfidf,segment_mask,return_attention=False):
        a=self.audio(audio); t=self.text(text_tfidf)
        score=(self.key(a)*self.query(t).unsqueeze(1)).sum(-1)/math.sqrt(self.d_model)+self.salience(a).squeeze(-1)
        score=score.masked_fill(~segment_mask.bool(),-1e9); w=torch.softmax(score,dim=1)
        ap=(a*w.unsqueeze(-1)).sum(1)
        fused=torch.cat([ap,t,ap*t,torch.abs(ap-t)],dim=-1)
        logit=self.classifier(fused).squeeze(-1)
        return (logit,w) if return_attention else logit

def parameter_count(model):
    return {"total":sum(p.numel() for p in model.parameters()),
            "trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)}
