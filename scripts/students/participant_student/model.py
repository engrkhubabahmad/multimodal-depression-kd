from __future__ import annotations
import torch
from torch import nn

class AudioSegmentEncoder(nn.Module):
    def __init__(self,input_dim=128,d_model=128,dropout=.3):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,d_model),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(d_model,d_model),nn.GELU(),nn.LayerNorm(d_model))
    def forward(self,x):
        b,s,d=x.shape; return self.net(x.reshape(b*s,d)).reshape(b,s,-1)

class TextSegmentEncoder(nn.Module):
    def __init__(self,vocab_size,d_model=128,heads=4,layers=1,dropout=.2,pad_id=0):
        super().__init__(); self.pad_id=pad_id; self.emb=nn.Embedding(vocab_size,d_model,padding_idx=pad_id)
        e=nn.TransformerEncoderLayer(d_model=d_model,nhead=heads,dim_feedforward=4*d_model,dropout=dropout,
            activation="gelu",batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(e,num_layers=layers); self.norm=nn.LayerNorm(d_model)
    def forward(self,ids):
        b,s,l=ids.shape; x=ids.reshape(b*s,l); missing=x.eq(self.pad_id).all(1); safe=x.clone(); safe[missing,0]=1
        mask=safe.eq(self.pad_id); h=self.enc(self.emb(safe),src_key_padding_mask=mask)
        valid=(~mask).unsqueeze(-1).float(); pooled=self.norm((h*valid).sum(1)/valid.sum(1).clamp_min(1.0)); pooled[missing]=0
        return pooled.reshape(b,s,-1)

class ParticipantReLiMPNet(nn.Module):
    def __init__(self,vocab_size,audio_dim=128,d_model=128,dropout=.3,use_reliability_fusion=False):
        super().__init__(); self.use_reliability_fusion=use_reliability_fusion
        self.audio=AudioSegmentEncoder(audio_dim,d_model,dropout); self.text=TextSegmentEncoder(vocab_size,d_model,4,1,dropout)
        self.segment_fusion=nn.Sequential(nn.Linear(2*d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.LayerNorm(d_model))
        self.attn=nn.Sequential(nn.Linear(d_model,64),nn.Tanh(),nn.Linear(64,1))
        self.classifier=nn.Sequential(nn.Linear(d_model,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,32),nn.GELU(),nn.Dropout(dropout/2),nn.Linear(32,1))
    def forward(self,audio,text,segment_mask,audio_quality=None,text_quality=None,return_attention=False):
        a=self.audio(audio); t=self.text(text)
        if self.use_reliability_fusion:
            if audio_quality is None or text_quality is None: raise ValueError("quality tensors required")
            aq=audio_quality.float().clamp(0,1); tq=text_quality.float().clamp(0,1); z=(aq+tq).clamp_min(1e-6)
            a=a*(aq/z).unsqueeze(-1); t=t*(tq/z).unsqueeze(-1)
        h=self.segment_fusion(torch.cat([a,t],-1)); m=segment_mask.bool()
        score=self.attn(h).squeeze(-1).masked_fill(~m,-1e9); w=torch.softmax(score,1); pooled=(h*w.unsqueeze(-1)).sum(1)
        logit=self.classifier(pooled).squeeze(-1)
        return (logit,w) if return_attention else logit

def parameter_count(model):
    return {"total":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)}
