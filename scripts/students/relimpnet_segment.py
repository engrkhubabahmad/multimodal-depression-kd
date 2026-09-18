"""Lightweight segment-level ReLiMP-Net student used by RA-PDS-KD."""
from __future__ import annotations
import torch
from torch import nn


class TextBranch(nn.Module):
    def __init__(self, vocab_size, d_model=128, heads=4, layers=1, dropout=0.2, pad_id=0):
        super().__init__(); self.pad_id=pad_id
        self.embedding=nn.Embedding(vocab_size,d_model,padding_idx=pad_id)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=heads,dim_feedforward=d_model*4,
                                       dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,num_layers=layers)
        self.norm=nn.LayerNorm(d_model)
    def forward(self,ids):
        missing=ids.eq(self.pad_id).all(dim=1); safe=ids.clone(); safe[missing,0]=1
        mask=safe.eq(self.pad_id); x=self.embedding(safe); x=self.encoder(x,src_key_padding_mask=mask)
        valid=(~mask).unsqueeze(-1).float(); denom=valid.sum(1).clamp(min=1.0); pooled=self.norm((x*valid).sum(1)/denom)
        pooled[missing]=0.0; return pooled


class AudioBranch(nn.Module):
    def __init__(self,input_dim=128,d_model=128,dropout=0.2):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,d_model),nn.GELU(),
                               nn.Dropout(dropout),nn.Linear(d_model,d_model),nn.GELU(),nn.LayerNorm(d_model))
    def forward(self,x): return self.net(x)


class ReLiMPNetSegment(nn.Module):
    """Same lightweight student backbone for all 3 experiments.

    `use_reliability_fusion=False`: clean equal treatment of both modalities.
    `use_reliability_fusion=True`: modality representations are scaled by normalized
    quality/availability weights before fusion. No teacher output is needed at inference.
    """
    def __init__(self,vocab_size,audio_dim=128,d_model=128,dropout=0.3,use_reliability_fusion=False):
        super().__init__(); self.use_reliability_fusion=use_reliability_fusion
        self.audio=AudioBranch(audio_dim,d_model,dropout)
        self.text=TextBranch(vocab_size,d_model,4,1,dropout)
        self.classifier=nn.Sequential(nn.Linear(d_model*2,d_model),nn.GELU(),nn.Dropout(dropout),
                                      nn.Linear(d_model,32),nn.GELU(),nn.Dropout(dropout/2),nn.Linear(32,1))
    def forward(self,audio_x,text_ids,audio_quality=None,text_quality=None):
        a=self.audio(audio_x); t=self.text(text_ids)
        if self.use_reliability_fusion:
            if audio_quality is None or text_quality is None: raise ValueError('Reliability fusion requires modality qualities')
            aq=audio_quality.float().clamp(0,1); tq=text_quality.float().clamp(0,1); z=(aq+tq).clamp_min(1e-6)
            wa=(aq/z).unsqueeze(-1); wt=(tq/z).unsqueeze(-1); a=a*wa; t=t*wt
        return self.classifier(torch.cat([a,t],dim=-1)).squeeze(-1)


def parameter_count(model):
    return {'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
