from __future__ import annotations
import torch
from torch import nn

class FrozenBranchFusion(nn.Module):
    """Compact participant-level fusion head over frozen v3 branch embeddings."""
    def __init__(self,audio_dim=256,text_dim=64,d_model=96,hidden=64,dropout=.35):
        super().__init__()
        self.audio_dim=int(audio_dim); self.text_dim=int(text_dim); self.d_model=int(d_model)
        self.audio_proj=nn.Sequential(nn.Linear(audio_dim,d_model),nn.GELU(),nn.Dropout(dropout/2))
        self.text_proj=nn.Sequential(nn.Linear(text_dim,d_model),nn.GELU(),nn.Dropout(dropout/2))
        self.fusion=nn.Sequential(
            nn.Linear(4*d_model,hidden),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(hidden,1)
        )

    def forward(self,audio_embedding,text_embedding,return_fused=False):
        a=self.audio_proj(audio_embedding); t=self.text_proj(text_embedding)
        x=torch.cat([a,t,a*t,torch.abs(a-t)],dim=-1)
        logit=self.fusion(x).squeeze(-1)
        return (logit,x) if return_fused else logit

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
