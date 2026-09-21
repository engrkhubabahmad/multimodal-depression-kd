from __future__ import annotations
import torch
from torch import nn

class CompactAudioBranch(nn.Module):
    """Teacher-aligned ComParE16 segment encoder, compressed from 256x2 LSTM to 128x1."""
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(130,128,3,1,1); self.bn=nn.BatchNorm1d(128); self.act=nn.ReLU()
        self.pool=nn.MaxPool1d(3,3); self.drop=nn.Dropout(.10)
        self.lstm=nn.LSTM(128,128,num_layers=1,batch_first=True,bidirectional=False)
        self.head=nn.Linear(128,1)
    def forward(self,x):
        x=self.drop(self.pool(self.act(self.bn(self.conv(x))))).transpose(1,2)
        _,(h,_)=self.lstm(x); emb=h[-1]; logit=self.head(emb).squeeze(-1)
        return logit,emb

def parameter_count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
