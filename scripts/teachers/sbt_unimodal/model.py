from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AlbertModel, Wav2Vec2Model

def masked_mean(x, mask):
    if mask is None: return x.mean(1)
    m=mask.to(x.dtype).unsqueeze(-1)
    return (x*m).sum(1)/m.sum(1).clamp(min=1)

class AudioSBTTeacher(nn.Module):
    """Unimodal audio teacher derived from SBT-Net's released Wav2Vec2 encoder + classifier head."""
    def __init__(self, model_name="facebook/wav2vec2-base", revision=None):
        super().__init__()
        self.encoder=Wav2Vec2Model.from_pretrained(model_name, revision=revision)
        h=self.encoder.config.hidden_size
        self.classifier=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))

    def encode(self,input_values,attention_mask=None):
        # Match unpadded demo waveforms. Batch padding must not affect group norm.
        if attention_mask is None:
            return self.encoder(input_values=input_values).last_hidden_state.mean(1)
        lengths=attention_mask.sum(1).long()
        rows=[None]*len(lengths)
        for length in lengths.unique():
            indices=(lengths==length).nonzero(as_tuple=True)[0]
            hidden=self.encoder(input_values=input_values[indices,:int(length)]).last_hidden_state
            pooled=hidden.mean(1)
            for j,index in enumerate(indices.tolist()): rows[index]=pooled[j]
        return torch.stack(rows)

    def forward(self,input_values,attention_mask=None):
        return self.classifier(self.encode(input_values,attention_mask)).squeeze(-1)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def unfreeze_top(self):
        self.encoder.requires_grad_(False)
        layers=self.encoder.encoder.layers
        for layer in layers[-2:]: layer.requires_grad_(True)

    def encoder_parameters_trainable(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

class TextSBTTeacher(nn.Module):
    """Unimodal text teacher using the released ALBERT-base encoder + released-style classifier head."""
    def __init__(self, model_name="albert-base-v2", revision=None):
        super().__init__()
        self.encoder=AlbertModel.from_pretrained(model_name, revision=revision)
        h=self.encoder.config.hidden_size
        self.classifier=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))

    def encode(self,input_ids,attention_mask,window_owner=None,n_samples=None):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        pooled=out.last_hidden_state[:,0]
        if window_owner is not None:
            owner=window_owner.to(pooled.device)
            result=pooled.new_zeros((n_samples,pooled.shape[-1]))
            result.index_add_(0,owner,pooled)
            counts=torch.bincount(owner,minlength=n_samples).to(pooled.dtype).unsqueeze(1)
            pooled=result/counts.clamp(min=1)
        return pooled

    def forward(self,input_ids,attention_mask,window_owner=None,n_samples=None):
        return self.classifier(self.encode(input_ids,attention_mask,window_owner,n_samples)).squeeze(-1)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def unfreeze_top(self):
        self.encoder.requires_grad_(False)
        # ALBERT shares parameters across logical layers. Unfreezing the last physical
        # layer-group is the closest HF implementation of the paper's "top layers" stage.
        groups=self.encoder.encoder.albert_layer_groups
        groups[-1].requires_grad_(True)
        # Pooler is unused: predictions use last_hidden_state, not pooler_output.

    def encoder_parameters_trainable(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]
