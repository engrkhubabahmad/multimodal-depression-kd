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
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.encoder=Wav2Vec2Model.from_pretrained(model_name)
        h=self.encoder.config.hidden_size
        self.classifier=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))

    def forward(self,input_values,attention_mask=None):
        out=self.encoder(input_values=input_values,attention_mask=attention_mask)
        fm=None
        if attention_mask is not None:
            fm=self.encoder._get_feature_vector_attention_mask(out.last_hidden_state.shape[1],attention_mask)
        return self.classifier(masked_mean(out.last_hidden_state,fm)).squeeze(-1)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def unfreeze_top(self):
        self.encoder.requires_grad_(False)
        layers=self.encoder.encoder.layers
        for layer in layers[-2:]: layer.requires_grad_(True)

    def encoder_parameters_trainable(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

class TextSBTTeacher(nn.Module):
    """Unimodal text teacher using the paper's ALBERT-large encoder + released-style classifier head."""
    def __init__(self, model_name="albert-large-v2"):
        super().__init__()
        self.encoder=AlbertModel.from_pretrained(model_name)
        h=self.encoder.config.hidden_size
        self.classifier=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,1))

    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        pooled=out.last_hidden_state[:,0]
        return self.classifier(pooled).squeeze(-1)

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)

    def unfreeze_top(self):
        self.encoder.requires_grad_(False)
        # ALBERT shares parameters across logical layers. Unfreezing the last physical
        # layer-group is the closest HF implementation of the paper's "top layers" stage.
        groups=self.encoder.encoder.albert_layer_groups
        groups[-1].requires_grad_(True)
        if self.encoder.pooler is not None:
            self.encoder.pooler.requires_grad_(True)

    def encoder_parameters_trainable(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]
