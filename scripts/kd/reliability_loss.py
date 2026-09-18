"""Reliability-aware binary KD utilities for the segment-level student."""
import torch
import torch.nn.functional as F

def normalize_reliability(audio_r,text_r,eps=1e-6):
    a=audio_r.clamp_min(eps); t=text_r.clamp_min(eps); z=a+t
    return a/z,t/z

def reliability_aware_kd_loss(student_logits,hard_labels,audio_prob,text_prob,audio_r,text_r,
                              temperature=2.0,lambda_kd=0.5):
    """Hard BCE + reliability-weighted dual-teacher binary KD on aligned segments."""
    hard_labels=hard_labels.float()
    wa,wt=normalize_reliability(audio_r.float(),text_r.float())
    target=(wa*audio_prob.float()+wt*text_prob.float()).clamp(1e-6,1-1e-6)
    hard=F.binary_cross_entropy_with_logits(student_logits.float(),hard_labels)
    teacher_logit=torch.logit(target)
    soft_target=torch.sigmoid(teacher_logit/temperature)
    kd=F.binary_cross_entropy_with_logits(student_logits.float()/temperature,soft_target)*(temperature**2)
    total=(1-lambda_kd)*hard+lambda_kd*kd
    return total,{'hard_loss':hard.detach(),'kd_loss':kd.detach(),
                  'audio_weight':wa.detach().mean(),'text_weight':wt.detach().mean()}
