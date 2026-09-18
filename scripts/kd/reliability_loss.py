"""Binary KD losses for segment-level RA-PDS-KD students."""
import torch
import torch.nn.functional as F


def normalize_reliability(audio_r,text_r,eps=1e-6):
    a=audio_r.float().clamp_min(0); t=text_r.float().clamp_min(0); z=a+t
    both_zero=z.le(eps); z=z.clamp_min(eps)
    wa=a/z; wt=t/z
    wa=torch.where(both_zero,torch.full_like(wa,0.5),wa); wt=torch.where(both_zero,torch.full_like(wt,0.5),wt)
    return wa,wt


def binary_soft_kd(student_logits,target_probability,temperature=2.0):
    p=target_probability.float().clamp(1e-6,1-1e-6); teacher_logit=torch.logit(p)
    soft=torch.sigmoid(teacher_logit/temperature)
    return F.binary_cross_entropy_with_logits(student_logits.float()/temperature,soft)*(temperature**2)


def standard_kd_loss(student_logits,hard_labels,audio_prob,text_prob,temperature=2.0,lambda_kd=0.5):
    hard=F.binary_cross_entropy_with_logits(student_logits.float(),hard_labels.float())
    target=0.5*(audio_prob.float()+text_prob.float()); kd=binary_soft_kd(student_logits,target,temperature)
    total=(1-lambda_kd)*hard+lambda_kd*kd
    return total,{'hard_loss':hard.detach(),'kd_loss':kd.detach(),'teacher_probability':target.detach().mean()}


def reliability_aware_kd_loss(student_logits,hard_labels,audio_prob,text_prob,
                              audio_quality,text_quality,audio_confidence,text_confidence,
                              temperature=2.0,lambda_kd=0.5):
    """Quality/availability-aware dual-teacher KD.

    Reliability = input quality/availability × teacher confidence.
    Missing modality quality must be 0; noisy modality quality is in (0,1).
    """
    ar=audio_quality.float().clamp(0,1)*audio_confidence.float().clamp(0,1)
    tr=text_quality.float().clamp(0,1)*text_confidence.float().clamp(0,1)
    wa,wt=normalize_reliability(ar,tr)
    target=(wa*audio_prob.float()+wt*text_prob.float()).clamp(1e-6,1-1e-6)
    hard=F.binary_cross_entropy_with_logits(student_logits.float(),hard_labels.float())
    kd=binary_soft_kd(student_logits,target,temperature)
    total=(1-lambda_kd)*hard+lambda_kd*kd
    return total,{'hard_loss':hard.detach(),'kd_loss':kd.detach(),'audio_weight':wa.detach().mean(),
                  'text_weight':wt.detach().mean(),'teacher_probability':target.detach().mean()}
