"""Small probability-fusion gate. Training objective sees TRAIN rows only."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def gate_features(logits,stability,scale):
    normalized=np.abs(logits)/np.maximum(scale,1e-4)
    return np.column_stack([np.ones(len(logits)),np.clip(normalized[:,0]-normalized[:,1],-3,3),
                            np.clip(stability[:,0]-stability[:,1],-1,1)])


def fuse(theta,features,probabilities,availability=None,prior=.5):
    w=expit(features@theta);weight=np.column_stack([w,1-w])
    if availability is not None:weight=weight*np.asarray(availability)
    mass=weight.sum(1);q=(weight*probabilities).sum(1)/np.maximum(mass,1e-12)
    return np.where(mass>0,q,prior)


def soft_targets(teacher_probability,mode):
    p=np.clip(teacher_probability,1e-6,1-1e-6)
    if mode=='ra_kd':
        entropy=-(p*np.log(p)+(1-p)*np.log(1-p))/np.log(2)
        reliability=np.maximum(1-entropy,.05)
        return (p*reliability).sum(1)/reliability.sum(1)
    return p.mean(1)


def objective(theta,features,probabilities,y,target,kd_weight,regularization=.01):
    w=expit(features@theta);q=np.clip(w*probabilities[:,0]+(1-w)*probabilities[:,1],1e-7,1-1e-7)
    n=len(y);count=np.bincount(y.astype(int),minlength=2)
    if np.any(count==0):raise ValueError('Both TRAIN classes required')
    cw=n/(2*count);sample_weight=cw[y.astype(int)]
    bce=lambda target:-(target*np.log(q)+(1-target)*np.log(1-q))
    loss=(1-kd_weight)*np.mean(sample_weight*bce(y))+kd_weight*np.mean(bce(target))
    dq=((1-kd_weight)*sample_weight*(q-y)+kd_weight*(q-target))/(q*(1-q))
    grad=features.T@(dq*(probabilities[:,0]-probabilities[:,1])*w*(1-w))/n
    return float(loss+regularization*np.dot(theta,theta)),grad+2*regularization*theta


def fit(features,probabilities,y,teacher,mode,callback=None):
    target=soft_targets(teacher,mode);alpha=0 if mode=='plain' else .3
    theta=np.zeros(features.shape[1]);steps=[0]
    def report(x):
        steps[0]+=1
        if callback is not None:callback(steps[0],x,objective(x,features,probabilities,y,target,alpha)[0])
    result=minimize(objective,theta,args=(features,probabilities,y,target,alpha),jac=True,
        method='L-BFGS-B',callback=report,options={'maxiter':100,'gtol':1e-9,'ftol':1e-12})
    if not np.isfinite(result.x).all():raise ValueError('Nonfinite fitted gate')
    return result.x,{'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
                    'iterations':int(result.nit),'train_objective':float(result.fun),
                    'selection':'TRAIN objective only; no DEV checkpoint selection for gate'}
