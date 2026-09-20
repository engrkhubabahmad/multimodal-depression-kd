from __future__ import annotations
import argparse,importlib.util,json,pickle,sys,types
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score

def one(root,p):
    h=list(Path(root).rglob(p))
    if len(h)!=1: raise FileNotFoundError(f"Expected one {p}; found {len(h)}")
    return h[0]

def load_author(path):
    if "optuna" not in sys.modules: sys.modules["optuna"]=types.ModuleType("optuna")
    s=importlib.util.spec_from_file_location("idiap_author_main",path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def local_labels(path):
    d=pd.read_csv(path); c={x.lower():x for x in d.columns}
    return {int(r[c["participant_id"]]):int(r[c["phq8_binary"]]) for _,r in d.iterrows()}

def metric(y,p,q):
    return {"n":len(y),"accuracy":float(accuracy_score(y,p)),
            "macro_f1":float(classification_report(y,p,output_dict=True,zero_division=0)["macro avg"]["f1-score"]),
            "auroc":float(roc_auc_score(y,q)),"confusion_matrix":confusion_matrix(y,p,labels=[0,1]).tolist()}

def train_probs(model,pos):
    model=model.cpu().eval(); v=len(model._vocab)
    with torch.no_grad():
        h1=model.get_H_1(model.Conv_0.cpu()); z=model.node_emb2out(model.A_norm.cpu()@h1)[v:]; q=torch.softmax(z,dim=-1)[:,pos]
    return q.numpy()

def dev_probs(model,state,pos,keep):
    model=model.cpu().eval(); v=len(model._vocab); A=state["A_dev"].cpu()[keep]
    H0=torch.cat([torch.eye(v),A[:,:v]]); B=torch.zeros((len(keep),v+len(keep))); B[:,:v]=A[:,:v]; B[:,v:]=torch.eye(len(keep))
    with torch.no_grad():
        d=model.get_H_1(B@H0); h=torch.cat([model.H_1_words[:v].cpu(),d]); q=torch.softmax(model.node_emb2out(B@h),dim=-1)[:,pos]
    return q.numpy()

def rows(ids,q,ymap,author_y=None):
    out=[]
    for i,(pid,p) in enumerate(zip(ids,q)):
        p=float(p); z=float(np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)))
        r={"participant_id":int(pid),"label":int(ymap[int(pid)]),"text_probability":p,"text_logit":z,"text_prediction":int(p>=.5)}
        if author_y is not None: r["author_label"]=int(author_y[i]); r["author_local_label_mismatch"]=int(author_y[i])!=int(ymap[int(pid)])
        out.append(r)
    return pd.DataFrame(out)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--daic-root",required=True); p.add_argument("--source-root",required=True); p.add_argument("--output",required=True); p.add_argument("--exclude-dev",type=int,nargs="*",default=[440])
    a=p.parse_args(argv); root,src,out=Path(a.daic_root),Path(a.source_root),Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tr_y=local_labels(one(root,"train_split_Depression_AVEC2017.csv")); dv_y=local_labels(one(root,"dev_split_Depression_AVEC2017.csv"))
    tr_ids=pd.read_csv(src/"data/AVEC_16_data/train_IDS.txt",sep="\t"); dv_ids=pd.read_csv(src/"data/AVEC_16_data/dev_IDS.txt",sep="\t")
    ids_tr=tr_ids.original_ID.astype(int).tolist(); ids_dv_all=dv_ids.original_ID.astype(int).tolist(); keep=[i for i,x in enumerate(ids_dv_all) if x not in set(a.exclude_dev)]; ids_dv=[ids_dv_all[i] for i in keep]
    if len(ids_tr)!=107 or len(ids_dv)!=34: raise AssertionError(f"Expected TRAIN=107 DEV=34; got {len(ids_tr)} {len(ids_dv)}")
    author=load_author(src/"main.py"); author.DEVICE=torch.device("cpu")
    with (src/"model/Participant/vtzer_inductgcn[250].pkl").open("rb") as f: vt=pickle.load(f)
    state=torch.load(src/"model/Participant/model_inductgcn[250].pkl",map_location="cpu",weights_only=False)
    model=author.InducTGCN(state["embedding_dim"],state["classes_"],0,vt); model.load_state_dict(dict(state["model_state_dict"])); model.A_B=state["A_dev"]; model.classes_=state["classes_"]
    classes=list(model.classes_); pos=classes.index("positive")
    qt=train_probs(model,pos); qd=dev_probs(model,state,pos,keep)
    tr=rows(ids_tr,qt,tr_y,tr_ids.category.astype(int).tolist()); dv=rows(ids_dv,qd,dv_y)
    tr.to_csv(out/"train_text_kd_targets.csv",index=False); dv.to_csv(out/"dev_text_predictions.csv",index=False)
    mt=metric(tr.label,tr.text_prediction,tr.text_probability); md=metric(dv.label,dv.text_prediction,dv.text_probability)
    s={"teacher":"idiap participant InducT-GCN top250","train_view":"checkpoint training graph document nodes; in-sample KD targets","dev_view":"checkpoint saved A_dev; participant 440 excluded",
       "train_participants":107,"dev_participants":34,"test_opened":False,"author_local_train_label_mismatches":tr.loc[tr.author_local_label_mismatch,"participant_id"].astype(int).tolist(),
       "train_in_sample":mt,"dev34":md}
    (out/"text_target_audit.json").write_text(json.dumps(s,indent=2)+"\n"); print(json.dumps(s,indent=2)); return s
if __name__=="__main__": main()
