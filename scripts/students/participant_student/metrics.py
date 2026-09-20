from __future__ import annotations
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,roc_auc_score,average_precision_score,brier_score_loss,log_loss,confusion_matrix,classification_report

def metric_dict(y,p,threshold=.5):
    y=np.asarray(y,int); p=np.asarray(p,float); pred=(p>=threshold).astype(int)
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
        "depressed_f1":float(f1_score(y,pred,zero_division=0)),"precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "auroc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),
        "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
        "threshold":float(threshold),"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist()}

def save_bundle(frame,out,prefix,threshold=.5):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); d=frame.copy(); d["prediction"]=(d.probability>=threshold).astype(int)
    d.to_csv(out/f"{prefix}_predictions.csv",index=False); m=metric_dict(d.label,d.probability,threshold)
    pd.DataFrame(confusion_matrix(d.label,d.prediction,labels=[0,1]),index=["actual_0","actual_1"],columns=["pred_0","pred_1"]).to_csv(out/f"{prefix}_confusion_matrix.csv")
    pd.DataFrame(classification_report(d.label,d.prediction,labels=[0,1],target_names=["non_depressed","depressed"],output_dict=True,zero_division=0)).T.to_csv(out/f"{prefix}_classification_report.csv")
    (out/f"{prefix}_metrics.json").write_text(json.dumps(m,indent=2)+"\n"); return m
