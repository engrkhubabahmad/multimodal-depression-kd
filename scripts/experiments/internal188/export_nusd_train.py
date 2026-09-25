"""Frozen selected NUSD TRAIN-107 inference; writes KD targets without TEST."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .nusd_ecapa import (configure_nusd, discover_checkpoint_run,
                         export_nusd_train_targets)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--daic-root",required=True,type=Path)
    p.add_argument("--experiment",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path); a=p.parse_args(argv)
    frozen=a.experiment/"teachers"/"nusd_ecapa_frozen"
    run,_=discover_checkpoint_run(frozen/"nusd_audio_cache"/"raw_svn_exp")
    selection_path=frozen/"evaluation"/"all_five"/"selection.json"
    selection=json.loads(selection_path.read_text())
    selected=int(selection["selected"]["run"])
    if selected not in range(1,6):raise ValueError("Invalid selected NUSD run")
    target=frozen/"evaluation"/"train_audio_targets.csv"
    raw=run/"results_dict_fscore_train"/"0.pickle"
    saved=frozen/"evaluation"/"all_five"/f"run_{selected}"/"train_raw_predictions.pickle"
    # NUSD overwrites 0.pickle on every run. Restore only the selected run's saved result.
    if saved.is_file():
        raw.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(saved,raw)
    else:
        raw.unlink(missing_ok=True)
    if not raw.is_file():
        configure_nusd(a.source,a.daic_root,frozen,frozen/"metadata",run,1)
        env=os.environ.copy(); env.update(DAIC_NUSD_RUN_INDEX=str(selected),DAIC_NUSD_FROZEN_EVAL="1",
                                          DAIC_NUSD_EVAL_SPLIT="train",PYTHONUNBUFFERED="1")
        cmd=[sys.executable,"-u","main_disent_fscore_grad.py","test","--validate","--cuda",
             "--prediction_metric=1","--threshold=fscore"]
        log=frozen/"evaluation"/"train_inference.log";log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w") as f:
            proc=subprocess.Popen(cmd,cwd=a.source,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:print(line,end="",flush=True);f.write(line);f.flush()
            code=proc.wait()
        if code:raise RuntimeError(f"NUSD TRAIN inference exited {code}; see {log}")
        if not raw.is_file():raise FileNotFoundError(raw)
        saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(raw,saved)
    result=export_nusd_train_targets(run,a.experiment/"split"/"manifest.csv",target.parent)
    y=result.label.to_numpy(int); pred=(result.audio_probability.to_numpy(float)>=.5).astype(int)
    metrics={"n":len(result),"accuracy":float(accuracy_score(y,pred)),
             "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
             "depressed_f1":float(f1_score(y,pred,pos_label=1,zero_division=0)),
             "confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),
             "in_sample":True,"test_opened":False}
    metrics["selected_run"]=selected
    metrics["selection_file"]=str(selection_path)
    audit_path=target.parent/"train_audio_audit.json"
    audit=json.loads(audit_path.read_text());audit["selected_run"]=selected
    audit["raw_prediction_file"]=str(saved)
    audit_path.write_text(json.dumps(audit,indent=2)+"\n")
    (target.parent/"train_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(f"Frozen NUSD TRAIN targets: {len(result)} participant rows -> {target}")
    print("TRAIN participant mean-probability metrics:",json.dumps(metrics,indent=2))


if __name__=="__main__":main()
