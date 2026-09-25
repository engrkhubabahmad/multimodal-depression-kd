"""Frozen DEV-34 evaluation of all five released NUSD ECAPA runs."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
from .nusd_ecapa import configure_nusd, discover_checkpoint_run, export_validation_predictions


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--daic-root",required=True,type=Path)
    p.add_argument("--experiment",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path); a=p.parse_args(argv)
    frozen=a.experiment/"teachers"/"nusd_ecapa_frozen"
    run,count=discover_checkpoint_run(frozen/"nusd_audio_cache"/"raw_svn_exp")
    if count!=5:raise ValueError("Expected five released NUSD ECAPA runs")
    manifest=a.experiment/"split"/"manifest.csv"
    split=pd.read_csv(manifest)
    if split.split.value_counts().to_dict()!={"train":107,"val":34,"student_test":47}:
        raise ValueError("Unexpected canonical split")
    output=frozen/"evaluation"/"all_five";output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index in range(1,6):
        folder=output/f"run_{index}";folder.mkdir(parents=True,exist_ok=True)
        raw=folder/"raw_predictions.pickle"
        if not raw.is_file():
            configure_nusd(a.source,a.daic_root,frozen,frozen/"metadata",run,1)
            env=os.environ.copy();env.update(DAIC_NUSD_RUN_INDEX=str(index),DAIC_NUSD_FROZEN_EVAL="1",PYTHONUNBUFFERED="1")
            env.pop("DAIC_NUSD_EVAL_SPLIT",None)
            cmd=[sys.executable,"-u","main_disent_fscore_grad.py","test","--validate","--cuda",
                 "--prediction_metric=1","--threshold=fscore"]
            source=run/"results_dict_fscore"/"0.pickle"
            source.unlink(missing_ok=True)
            with (folder/"inference.log").open("w") as log:
                proc=subprocess.Popen(cmd,cwd=a.source,env=env,stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in proc.stdout:print(line,end="",flush=True);log.write(line);log.flush()
                code=proc.wait()
            if code:raise RuntimeError(f"NUSD run {index} exited {code}; inspect {folder/'inference.log'}")
            source=run/"results_dict_fscore"/"0.pickle"
            if not source.is_file():raise FileNotFoundError(source)
            shutil.copy2(source,raw)
        metric=export_validation_predictions(run,folder,manifest,1,prediction_file=raw)
        rows.append({"run":index,"macro_f1":metric["macro_f1"],
                     "depressed_f1":metric["depressed_f1"],"auroc":metric["auroc"],
                     "confusion_matrix":metric["confusion_matrix"]})
        print("NUSD DEV-34 run",index,rows[-1],flush=True)
    best=max(rows,key=lambda r:(r["macro_f1"],r["depressed_f1"],r["auroc"],-r["run"]))
    (output/"selection.json").write_text(json.dumps({"criterion":"DEV-34 participant mean-probability macro F1, then depressed F1, then AUROC",
        "selected":best,"all_runs":rows,"test_opened":False,
        "limitation":"DEV was used for checkpoint selection; selected DEV metric is exploratory"},indent=2)+"\n")
    print("SELECTED FROZEN NUSD AUDIO TEACHER:",best,"| TEST CLOSED",flush=True)


if __name__=="__main__":main()
