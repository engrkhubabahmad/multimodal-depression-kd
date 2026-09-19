"""Report saved DEV teacher outputs only; reserved TEST stays closed."""
from pathlib import Path
import argparse
import pandas as pd
from scripts.teachers.sbt_unimodal.train import metrics

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('predictions')
    args=parser.parse_args(argv); rows=pd.read_csv(Path(args.predictions))
    if rows.empty or not rows.split.eq('dev').all():
        raise ValueError('This teacher evaluator accepts DEV predictions only')
    print(metrics(rows.label,rows.probability))

if __name__=='__main__': main()
