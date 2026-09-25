"""Freeze canonical DAIC-WOZ CV TRAIN/DEV and reserve TEST (participant 440 excluded)."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .split import digest, table


def build(root):
    metadata=Path(root)/'metadata'
    train=table(metadata/'train_split_Depression_AVEC2017.csv','participant_id','phq8_binary')
    dev_all=table(metadata/'dev_split_Depression_AVEC2017.csv','participant_id','phq8_binary')
    dev=dev_all.loc[dev_all.participant_id.ne(440)].copy()
    test_ids=table(metadata/'test_split_Depression_AVEC2017.csv','participant_id')
    test=table(metadata/'full_test_split.csv','participant_id','phq_binary')
    raw_test=pd.read_csv(metadata/'full_test_split.csv');raw_test.columns=raw_test.columns.str.strip().str.lower()
    if set(test_ids.participant_id)!=set(test.participant_id): raise ValueError('Official TEST roster and full-test labels differ')
    if 'phq_score' not in raw_test or not ((raw_test.phq_score.to_numpy(float)>=10).astype(int)==raw_test.phq_binary.to_numpy(int)).all():
        raise ValueError('Official TEST binary labels disagree with PHQ threshold 10')
    if 440 not in set(dev_all.participant_id): raise ValueError('Expected corrupt DEV participant 440')
    if (len(train),len(dev_all),len(dev),len(test))!=(107,35,34,47): raise ValueError('Unexpected official split counts')
    parts=[train.assign(split='train',source_split='canonical_train'),dev.assign(split='val',source_split='canonical_dev'),test.assign(split='student_test',source_split='canonical_test')]
    out=pd.concat(parts,ignore_index=True).sort_values('participant_id').reset_index(drop=True)
    if len(out)!=188 or out.participant_id.nunique()!=188 or 440 in set(out.participant_id): raise ValueError('Expected 188 unique labeled participants excluding 440')
    if set(out.split.value_counts().to_dict().items())!={('train',107),('val',34),('student_test',47)}: raise ValueError('Unexpected split sizes')
    sets=[set(x.participant_id) for x in parts]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError('Canonical split overlap')
    return out,{'train_split_Depression_AVEC2017.csv':metadata/'train_split_Depression_AVEC2017.csv','dev_split_Depression_AVEC2017.csv':metadata/'dev_split_Depression_AVEC2017.csv','test_split_Depression_AVEC2017.csv':metadata/'test_split_Depression_AVEC2017.csv','full_test_split.csv':metadata/'full_test_split.csv'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--daic-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    frame,sources=build(a.daic_root);signature={'protocol':'official_train107_dev34_test47_exclude440','source_sha256':{k:digest(v) for k,v in sources.items()},'code_sha256':digest(Path(__file__))}
    marker=a.output/'complete.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('signature')!=signature or digest(a.output/'manifest.csv')!=saved.get('manifest_sha256'): raise ValueError('Existing official split differs; choose a new output folder')
        print('Verified existing canonical 107/34/47 split:',a.output);return frame
    if a.output.exists() and any(a.output.iterdir()): raise ValueError('Refusing to overwrite nonempty split folder')
    a.output.mkdir(parents=True,exist_ok=True);frame.to_csv(a.output/'manifest.csv',index=False)
    marker.write_text(json.dumps({'signature':signature,'manifest_sha256':digest(a.output/'manifest.csv'),'class_counts':{k:g.label.value_counts().sort_index().to_dict() for k,g in frame.groupby('split')},'test_media_opened':False,'test_label_source':'full_test_split.csv metadata only','excluded_participant_id':440},indent=2)+'\n')
    print('Created canonical split:',frame.split.value_counts().to_dict());return frame

if __name__=='__main__':main()
