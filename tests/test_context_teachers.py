"""Synthetic checks: no DAIC data, encoder downloads, or Colab required."""
import importlib.util, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.teachers.context_utils import (participant_weights,window_indices,materialize_windows,
    validate_manifest,align_predictions)


def fixture():
    rows=[]
    for pid,split,label in [(1,'train',0),(2,'train',1),(3,'dev',0),(4,'dev',1)]:
        for turn in range(4):
            rows.append(dict(segment_id=f'{pid}_{turn:04d}_00',participant_id=pid,split=split,label=label,
                             start=turn*5.,stop=turn*5.+2,source_turn=turn,part=0,parts=1))
    return pd.DataFrame(rows)


class ContextTests(unittest.TestCase):
    def test_weights_balance_people_and_classes(self):
        m=pd.DataFrame({'participant_id':[1]*2+[2]*4+[3]*3,'label':[0]*6+[1]*3})
        w=participant_weights(m); self.assertAlmostEqual(float(w.mean()),1.)
        self.assertAlmostEqual(float(w[:6].sum()),float(w[6:].sum()),places=5)
        self.assertAlmostEqual(float(w[:2].sum()),float(w[2:6].sum()),places=5)

    def test_windows_preserve_centers_and_boundaries(self):
        m=fixture().sample(frac=1,random_state=2).reset_index(drop=True); idx,c=window_indices(m)
        for i,row in enumerate(idx):
            valid=row[row>=0]; self.assertEqual(row[c[i]],i)
            self.assertTrue((m.iloc[valid].participant_id==m.iloc[i].participant_id).all())
            self.assertTrue((m.iloc[valid].split==m.iloc[i].split).all())
        values,mask=materialize_windows(np.ones((len(m),3),np.float32),idx)
        self.assertTrue((values[~mask]==0).all()); self.assertTrue((values[mask]==1).all())

    def test_skipped_turn_and_long_gap_break_context(self):
        m=fixture().iloc[:4].copy(); m.loc[2:,'source_turn']+=1
        idx,_=window_indices(m); self.assertNotIn(2,idx[1]); self.assertNotIn(1,idx[2])
        m=fixture().iloc[:4].copy(); m.loc[2:,['start','stop']]+=100
        idx,_=window_indices(m,max_gap=30); self.assertNotIn(2,idx[1])

    def test_skipped_parts_break_context(self):
        m=fixture().iloc[:3].copy(); m['source_turn']=0; m['part']=[0,2,3]; m['parts']=4
        idx,_=window_indices(m); self.assertNotIn(1,idx[0]); self.assertIn(2,idx[1])

    def test_split_and_label_guards(self):
        m=fixture(); tr=m.loc[m.split.eq('train'),['participant_id','label']].drop_duplicates()
        dv=m.loc[m.split.eq('dev'),['participant_id','label']].drop_duplicates()
        validate_manifest(m,tr,dv,[5])
        with self.assertRaises(ValueError): validate_manifest(m,tr,dv,[1])
        bad=m.copy(); bad.loc[0,'label']=1
        with self.assertRaises(ValueError): validate_manifest(bad,tr,dv,[5])

    def test_predictions_reordered_and_bad_labels_rejected(self):
        m=fixture(); d=m.copy(); d['probability']=np.linspace(.1,.9,len(m))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'predictions.csv'; d.iloc[::-1].to_csv(p,index=False)
            np.testing.assert_allclose(align_predictions(p,m),d.probability)
            d.loc[0,'label']=1; d.to_csv(p,index=False)
            with self.assertRaises(ValueError): align_predictions(p,m)

    def test_notebook_arguments_are_explicit(self):
        from unittest.mock import patch
        from scripts.teachers.train_context_teachers import parse_args
        with patch('sys.argv',['ipykernel_launcher','-f','kernel.json']):
            self.assertEqual(parse_args([]).window,5)

    @unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow unavailable in this test environment')
    def test_keras_training_and_checkpoint_roundtrip(self):
        import tensorflow as tf
        from scripts.teachers.train_context_teachers import build_model
        rng=np.random.default_rng(103); x=rng.normal(size=(8,6)).astype(np.float32); y=np.array([0,1]*4,np.float32).reshape(-1,1)
        for kind in ['balanced_mlp','context_gru']:
            tf.keras.backend.clear_session(); model=build_model(tf,6,kind,5,3e-4)
            inputs=x
            if kind=='context_gru':
                context=np.repeat(x[:,None,:],5,axis=1); mask=np.ones((8,5),bool)
                mask[:,-2:]=False; context[:,-2:]=0
                inputs={'target':x,'context':context,'mask':mask}
            loss=model.train_on_batch(inputs,y,sample_weight=np.ones(8))
            self.assertTrue(np.isfinite(loss).all())
            before=model(inputs,training=False).numpy()
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'teacher.keras'; model.save(p); restored=tf.keras.models.load_model(p)
                np.testing.assert_allclose(before,restored(inputs,training=False).numpy(),atol=1e-6)


if __name__=='__main__': unittest.main()
