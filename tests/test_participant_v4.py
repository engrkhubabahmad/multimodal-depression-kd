import contextlib
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.students.participant_student_v4 import export_text, compact_fusion, split42_baseline, split42_ra, score_split42_student
from scripts.students.participant_student_v4.common import (
    cached_or_create, complete, load_embeddings, check_splits)


class Improvements(unittest.TestCase):
    def test_inductive_algebra_matches_dense_graph(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(7, 9)).astype('float32')
        w1 = rng.normal(size=(4, 9)).astype('float32')
        wo = rng.normal(size=(2, 4)).astype('float32')
        words = rng.normal(size=(9, 4)).astype('float32')
        b = np.concatenate([x, np.eye(7, dtype='float32')], 1)
        h0 = np.concatenate([np.eye(9, dtype='float32'), x])
        dense = b @ np.concatenate([words, np.maximum((b @ h0) @ w1.T, 0)])
        e, p = export_text.inductive_forward(x, w1, wo, words)
        np.testing.assert_allclose(e, dense, atol=2e-6)
        logits = dense @ wo.T
        np.testing.assert_allclose(p, expit(logits[:, 1] - logits[:, 0]), atol=1e-6)

    def test_head_train_scaling_and_roundtrip(self):
        rng = np.random.default_rng(7); x = rng.normal(size=(30, 5)); x[:, -1] = 2
        y = np.tile([0, 1], 15); base = np.zeros(30)
        h, info = compact_fusion.fit_head(x, y, base)
        np.testing.assert_allclose(h['mean'], x.mean(0))
        self.assertEqual(h['scale'][-1], 1.)
        self.assertTrue(info['converged'])
        self.assertLess(info['gradient_inf_norm'], 1e-5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'head.npz'; np.savez(path, **h)
            with np.load(path) as z:
                np.testing.assert_array_equal(compact_fusion.predict(h, x, base),
                                              compact_fusion.predict(z, x, base))

    def test_alignment_and_cache_fail_closed(self):
        with self.assertRaises(ValueError):
            compact_fusion.aligned({'participant_ids': np.array([1]), 'labels': np.array([0])},
                                   {'participant_ids': np.array([2]), 'labels': np.array([0])})
        with self.assertRaises(ValueError):
            check_splits({'participant_ids': [1]}, {'participant_ids': [1]})
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'run'; self.assertFalse(cached_or_create(out, {'x': 1}))
            (out / 'data').write_text('original'); complete(out, {'x': 1})
            self.assertTrue(cached_or_create(out, {'x': 1}))
            with self.assertRaises(ValueError): cached_or_create(out, {'x': 2})
            (out / 'data').write_text('changed')
            with self.assertRaises(ValueError): cached_or_create(out, {'x': 1})

    def make_fixture(self, root):
        source = root / 'source'; source.mkdir(); (root / 'metadata').mkdir()
        audio = root / 'audio'; audio.mkdir()
        rng = np.random.default_rng(11)
        v = TfidfVectorizer().fit(['happy sleep', 'sad tired', 'happy tired'])
        weights = (rng.normal(size=(3, 4)).astype('float32'),
                   rng.normal(size=(2, 3)).astype('float32'),
                   rng.normal(size=(4, 3)).astype('float32'))
        for split, start, counts in [('train', 1000, [77, 30]), ('dev', 2000, [23, 11])]:
            ids = np.arange(start, start + sum(counts)); labels = np.repeat([0, 1], counts)
            pd.DataFrame({'Participant_ID': ids, 'PHQ8_Binary': labels}).to_csv(
                root / 'metadata' / f'{split}_split_Depression_AVEC2017.csv', index=False)
            docs = ['happy sleep' if i % 3 else 'sad tired' for i in ids]
            for pid, doc in zip(ids, docs):
                (root / f'{pid}_TRANSCRIPT.csv').write_text('speaker\tvalue\nEllie\tignored\nParticipant\t' + doc + '\n')
            e, p = export_text.inductive_forward(v.transform(docs).toarray(), *weights)
            np.savez(source / f'{split}_text_embeddings.npz', participant_ids=ids,
                     labels=labels, embedding=e, probability=p)
            np.savez(audio / f'{split}_audio_embeddings.npz', participant_ids=ids[::-1],
                     labels=labels[::-1], embedding=rng.normal(size=(len(ids), 5)), probability=np.full(len(ids), .4))
        with (source / 'vectorizer.pkl').open('wb') as f: pickle.dump(v, f)
        (source / 'inference_state.pt').write_text('mock weights')
        (source / 'metrics.json').write_text('{}')
        return source, audio, weights

    def test_export_fusion_end_to_end_and_dev_label_independence(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); source, audio, weights = self.make_fixture(root)
            out = root / 'consistent'
            args = ['--daic-root', str(root), '--text-branch', str(source), '--output', str(out)]
            with patch.object(export_text, 'load_weights', return_value=weights):
                export_text.main(args); export_text.main(args)
            run = root / 'fusion'
            compact_fusion.main(['--text-export', str(out), '--audio-branch', str(audio), '--output', str(run)])
            metrics = pd.read_csv(run / 'metrics.csv'); self.assertEqual(len(metrics), 10)
            for mode in compact_fusion.MODES:
                with np.load(run / f'{mode}_head.npz') as h:
                    self.assertTrue(np.isfinite(h['theta']).all())
            # Change DEV labels in both sources, preserving class counts, and regenerate.
            for folder, modality in [(source, 'text'), (audio, 'audio')]:
                path = folder / f'dev_{modality}_embeddings.npz'
                d = load_embeddings(path); d['labels'] = np.roll(d['labels'], 1)
                np.savez(path, **d)
            split_path = root / 'metadata/dev_split_Depression_AVEC2017.csv'
            d = pd.read_csv(split_path); d['PHQ8_Binary'] = np.roll(d['PHQ8_Binary'], 1); d.to_csv(split_path, index=False)
            out2 = root / 'consistent2'
            with patch.object(export_text, 'load_weights', return_value=weights):
                export_text.main(args[:-1] + [str(out2)])
            run2 = root / 'fusion2'
            compact_fusion.main(['--text-export', str(out2), '--audio-branch', str(audio), '--output', str(run2)])
            for mode in compact_fusion.MODES:
                with np.load(run / f'{mode}_head.npz') as first, np.load(run2 / f'{mode}_head.npz') as second:
                    for key in first.files: np.testing.assert_array_equal(first[key], second[key])
            # Export must reject altered DEV embeddings rather than silently replacing them.
            path = source / 'dev_text_embeddings.npz'; data = load_embeddings(path)
            data['embedding'] += 1; np.savez(path, **data)
            with patch.object(export_text, 'load_weights', return_value=weights), self.assertRaisesRegex(ValueError, 'parity failed'):
                export_text.main(args[:-1] + [str(root / 'bad')])

    def test_seed42_fresh_split_and_holdout_remains_unscored(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td); source, _, _ = self.make_fixture(root)
            feat = root / 'features'; feat.mkdir(); rows = []
            for split, ids in [('train', range(1000, 1107)), ('dev', range(2000, 2034))]:
                metadata = pd.read_csv(root / 'metadata' / f'{split}_split_Depression_AVEC2017.csv')
                for pid, label in zip(metadata.Participant_ID, metadata.PHQ8_Binary):
                    path = feat / f'{pid}.npy'
                    np.save(path, np.full((130, 7), float(pid % 7 + label), dtype='float32'))
                    rows.append({'participant_id': pid, 'label': label, 'feature_path': str(path), 'split': split})
            pd.DataFrame(rows).to_csv(feat / 'participant_manifest.csv', index=False)
            out = root / 'fresh'
            args = ['--daic-root', str(root), '--features', str(feat), '--output', str(out), '--seed', '42']
            split42_baseline.run(args); split42_baseline.run(args)
            manifest = pd.read_csv(out / 'split_manifest.csv')
            self.assertEqual(manifest.experimental_split.value_counts().to_dict(),
                             {'train': 99, 'dev': 21, 'holdout': 21})
            self.assertEqual(len(pd.read_csv(out / 'metrics.csv')), 6)
            self.assertFalse(list(out.glob('*holdout*predictions*')))
            student = root / 'student'
            split42_ra.run(['--daic-root', str(root), '--features', str(feat),
                            '--split-baseline', str(out), '--output', str(student)])
            targets = pd.read_csv(student / 'train_teacher_logits_probabilities.csv')
            self.assertEqual(len(targets), 99)
            for name in ('audio', 'text'):
                np.testing.assert_allclose(expit(targets[f'{name}_logit']),
                                           targets[f'{name}_probability'], atol=1e-7)
            self.assertEqual(len(pd.read_csv(student / 'metrics.csv')), 6)
            self.assertEqual(len(pd.read_csv(student / 'dev_missing_noise.csv')), 369)
            self.assertFalse(list(student.glob('*holdout*predictions*')))
            final = root / 'final_student_test'
            testargs = ['--daic-root', str(root), '--features', str(feat),
                        '--split-baseline', str(out), '--student-run', str(student),
                        '--student-mode', 'ra_kd', '--allow-exploratory-proxy-test', '--output', str(final)]
            with self.assertRaisesRegex(ValueError, 'closed for the proxy'):
                score_split42_student.run([a for a in testargs if a != '--allow-exploratory-proxy-test'])
            score_split42_student.run(testargs); score_split42_student.run(testargs)
            self.assertEqual(len(pd.read_csv(final / 'student_test_predictions.csv')), 21)
            self.assertFalse(list(final.glob('*teacher*predictions*')))
            with self.assertRaisesRegex(ValueError, 'already evaluated'):
                score_split42_student.run(testargs[:testargs.index('--student-mode')]
                                           + ['--student-mode', 'standard_kd', '--allow-exploratory-proxy-test',
                                              '--output', str(root / 'other_test')])
            with self.assertRaises(ValueError):
                split42_baseline.run(args[:-1] + ['43'])


if __name__ == '__main__': unittest.main()
