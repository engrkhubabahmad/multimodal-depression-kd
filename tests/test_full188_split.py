import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.experiments.full188.split import main, prepare


class Split188(unittest.TestCase):
    def fixture(self, root):
        folder = root / 'metadata'; folder.mkdir()
        tr = np.arange(1000, 1107); dv = np.r_[np.arange(2000, 2034), 440]
        te = np.arange(3000, 3047)
        for name, ids in [('train', tr), ('dev', dv)]:
            pd.DataFrame({'Participant_ID': ids,
                          'PHQ8_Binary': np.arange(len(ids)) % 2}).to_csv(
                folder / f'{name}_split_Depression_AVEC2017.csv', index=False)
        pd.DataFrame({'participant_ID': te, 'Gender': 0}).to_csv(
            folder / 'test_split_Depression_AVEC2017.csv', index=False)
        label = np.arange(len(te)) % 2
        pd.DataFrame({'Participant_ID': te, 'PHQ_Binary': label,
                      'PHQ_Score': np.where(label, 12, 2)}).to_csv(folder / 'full_test_split.csv', index=False)
        return folder

    def test_exact_stratified_split_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); folder = self.fixture(root)
            first, _ = prepare(root)
            second, _ = prepare(root)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first.split.value_counts().to_dict(), {'train': 150, 'val': 19, 'student_test': 19})
            self.assertEqual(first.participant_id.nunique(), 188)
            self.assertNotIn(440, first.participant_id.values)
            out = root / 'experiment'
            main(['--daic-root', str(root), '--output', str(out)])
            main(['--daic-root', str(root), '--output', str(out)])
            self.assertEqual(json.loads((out / 'complete.json').read_text())['manifest_sha256'],
                             __import__('scripts.experiments.full188.split', fromlist=['digest']).digest(out / 'manifest.csv'))
            d = pd.read_csv(folder / 'full_test_split.csv'); d.loc[0, 'PHQ_Binary'] = 1
            d.to_csv(folder / 'full_test_split.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'PHQ threshold'):
                prepare(root)

    def test_missing_or_mismatched_test_labels_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); folder = self.fixture(root)
            (folder / 'full_test_split.csv').unlink()
            with self.assertRaises(FileNotFoundError): prepare(root)
            self.fixture_data = pd.DataFrame({'Participant_ID': range(3000, 3047), 'PHQ_Binary': 0, 'PHQ_Score': 0})
            self.fixture_data.loc[0, 'Participant_ID'] = 9999
            self.fixture_data.to_csv(folder / 'full_test_split.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'IDs differ'):
                prepare(root)

    def test_seed103_is_distinct_and_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); self.fixture(root)
            a, _ = prepare(root, 103)
            b, _ = prepare(root, 103)
            baseline, _ = prepare(root, 42)
            pd.testing.assert_frame_equal(a, b)
            self.assertEqual(a.split.value_counts().to_dict(),
                             {'train': 150, 'val': 19, 'student_test': 19})
            self.assertFalse(a.split.equals(baseline.split))


if __name__ == '__main__': unittest.main()
