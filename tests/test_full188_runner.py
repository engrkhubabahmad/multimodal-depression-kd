"""Completed feature extraction must not copy the full OpenSMILE release again."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.experiments.full188.run_all import main


class Runner(unittest.TestCase):
    def test_completed_features_skip_without_opensmile(self):
        with tempfile.TemporaryDirectory() as tmp:
            daic = Path(tmp) / "DAIC_WOZ"
            paths = (
                daic / "metadata/full_test_split.csv",
                daic / "metadata/train_split_Depression_AVEC2017.csv",
                daic / "metadata/dev_split_Depression_AVEC2017.csv",
                daic / "experiments/ussd_compare16/participant_manifest.csv",
                daic.parent / "tools/solo_teacher_sources/bias_in_daic-woz/main.py",
                daic / "experiments/full188_80_10_10_seed42/features/provenance.json",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch("scripts.experiments.full188.run_all.shutil.copytree") as copy:
                main(["--daic-root", str(daic), "--stage", "features", "--seed", "42"])
                copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
