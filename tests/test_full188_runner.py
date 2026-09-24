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

    def test_my_drive_shortcut_preserves_sibling_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive = Path(tmp) / "MyDrive"
            actual = Path(tmp) / ".shortcut-targets-by-id" / "dataset" / "DAIC_WOZ"
            actual.mkdir(parents=True)
            drive.mkdir()
            alias = drive / "DAIC_WOZ"
            alias.symlink_to(actual, target_is_directory=True)
            paths = (
                actual / "metadata/full_test_split.csv",
                actual / "metadata/train_split_Depression_AVEC2017.csv",
                actual / "metadata/dev_split_Depression_AVEC2017.csv",
                actual / "experiments/ussd_compare16/participant_manifest.csv",
                actual / "experiments/full188_80_10_10_seed42/features/provenance.json",
                drive / "tools/solo_teacher_sources/bias_in_daic-woz/main.py",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            main(["--daic-root", str(alias), "--stage", "features", "--seed", "42"])


if __name__ == "__main__":
    unittest.main()
