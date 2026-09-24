import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.experiments.full188.cleanup_previous import main
from scripts.experiments.full188.split import digest


class Cleanup(unittest.TestCase):
    def test_refuses_old_cache_dependency_then_removes_only_old_runs(self):
        with tempfile.TemporaryDirectory() as td:
            daic = Path(td) / "DAIC_WOZ"
            exp = daic / "experiments"
            new = exp / "full188_80_10_10_seed103"
            old = exp / "full188_seed42"
            old_feature = old / "features/old.npy"
            old_feature.parent.mkdir(parents=True)
            old_feature.touch()
            (exp / "full188_seed103").mkdir()
            (exp / "ussd_compare16").mkdir()
            (new / "features").mkdir(parents=True)
            (new / "split").mkdir()
            (new / "features/provenance.json").write_text("{}")
            ids = range(188)
            pd.DataFrame({"participant_id": ids}).to_csv(new / "split/manifest.csv", index=False)
            (new / "split/complete.json").write_text(json.dumps({"signature": {"split": [150, 19, 19]},
                "manifest_sha256": digest(new / "split/manifest.csv")}))
            cache = pd.DataFrame({"participant_id": ids, "feature_path": str(old_feature)})
            path = new / "features/participant_manifest.csv"
            cache.to_csv(path, index=False)
            args = ["--daic-root", str(daic), "--seed", "103"]
            with self.assertRaisesRegex(ValueError, "depends on old experiment"):
                main(args)
            self.assertTrue(old.exists())
            shared = exp / "ussd_compare16/shared.npy"
            shared.touch()
            cache["feature_path"] = str(shared)
            cache.to_csv(path, index=False)
            main(args)
            self.assertFalse(old.exists())
            self.assertFalse((exp / "full188_seed103").exists())
            self.assertTrue(shared.exists())
            self.assertTrue((new / "previous_cleanup.json").exists())


if __name__ == "__main__":
    unittest.main()
