import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.full188.features import main
from scripts.experiments.full188.split import digest


class CachedFeatures(unittest.TestCase):
    def test_complete_cache_needs_no_opensmile_and_becomes_independent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split = root / "new/split"; split.mkdir(parents=True)
            source = root / "old/features"; source.mkdir(parents=True)
            canonical = root / "canonical"; canonical.mkdir()
            output = root / "new/features"
            shared = canonical / "shared.npy"; np.save(shared, np.zeros((130, 1), np.float32))
            old_file = source / "old.npy"; np.save(old_file, np.zeros((130, 1), np.float32))
            frame = pd.DataFrame({"participant_id": range(188), "label": [0] * 188,
                "split": ["train"] * 150 + ["val"] * 19 + ["student_test"] * 19})
            frame.to_csv(split / "manifest.csv", index=False)
            (split / "complete.json").write_text(json.dumps({"manifest_sha256": digest(split / "manifest.csv")}))
            prior = frame[["participant_id", "label"]].copy()
            prior["feature_path"] = [str(shared)] * 141 + [str(old_file)] * 47
            prior.to_csv(source / "participant_manifest.csv", index=False)
            (source / "provenance.json").write_text(json.dumps({"compare16_config_sha256": "recorded-source-hash"}))
            main(["--split-dir", str(split), "--daic-root", str(root),
                  "--canonical-features", str(source), "--output", str(output),
                  "--compare16-config", str(root / "missing.conf"),
                  "--smile-extract", str(root / "missing-binary")])
            audit = json.loads((output / "provenance.json").read_text())
            self.assertEqual((audit["reuse_count"], audit["fresh_count"], audit["copied_to_independent_cache"]),
                             (188, 0, 47))
            self.assertEqual(audit["compare16_config_sha256"], "recorded-source-hash")
            result = pd.read_csv(output / "participant_manifest.csv")
            self.assertTrue(all(Path(path).is_relative_to(output) for path in result.iloc[141:].feature_path))


if __name__ == "__main__":
    unittest.main()
