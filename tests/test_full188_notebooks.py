"""The staged 80:10:10 Colab notebook references the current GitHub branch safely."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "notebooks"


class NotebookSetup(unittest.TestCase):
    def test_protocol_and_auth(self):
        notebook = json.loads((ROOT / "full188_80_10_10_seed103.ipynb").read_text())
        source = "\n".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("full188_80_10_10_seed103", source)
        self.assertIn("BRANCH = 'feature/full188-fresh-teachers-student-seed42'", source)
        self.assertIn("env=auth_env", source)
        self.assertNotIn("git = ['git', '-c'", source)
        self.assertIn("cleanup_previous", source)
        self.assertIn("'--seed', '103'", source)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "80-10-10 notebook", "exec")


if __name__ == "__main__":
    unittest.main()
