"""Colab setup must fetch the existing PR branch without echoing credentials."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "notebooks"
BRANCH = "feature/full188-fresh-teachers-student-seed42"


class NotebookSetup(unittest.TestCase):
    def test_seed_notebooks_fetch_existing_branch_safely(self):
        for seed in (42, 103):
            with self.subTest(seed=seed):
                notebook = json.loads((ROOT / f"full188_seed{seed}_start_to_finish.ipynb").read_text())
                setup = "".join(notebook["cells"][1]["source"])
                self.assertIn(f"BRANCH = '{BRANCH}'", setup)
                self.assertIn("env=auth_env", setup)
                self.assertNotIn("git = ['git', '-c'", setup)
                stage_cells = '\n'.join(''.join(cell['source']) for cell in notebook['cells']
                                        if cell['cell_type'] == 'code')
                self.assertIn(f"'--seed', '{seed}'", stage_cells)
                self.assertNotIn(f"'--seed', {seed}", stage_cells)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), str(seed), "exec")


if __name__ == "__main__":
    unittest.main()
