from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from . import AUTHOR_COMMIT, AUTHOR_REPO, CHECKPOINT_NAME, RUN4_REL
from .common import save_json, sha256


def run(cmd): subprocess.run(cmd, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--root", default="/content/solo_teacher_sources/USSD-depression"); p.add_argument("--manifest")
    a = p.parse_args(argv); root = Path(a.root)
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True); run(["git", "clone", AUTHOR_REPO, str(root)])
    run(["git", "-C", str(root), "fetch", "origin", AUTHOR_COMMIT]); run(["git", "-C", str(root), "checkout", "--detach", AUTHOR_COMMIT])
    resolved = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if resolved != AUTHOR_COMMIT: raise AssertionError(f"Expected {AUTHOR_COMMIT}, got {resolved}")
    run4 = root / RUN4_REL; ckpt = run4 / CHECKPOINT_NAME; stats = run4 / "data_saver.pickle"
    for path in (ckpt, stats):
        if not path.exists(): raise FileNotFoundError(path)
    info = {"author_repo": AUTHOR_REPO, "author_commit": AUTHOR_COMMIT, "run": 4, "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt), "data_saver": str(stats), "data_saver_sha256": sha256(stats), "test_opened": False}
    if a.manifest: save_json(Path(a.manifest), info)
    print(json.dumps(info, indent=2)); return info


if __name__ == "__main__": main()
