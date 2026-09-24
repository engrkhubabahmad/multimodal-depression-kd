"""Remove only old 70/15/15 experiment folders after independent 80/10/10 cache verification."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .split import digest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=103)
    a = p.parse_args([str(v) for v in argv] if argv is not None else None)
    root = a.daic_root.expanduser().absolute()
    experiments = root / "experiments"
    new = experiments / f"full188_80_10_10_seed{a.seed}"
    old = [experiments / "full188_seed42", experiments / "full188_seed103"]
    if not (new / "features/provenance.json").is_file():
        raise RuntimeError("Complete the new audio feature cache before cleanup")
    marker = json.loads((new / "split/complete.json").read_text())
    if marker["signature"]["split"] != [150, 19, 19] or \
            marker["manifest_sha256"] != digest(new / "split/manifest.csv"):
        raise ValueError("New split is incomplete or modified")
    split = pd.read_csv(new / "split/manifest.csv")
    manifest = pd.read_csv(new / "features/participant_manifest.csv")
    if len(manifest) != 188 or manifest.participant_id.nunique() != 188:
        raise ValueError("New feature cache is incomplete")
    if set(manifest.participant_id) != set(split.participant_id):
        raise ValueError("New cache and split IDs differ")
    for path in map(Path, manifest.feature_path):
        resolved = path.resolve(strict=True)
        if any(resolved.is_relative_to(folder.resolve()) for folder in old):
            raise ValueError(f"New cache still depends on old experiment: {resolved}")
    removed = []
    for folder in old:
        if folder.exists():
            print("Removing previous experiment:", folder, flush=True)
            shutil.rmtree(folder)
            removed.append(str(folder))
    audit = {"utc": datetime.now(timezone.utc).isoformat(), "removed": removed,
             "retained": [str(new), str(experiments / "ussd_compare16")],
             "verified_feature_count": len(manifest)}
    (new / "previous_cleanup.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Previous experimental splits and outputs removed. Raw DAIC-WOZ and shared cache retained.")


if __name__ == "__main__":
    main()
