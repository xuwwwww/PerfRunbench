from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.run_state import RUNS_DIR, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore files changed by a previous AutoTuneAI run.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")
    manifest = load_manifest(run_dir)
    changed_files = manifest.get("changed_files", [])
    if not changed_files:
        print(f"Run {args.run_id} has no changed files to restore.")
        return
    for item in changed_files:
        path = Path(item["path"])
        backup = Path(item["backup"])
        if not backup.exists():
            raise SystemExit(f"Backup missing for {path}: {backup}")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)
        print(f"Restored {path} from {backup}")


if __name__ == "__main__":
    main()

