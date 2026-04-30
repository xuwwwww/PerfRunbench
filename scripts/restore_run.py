from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.run_state import RUNS_DIR, load_manifest
from autotune.source_tuner.transaction import SourceTuningError, restore_changed_files


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
    try:
        restored = restore_changed_files(run_dir)
    except SourceTuningError as exc:
        raise SystemExit(str(exc)) from exc
    for path in restored:
        print(f"Restored {path}")


if __name__ == "__main__":
    main()
