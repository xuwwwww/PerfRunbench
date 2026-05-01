from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.run_state import RUNS_DIR, load_manifest
from autotune.source_tuner.transaction import SourceTuningError, restore_changed_files
from autotune.system_tuner.runtime import restore_system_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore files and runtime system settings changed by an AutoTuneAI run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sudo", action="store_true", help="Use sudo when restoring runtime system settings.")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")
    manifest = load_manifest(run_dir)
    restored_any = False
    changed_files = manifest.get("changed_files", [])
    if changed_files:
        try:
            restored = restore_changed_files(run_dir)
        except SourceTuningError as exc:
            raise SystemExit(str(exc)) from exc
        for path in restored:
            print(f"Restored {path}")
        restored_any = True

    system_restored = restore_system_tuning(run_dir, use_sudo=args.sudo)
    for item in system_restored:
        if item["return_code"] == 0:
            print(f"Restored system setting {item['key']}={item['restored_value']}")
        else:
            print(f"Failed to restore system setting {item['key']}: {item['error']}")
        restored_any = True

    if not restored_any:
        print(f"Run {args.run_id} has no changed files or system settings to restore.")


if __name__ == "__main__":
    main()
