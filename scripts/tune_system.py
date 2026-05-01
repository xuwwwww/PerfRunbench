from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.system_tuner.runtime import (
    SystemTuningError,
    apply_system_tuning,
    available_profiles,
    recommend_system_tuning,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend or apply reversible runtime system tuning.")
    parser.add_argument("--profile", default="linux-training-safe", choices=available_profiles())
    parser.add_argument("--apply", action="store_true", help="Apply the runtime system tuning profile.")
    parser.add_argument("--sudo", action="store_true", help="Use sudo for sysctl writes.")
    args = parser.parse_args()

    try:
        if not args.apply:
            print(json.dumps(recommend_system_tuning(args.profile), indent=2, sort_keys=True))
            return
        run_dir, result = apply_system_tuning(args.profile, use_sudo=args.sudo)
    except SystemTuningError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Run directory: {run_dir}")
    print(f"Before snapshot: {run_dir / 'system_tuning_before.json'}")
    print(f"After snapshot: {run_dir / 'system_tuning_after.json'}")
    print(f"Diff: {run_dir / 'system_tuning_diff.json'}")


if __name__ == "__main__":
    main()
