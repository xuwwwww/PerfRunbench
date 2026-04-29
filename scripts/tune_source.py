from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.source_tuner.transaction import SourceTuningError, apply_find_replace


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely apply a reversible source-code find/replace.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--find", required=True)
    parser.add_argument("--replace", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--apply", action="store_true", help="Apply the edit. Without this flag, only previews.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = apply_find_replace(
            args.file,
            args.find,
            args.replace,
            run_id=args.run_id,
            apply=args.apply,
        )
    except SourceTuningError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(result, indent=2))
        return
    action = "Applied" if result["applied"] else "Dry run"
    print(f"{action}: {result['target_file']}")
    print(f"matches: {result['matches']}")
    if result.get("run_id"):
        print(f"run_id: {result['run_id']}")
    if result.get("backup"):
        print(f"backup: {result['backup']}")


if __name__ == "__main__":
    main()

