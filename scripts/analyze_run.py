from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.run_analysis import analyze_run, format_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a PerfRunbench run's resource guard behavior.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    analysis = analyze_run(args.run_id)
    if args.json:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print(format_analysis(analysis))


if __name__ == "__main__":
    main()
