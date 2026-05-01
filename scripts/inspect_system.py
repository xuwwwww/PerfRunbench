from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.profiler.hardware_info import collect_hardware_info, write_hardware_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect hardware, runtime packages, and resource limits.")
    parser.add_argument("--output", default="results/reports/system_info.json")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    info = collect_hardware_info()
    print(json.dumps(info, indent=2, sort_keys=True))
    if not args.no_write:
        path = write_hardware_info(args.output, info)
        print(f"Wrote system info to {path}")


if __name__ == "__main__":
    main()

