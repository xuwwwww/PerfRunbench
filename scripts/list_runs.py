from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.run_state import list_runs


def main() -> None:
    runs = list_runs()
    if not runs:
        print("No PerfRunbench runs found.")
        return
    for run in runs:
        print(
            f"{run['run_id']}  status={run.get('status')}  "
            f"return_code={run.get('return_code')}  command={' '.join(run.get('command', []))}"
        )


if __name__ == "__main__":
    main()
