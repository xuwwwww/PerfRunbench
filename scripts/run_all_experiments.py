from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = [
        [sys.executable, "scripts/run_benchmark.py", "--config", "configs/resnet18.yaml"],
        [sys.executable, "scripts/run_autotune.py", "--config", "configs/resnet18.yaml", "--search", "cost_model"],
        [sys.executable, "scripts/run_scheduler.py", "--workload", "burst", "--scheduler", "deadline_aware"],
    ]
    for command in commands:
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
