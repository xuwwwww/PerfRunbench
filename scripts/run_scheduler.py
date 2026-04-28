from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.scheduler.deadline_aware import schedule_deadline_aware
from autotune.scheduler.dynamic_batching import schedule_dynamic_batching
from autotune.scheduler.fcfs import schedule_fcfs
from autotune.scheduler.simulator import summarize
from autotune.scheduler.static_batching import schedule_static_batching
from autotune.scheduler.workload_generator import generate_workload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["low", "medium", "burst"], default="medium")
    parser.add_argument(
        "--scheduler",
        choices=["fcfs", "static", "dynamic", "deadline_aware"],
        default="deadline_aware",
    )
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    requests = generate_workload(args.workload, args.count)
    if args.scheduler == "fcfs":
        completed = schedule_fcfs(requests)
    elif args.scheduler == "static":
        completed = schedule_static_batching(requests)
    elif args.scheduler == "dynamic":
        completed = schedule_dynamic_batching(requests)
    else:
        completed = schedule_deadline_aware(requests)

    print(json.dumps(summarize(completed), indent=2))


if __name__ == "__main__":
    main()
