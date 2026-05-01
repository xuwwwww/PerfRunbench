from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.budget import ResourceBudget
from autotune.training_tuner.batch_size import BatchSizeTuningError, tune_batch_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune training batch size under a resource budget.")
    parser.add_argument("--file", required=True, help="Training config file to edit reversibly.")
    parser.add_argument("--batch-size-key", default="batch_size")
    parser.add_argument("--values", nargs="+", type=int, required=True)
    parser.add_argument("--output", default="results/reports/training_tuning_summary.json")
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--reserve-memory-gb", type=float, default=0.0)
    parser.add_argument("--reserve-cores", type=int, default=0)
    parser.add_argument("--cpu-quota-percent", type=float)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--hard-kill", action="store_true")
    parser.add_argument("--executor", choices=["auto", "local", "systemd"], default="local")
    parser.add_argument("--sudo", action="store_true", help="Use sudo for systemd-run privileged scopes.")
    parser.add_argument(
        "--allow-sudo-auto",
        action="store_true",
        help="Allow --executor auto to use sudo when capability detection says systemd requires it.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Usage: python scripts/tune_training_config.py --file CONFIG --values ... -- <command>")

    budget = ResourceBudget(
        memory_budget_gb=args.memory_budget_gb,
        reserve_memory_gb=args.reserve_memory_gb,
        reserve_cores=args.reserve_cores,
        cpu_quota_percent=args.cpu_quota_percent,
        enforce=True,
    )
    try:
        result = tune_batch_size(
            args.file,
            args.batch_size_key,
            args.values,
            command,
            budget,
            args.output,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            executor=args.executor,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
        )
    except (BatchSizeTuningError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result, indent=2))
    print(f"Wrote training tuning summary to {args.output}")


if __name__ == "__main__":
    main()
