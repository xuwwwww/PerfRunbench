from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.budget import ResourceBudget
from autotune.resource.workload_runner import run_with_budget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a training or inference command with resource monitoring and optional budget enforcement."
    )
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
        raise SystemExit("Usage: python scripts/run_with_budget.py [budget args] -- <command>")

    budget = ResourceBudget(
        memory_budget_gb=args.memory_budget_gb,
        reserve_memory_gb=args.reserve_memory_gb,
        reserve_cores=args.reserve_cores,
        cpu_quota_percent=args.cpu_quota_percent,
        enforce=True,
    )
    try:
        return_code, run_dir = run_with_budget(
            command,
            budget,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            executor=args.executor,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Run directory: {run_dir}")
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
