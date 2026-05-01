from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.budget import ResourceBudget
from autotune.resource.systemd_executor import preflight_systemd_executor


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight-check systemd/root executor support before running a workload.")
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--cpu-quota-percent", type=float)
    parser.add_argument("--sudo", action="store_true", help="Check sudo-based systemd scope mode.")
    parser.add_argument(
        "--check-sudo-cache",
        action="store_true",
        help="Also check whether sudo is already authenticated for non-interactive use.",
    )
    parser.add_argument("--probe", action="store_true", help="Run a harmless systemd scope probe with the requested limits.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = ["true"]

    budget = ResourceBudget(
        memory_budget_gb=args.memory_budget_gb,
        cpu_quota_percent=args.cpu_quota_percent,
        enforce=True,
    )
    preflight = preflight_systemd_executor(
        command,
        budget,
        use_sudo=args.sudo,
        check_sudo_cache=args.check_sudo_cache,
        probe=args.probe,
    )
    record = preflight.to_record()

    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print(f"runnable: {record['runnable']}")
        print(f"systemd-run: {record['systemd_run_path']}")
        print(f"systemd state: {record['systemd_state']}")
        print(f"sudo: {record['sudo_path']}")
        print(f"sudo cached: {record['sudo_cached']}")
        print(f"probe succeeded: {record['probe_succeeded']}")
        if record["command_preview"]:
            print("command preview:")
            print(" ".join(record["command_preview"]))
        if record["probe_output"]:
            print("probe output:")
            print(record["probe_output"])
        for note in record["notes"]:
            print(f"note: {note}")
        for warning in record["warnings"]:
            print(f"warning: {warning}")
        for error in record["errors"]:
            print(f"error: {error}")

    raise SystemExit(0 if preflight.runnable else 1)


if __name__ == "__main__":
    main()
