from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.resource.budget import ResourceBudget
from autotune.source_tuner.tuned_runner import SourceEdit, run_tuned_with_budget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply reversible source edits, run a command with resource monitoring, then restore edits."
    )
    parser.add_argument(
        "--edit",
        nargs=3,
        action="append",
        metavar=("FILE", "FIND", "REPLACE"),
        help="A reversible source edit. Can be provided multiple times.",
    )
    parser.add_argument(
        "--edits-file",
        help="JSON file containing edits: [{\"file\": \"train.py\", \"find\": \"...\", \"replace\": \"...\"}]",
    )
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--reserve-memory-gb", type=float, default=0.0)
    parser.add_argument("--reserve-cores", type=int, default=0)
    parser.add_argument("--cpu-quota-percent", type=float)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--hard-kill", action="store_true")
    parser.add_argument("--executor", choices=["local", "systemd"], default="local")
    parser.add_argument("--sudo", action="store_true", help="Use sudo for systemd-run privileged scopes.")
    parser.add_argument("--keep-changes", action="store_true", help="Do not auto-restore after command exits.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Usage: python scripts/run_tuned_with_budget.py --edit FILE FIND REPLACE -- <command>")
    edits = _load_edits(args.edit, args.edits_file)
    if not edits:
        raise SystemExit("At least one --edit FILE FIND REPLACE or --edits-file is required.")

    budget = ResourceBudget(
        memory_budget_gb=args.memory_budget_gb,
        reserve_memory_gb=args.reserve_memory_gb,
        reserve_cores=args.reserve_cores,
        cpu_quota_percent=args.cpu_quota_percent,
        enforce=True,
    )
    return_code, run_dir = run_tuned_with_budget(
        command,
        edits,
        budget,
        sample_interval_seconds=args.sample_interval_seconds,
        hard_kill=args.hard_kill,
        auto_restore=not args.keep_changes,
        executor=args.executor,
        use_sudo=args.sudo,
    )
    print(f"Run directory: {run_dir}")
    raise SystemExit(return_code)


def _load_edits(raw_edits: list[list[str]] | None, edits_file: str | None) -> list[SourceEdit]:
    edits: list[SourceEdit] = []
    if raw_edits:
        edits.extend(SourceEdit(file=item[0], find_text=item[1], replace_text=item[2]) for item in raw_edits)
    if edits_file:
        payload = json.loads(Path(edits_file).read_text(encoding="utf-8"))
        edits.extend(
            SourceEdit(file=item["file"], find_text=item["find"], replace_text=item["replace"]) for item in payload
        )
    return edits


if __name__ == "__main__":
    main()
