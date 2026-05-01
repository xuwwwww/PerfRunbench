from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotune.profiler.hardware_info import collect_hardware_info, write_hardware_info
from autotune.report.run_report import generate_run_report
from autotune.resource.budget import ResourceBudget
from autotune.resource.executor_capabilities import collect_executor_capabilities
from autotune.resource.run_analysis import analyze_run, format_analysis
from autotune.resource.run_state import RUNS_DIR, list_runs, load_manifest
from autotune.resource.workload_runner import run_with_budget
from autotune.source_tuner.transaction import SourceTuningError, restore_changed_files
from autotune.system_tuner.runtime import (
    SystemTuningError,
    apply_system_tuning,
    available_profiles,
    recommend_system_tuning,
    restore_system_tuning,
)
from autotune.training_tuner.batch_size import BatchSizeTuningError, tune_batch_size


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotuneai", description="AutoTuneAI resource and tuning toolkit.")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect hardware, runtime, limits, and recommendations.")
    inspect_parser.add_argument("--output", default="results/reports/system_info.json")
    inspect_parser.add_argument("--no-write", action="store_true")
    inspect_parser.set_defaults(handler=_cmd_inspect)

    executors = subparsers.add_parser("executors", help="Inspect resource executor capabilities.")
    executors.add_argument("--probe-docker", action="store_true")
    executors.add_argument("--probe-systemd", action="store_true")
    executors.add_argument("--check-sudo-cache", action="store_true")
    executors.set_defaults(handler=_cmd_executors)

    run = subparsers.add_parser("run", help="Run a command with resource monitoring and optional limits.")
    _add_budget_args(run)
    run.add_argument("workload", nargs=argparse.REMAINDER)
    run.add_argument("--tune-system", choices=available_profiles(), help="Apply a runtime system tuning profile before running.")
    run.add_argument(
        "--no-restore-system-after",
        action="store_true",
        help="Keep runtime system tuning after the workload instead of restoring the before snapshot.",
    )
    run.add_argument("--system-tuning-sudo", action="store_true", help="Use sudo for runtime sysctl tuning writes.")
    run.set_defaults(handler=_cmd_run)

    analyze = subparsers.add_parser("analyze", help="Analyze a run's resource guard behavior.")
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(handler=_cmd_analyze)

    report = subparsers.add_parser("report", help="Generate a Markdown report for a run.")
    report.add_argument("--run-id", required=True)
    report.add_argument("--output")
    report.set_defaults(handler=_cmd_report)

    tune_system = subparsers.add_parser("tune-system", help="Recommend or apply reversible runtime system tuning.")
    tune_system.add_argument("--profile", default="linux-training-safe", choices=available_profiles())
    tune_system.add_argument("--apply", action="store_true")
    tune_system.add_argument("--sudo", action="store_true")
    tune_system.set_defaults(handler=_cmd_tune_system)

    tune_batch = subparsers.add_parser("tune-batch", help="Tune a numeric batch-size style config key.")
    tune_batch.add_argument("--file", required=True)
    tune_batch.add_argument("--batch-size-key", default="batch_size")
    tune_batch.add_argument("--values", nargs="+", type=int, required=True)
    tune_batch.add_argument("--output", default="results/reports/training_tuning_summary.json")
    _add_budget_args(tune_batch)
    tune_batch.add_argument("workload", nargs=argparse.REMAINDER)
    tune_batch.set_defaults(handler=_cmd_tune_batch)

    list_parser = subparsers.add_parser("list-runs", help="List AutoTuneAI runs.")
    list_parser.set_defaults(handler=_cmd_list_runs)

    restore = subparsers.add_parser("restore", help="Restore files and runtime system settings changed by a run.")
    restore.add_argument("--run-id", required=True)
    restore.add_argument("--sudo", action="store_true")
    restore.set_defaults(handler=_cmd_restore)
    return parser


def _add_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--reserve-memory-gb", type=float, default=0.0)
    parser.add_argument("--reserve-cores", type=int, default=0)
    parser.add_argument("--cpu-quota-percent", type=float)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--hard-kill", action="store_true")
    parser.add_argument("--executor", choices=["auto", "local", "systemd"], default="local")
    parser.add_argument("--sudo", action="store_true", help="Use sudo for privileged executor operations.")
    parser.add_argument(
        "--allow-sudo-auto",
        action="store_true",
        help="Allow --executor auto to use sudo when capability detection says systemd requires it.",
    )


def _cmd_inspect(args: argparse.Namespace) -> int:
    info = collect_hardware_info()
    print(json.dumps(info, indent=2, sort_keys=True))
    if not args.no_write:
        path = write_hardware_info(args.output, info)
        print(f"Wrote system info to {path}")
    return 0


def _cmd_executors(args: argparse.Namespace) -> int:
    capabilities = collect_executor_capabilities(
        probe_docker=args.probe_docker,
        probe_systemd=args.probe_systemd,
        check_sudo_cache=args.check_sudo_cache,
    )
    print(json.dumps(capabilities, indent=2, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    command = _command_after_separator(args.workload, "Usage: autotuneai run [budget args] -- <command>")
    budget = _budget_from_args(args)
    try:
        return_code, run_dir = run_with_budget(
            command,
            budget,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            executor=args.executor,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
            tune_system_profile=args.tune_system,
            restore_system_after=not args.no_restore_system_after,
            system_tuning_sudo=args.system_tuning_sudo,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Run directory: {run_dir}")
    return return_code


def _cmd_analyze(args: argparse.Namespace) -> int:
    analysis = analyze_run(args.run_id)
    if args.json:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print(format_analysis(analysis))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    try:
        report_path = generate_run_report(args.run_id, args.output)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote run report to {report_path}")
    return 0


def _cmd_tune_system(args: argparse.Namespace) -> int:
    try:
        if not args.apply:
            print(json.dumps(recommend_system_tuning(args.profile), indent=2, sort_keys=True))
            return 0
        run_dir, result = apply_system_tuning(args.profile, use_sudo=args.sudo)
    except SystemTuningError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Run directory: {run_dir}")
    print(f"Before snapshot: {run_dir / 'system_tuning_before.json'}")
    print(f"After snapshot: {run_dir / 'system_tuning_after.json'}")
    print(f"Diff: {run_dir / 'system_tuning_diff.json'}")
    return 0


def _cmd_tune_batch(args: argparse.Namespace) -> int:
    command = _command_after_separator(args.workload, "Usage: autotuneai tune-batch --file CONFIG --values ... -- <command>")
    try:
        result = tune_batch_size(
            args.file,
            args.batch_size_key,
            args.values,
            command,
            _budget_from_args(args),
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
    return 0


def _cmd_list_runs(args: argparse.Namespace) -> int:
    runs = list_runs()
    if not runs:
        print("No AutoTuneAI runs found.")
        return 0
    for run in runs:
        print(
            f"{run['run_id']}  status={run.get('status')}  "
            f"return_code={run.get('return_code')}  command={' '.join(run.get('command', []))}"
        )
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    run_dir = RUNS_DIR / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run not found: {run_dir}")
    manifest = load_manifest(run_dir)
    restored_any = False
    changed_files = manifest.get("changed_files", [])
    if changed_files:
        try:
            restored = restore_changed_files(run_dir)
        except SourceTuningError as exc:
            raise SystemExit(str(exc)) from exc
        for path in restored:
            print(f"Restored {path}")
        restored_any = True
    for item in restore_system_tuning(run_dir, use_sudo=args.sudo):
        if item["return_code"] == 0:
            print(f"Restored system setting {item['key']}={item['restored_value']}")
        else:
            print(f"Failed to restore system setting {item['key']}: {item['error']}")
        restored_any = True
    if not restored_any:
        print(f"Run {args.run_id} has no changed files or system settings to restore.")
    return 0


def _budget_from_args(args: argparse.Namespace) -> ResourceBudget:
    return ResourceBudget(
        memory_budget_gb=args.memory_budget_gb,
        reserve_memory_gb=args.reserve_memory_gb,
        reserve_cores=args.reserve_cores,
        cpu_quota_percent=args.cpu_quota_percent,
        enforce=True,
    )


def _command_after_separator(command: list[str], usage: str) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit(usage)
    return command


if __name__ == "__main__":
    raise SystemExit(main())
