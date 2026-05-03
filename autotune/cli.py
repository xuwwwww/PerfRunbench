from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotune.profiler.hardware_info import collect_hardware_info, write_hardware_info
from autotune.gpu.nvidia_tuner import (
    NvidiaTuningError,
    apply_nvidia_tuning,
    available_nvidia_profiles,
    recommend_nvidia_tuning,
    restore_nvidia_tuning,
)
from autotune.report.run_report import generate_run_report
from autotune.resource.budget import ResourceBudget
from autotune.resource.comparison_runner import compare_tuning
from autotune.resource.executor_capabilities import collect_executor_capabilities
from autotune.resource.memory_calibration import calibrate_memory
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
from autotune.system_tuner.profile_selector import select_system_profile
from autotune.training_tuner.batch_size import BatchSizeTuningError, tune_batch_size
from autotune.training_tuner.multi_knob import parse_knob_specs, tune_training_knobs

DEMO_WORKLOAD = ["python", "examples/dummy_train.py"]
DEMO_CONFIG = "examples/train_config.yaml"
WORKLOAD_PROFILE_CHOICES = ["auto", "training", "memory", "throughput", "low-latency", "cpu-conservative"]


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
        "--auto-tune-system",
        action="store_true",
        help="Automatically apply the recommended runtime system tuning profile when supported.",
    )
    run.add_argument(
        "--no-restore-system-after",
        action="store_true",
        help="Keep runtime system tuning after the workload instead of restoring the before snapshot.",
    )
    run.add_argument("--system-tuning-sudo", action="store_true", help="Use sudo for runtime sysctl tuning writes.")
    run.add_argument("--workload-profile", choices=WORKLOAD_PROFILE_CHOICES, default="auto")
    run.add_argument("--tune-gpu", choices=available_nvidia_profiles(), help="Apply an NVIDIA runtime tuning profile first.")
    run.add_argument("--auto-tune-gpu", action="store_true", help="Apply the default NVIDIA throughput runtime profile when nvidia-smi is available.")
    run.add_argument("--gpu-tuning-sudo", action="store_true", help="Use sudo for NVIDIA runtime tuning writes.")
    run.add_argument("--no-restore-gpu-after", action="store_true")
    run.set_defaults(handler=_cmd_run)

    analyze = subparsers.add_parser("analyze", help="Analyze a run's resource guard behavior.")
    analyze.add_argument("--run-id", required=True)
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(handler=_cmd_analyze)

    report = subparsers.add_parser("report", help="Generate a Markdown report for a run.")
    report.add_argument("--run-id", required=True)
    report.add_argument("--output")
    report.set_defaults(handler=_cmd_report)

    calibrate = subparsers.add_parser("calibrate-memory", help="Measure memory budget behavior on this machine.")
    calibrate.add_argument("--budget-gb", nargs="+", type=float, required=True)
    calibrate.add_argument("--workload-memory-mb", type=int, default=1024)
    calibrate.add_argument("--duration-seconds", type=float, default=5.0)
    calibrate.add_argument("--workers", type=int, default=2)
    calibrate.add_argument("--output", default="results/reports/memory_calibration.json")
    _add_budget_executor_args(calibrate)
    calibrate.add_argument("--sample-interval-seconds", type=float, default=0.1)
    calibrate.add_argument("--hard-kill", action="store_true")
    calibrate.set_defaults(handler=_cmd_calibrate_memory)

    compare = subparsers.add_parser("compare-tuning", help="Run baseline and tuned workloads and compare metrics.")
    _add_budget_args(compare)
    compare.add_argument("--profile", choices=available_profiles(), default=None)
    compare.add_argument("--workload-profile", choices=WORKLOAD_PROFILE_CHOICES, default="auto")
    compare.add_argument("--system-tuning-sudo", action="store_true")
    compare.add_argument("--repeat", type=int, default=1, help="Run baseline/tuned pairs multiple times and report medians.")
    compare.add_argument("--output", default="results/reports/tuning_comparison.json")
    compare.add_argument("workload", nargs=argparse.REMAINDER)
    compare.set_defaults(handler=_cmd_compare_tuning)

    compare_runs = subparsers.add_parser("compare-runs", help="Compare two existing runs by run id.")
    compare_runs.add_argument("--baseline-run-id", required=True)
    compare_runs.add_argument("--tuned-run-id", required=True)
    compare_runs.add_argument("--profile", default="manual")
    compare_runs.add_argument("--output", default="results/reports/run_comparison.json")
    compare_runs.set_defaults(handler=_cmd_compare_runs)

    demo = subparsers.add_parser("demo", help="Run built-in repo demo workflows against the dummy workload.")
    _add_budget_args(demo)
    demo.add_argument(
        "--scenario",
        choices=["run", "tune-batch", "compare-tuning", "all"],
        default="run",
    )
    demo.add_argument("--workload-profile", choices=WORKLOAD_PROFILE_CHOICES, default="auto")
    demo.add_argument("--system-tuning-sudo", action="store_true")
    demo.add_argument("--batch-values", nargs="+", type=int, default=[128, 64, 32])
    demo.add_argument("--output-dir", default="results/reports")
    demo.set_defaults(handler=_cmd_demo)

    tune_system = subparsers.add_parser("tune-system", help="Recommend or apply reversible runtime system tuning.")
    tune_system.add_argument("--profile", default=None, choices=available_profiles())
    tune_system.add_argument("--apply", action="store_true")
    tune_system.add_argument("--sudo", action="store_true")
    tune_system.set_defaults(handler=_cmd_tune_system)

    tune_gpu = subparsers.add_parser("tune-gpu", help="Recommend or apply reversible NVIDIA runtime tuning.")
    tune_gpu.add_argument("--profile", default="nvidia-throughput", choices=available_nvidia_profiles())
    tune_gpu.add_argument("--apply", action="store_true")
    tune_gpu.add_argument("--sudo", action="store_true")
    tune_gpu.set_defaults(handler=_cmd_tune_gpu)

    tune_batch = subparsers.add_parser("tune-batch", help="Tune a numeric batch-size style config key.")
    tune_batch.add_argument("--file", required=True)
    tune_batch.add_argument("--key", default=None, help="Numeric config key to tune, for example batch_size or num_workers.")
    tune_batch.add_argument("--batch-size-key", default=None, help="Backward-compatible alias for --key.")
    tune_batch.add_argument("--values", nargs="+", type=int, required=True)
    tune_batch.add_argument("--output", default="results/reports/training_tuning_summary.json")
    _add_budget_args(tune_batch)
    tune_batch.add_argument("workload", nargs=argparse.REMAINDER)
    tune_batch.set_defaults(handler=_cmd_tune_batch)

    tune_training = subparsers.add_parser("tune-training", help="Tune multiple numeric training knobs sequentially.")
    tune_training.add_argument("--file", required=True)
    tune_training.add_argument("--knob", action="append", required=True, help="Key and candidate values, for example batch_size=128,64,32")
    tune_training.add_argument("--objective", choices=["throughput", "duration", "memory"], default="throughput")
    tune_training.add_argument("--min-final-accuracy", type=float, default=None)
    tune_training.add_argument("--output", default="results/reports/training_plan_summary.json")
    _add_budget_args(tune_training)
    tune_training.add_argument("workload", nargs=argparse.REMAINDER)
    tune_training.set_defaults(handler=_cmd_tune_training)

    list_parser = subparsers.add_parser("list-runs", help="List AutoTuneAI runs.")
    list_parser.set_defaults(handler=_cmd_list_runs)

    restore = subparsers.add_parser("restore", help="Restore files and runtime system settings changed by a run.")
    restore.add_argument("--run-id", required=True)
    restore.add_argument("--sudo", action="store_true")
    restore.add_argument("--gpu-sudo", action="store_true")
    restore.set_defaults(handler=_cmd_restore)
    return parser


def _add_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--reserve-memory-gb", type=float, default=0.0)
    parser.add_argument("--reserve-cores", type=int, default=0)
    parser.add_argument("--cpu-quota-percent", type=float)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--hard-kill", action="store_true")
    _add_budget_executor_args(parser)


def _add_budget_executor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--executor", choices=["auto", "local", "systemd", "docker"], default="local")
    parser.add_argument("--sudo", action="store_true", help="Use sudo for privileged executor operations.")
    parser.add_argument(
        "--allow-sudo-auto",
        action="store_true",
        help="Allow --executor auto to use sudo when capability detection says systemd requires it.",
    )
    parser.add_argument(
        "--docker-image",
        default="python:3.12-slim",
        help="Docker image to use when --executor docker is selected.",
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
    tune_system_profile = _resolve_system_tuning_profile(args)
    tune_gpu_profile = _resolve_gpu_tuning_profile(args)
    try:
        return_code, run_dir = run_with_budget(
            command,
            budget,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            executor=args.executor,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
            tune_system_profile=tune_system_profile,
            restore_system_after=not args.no_restore_system_after,
            system_tuning_sudo=args.system_tuning_sudo,
            docker_image=args.docker_image,
            tune_gpu_profile=tune_gpu_profile,
            restore_gpu_after=not args.no_restore_gpu_after,
            gpu_tuning_sudo=args.gpu_tuning_sudo,
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


def _cmd_calibrate_memory(args: argparse.Namespace) -> int:
    try:
        result = calibrate_memory(
            args.budget_gb,
            workload_memory_mb=args.workload_memory_mb,
            duration_seconds=args.duration_seconds,
            workers=args.workers,
            output=args.output,
            executor=args.executor,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
            docker_image=args.docker_image,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote memory calibration to {args.output}")
    return 0


def _cmd_compare_tuning(args: argparse.Namespace) -> int:
    command = _command_after_separator(args.workload, "Usage: autotuneai compare-tuning [budget args] -- <command>")
    budget = _budget_from_args(args)
    profile = args.profile or select_system_profile(budget, workload_profile=args.workload_profile).profile
    try:
        result = compare_tuning(
            command,
            budget,
            tuned_profile=profile,
            output=args.output,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            executor=args.executor,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
            system_tuning_sudo=args.system_tuning_sudo,
            docker_image=args.docker_image,
            repeat=args.repeat,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote tuning comparison to {args.output}")
    return 0


def _cmd_compare_runs(args: argparse.Namespace) -> int:
    from autotune.resource.comparison_runner import build_comparison_result

    result = build_comparison_result(
        args.baseline_run_id,
        args.tuned_run_id,
        tuned_profile=args.profile,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote run comparison to {args.output}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    budget = _budget_from_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = ["run", "tune-batch", "compare-tuning"] if args.scenario == "all" else [args.scenario]
    results: dict[str, object] = {}

    if "run" in scenarios:
        try:
            return_code, run_dir = run_with_budget(
                DEMO_WORKLOAD,
                budget,
                sample_interval_seconds=args.sample_interval_seconds,
                hard_kill=args.hard_kill,
                executor=args.executor,
                use_sudo=args.sudo,
                allow_sudo_auto=args.allow_sudo_auto,
                docker_image=args.docker_image,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        results["run"] = {"return_code": return_code, "run_dir": str(run_dir)}

    if "tune-batch" in scenarios:
        try:
            summary = tune_batch_size(
                DEMO_CONFIG,
                "batch_size",
                args.batch_values,
                DEMO_WORKLOAD,
                budget,
                output_dir / "demo_tune_batch_summary.json",
                sample_interval_seconds=args.sample_interval_seconds,
                hard_kill=args.hard_kill,
                executor=args.executor,
                use_sudo=args.sudo,
                allow_sudo_auto=args.allow_sudo_auto,
                docker_image=args.docker_image,
            )
        except (BatchSizeTuningError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
        results["tune_batch"] = {
            "recommended_value": summary.get("recommended_value"),
            "output": str(output_dir / "demo_tune_batch_summary.json"),
        }

    if "compare-tuning" in scenarios:
        profile = select_system_profile(budget, workload_profile=args.workload_profile).profile
        try:
            comparison = compare_tuning(
                DEMO_WORKLOAD,
                budget,
                tuned_profile=profile,
                output=output_dir / "demo_tuning_comparison.json",
                sample_interval_seconds=args.sample_interval_seconds,
                hard_kill=args.hard_kill,
                executor=args.executor,
                use_sudo=args.sudo,
                allow_sudo_auto=args.allow_sudo_auto,
                system_tuning_sudo=args.system_tuning_sudo,
                docker_image=args.docker_image,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        results["compare_tuning"] = {
            "tuned_profile": comparison.get("tuned_profile"),
            "output": str(output_dir / "demo_tuning_comparison.json"),
        }

    print(json.dumps({"kind": "demo", "scenario": args.scenario, "results": results}, indent=2, sort_keys=True))
    return 0


def _cmd_tune_system(args: argparse.Namespace) -> int:
    profile = args.profile or select_system_profile(ResourceBudget()).profile
    try:
        if not args.apply:
            print(json.dumps(recommend_system_tuning(profile), indent=2, sort_keys=True))
            return 0
        run_dir, result = apply_system_tuning(profile, use_sudo=args.sudo)
    except SystemTuningError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Run directory: {run_dir}")
    print(f"Before snapshot: {run_dir / 'system_tuning_before.json'}")
    print(f"After snapshot: {run_dir / 'system_tuning_after.json'}")
    print(f"Diff: {run_dir / 'system_tuning_diff.json'}")
    return 0


def _cmd_tune_gpu(args: argparse.Namespace) -> int:
    try:
        if not args.apply:
            print(json.dumps(recommend_nvidia_tuning(args.profile), indent=2, sort_keys=True))
            return 0
        run_dir, result = apply_nvidia_tuning(args.profile, use_sudo=args.sudo)
    except NvidiaTuningError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Run directory: {run_dir}")
    print(f"Before snapshot: {run_dir / 'gpu_tuning_before.json'}")
    print(f"After snapshot: {run_dir / 'gpu_tuning_after.json'}")
    print(f"Diff: {run_dir / 'gpu_tuning_diff.json'}")
    return 0


def _cmd_tune_batch(args: argparse.Namespace) -> int:
    command = _command_after_separator(args.workload, "Usage: autotuneai tune-batch --file CONFIG --values ... -- <command>")
    key = args.key or args.batch_size_key or "batch_size"
    try:
        result = tune_batch_size(
            args.file,
            key,
            args.values,
            command,
            _budget_from_args(args),
            args.output,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            executor=args.executor,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
            docker_image=args.docker_image,
        )
    except (BatchSizeTuningError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))
    print(f"Wrote training tuning summary to {args.output}")
    return 0


def _cmd_tune_training(args: argparse.Namespace) -> int:
    command = _command_after_separator(args.workload, "Usage: autotuneai tune-training --file CONFIG --knob key=v1,v2 -- <command>")
    try:
        result = tune_training_knobs(
            args.file,
            parse_knob_specs(args.knob),
            command,
            _budget_from_args(args),
            args.output,
            objective=args.objective,
            min_final_accuracy=args.min_final_accuracy,
            sample_interval_seconds=args.sample_interval_seconds,
            hard_kill=args.hard_kill,
            executor=args.executor,
            use_sudo=args.sudo,
            allow_sudo_auto=args.allow_sudo_auto,
            docker_image=args.docker_image,
        )
    except (BatchSizeTuningError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2))
    print(f"Wrote training plan summary to {args.output}")
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
    for item in restore_nvidia_tuning(run_dir, use_sudo=args.gpu_sudo):
        if item["return_code"] == 0:
            print(f"Restored NVIDIA setting with command: {' '.join(item['command'])}")
        else:
            print(f"Failed to restore NVIDIA setting: {item['error']}")
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


def _resolve_system_tuning_profile(args: argparse.Namespace) -> str | None:
    if args.tune_system and args.auto_tune_system:
        raise SystemExit("--tune-system and --auto-tune-system cannot be used together")
    if args.tune_system:
        return args.tune_system
    if not args.auto_tune_system:
        return None
    return select_system_profile(_budget_from_args(args), workload_profile=args.workload_profile).profile


def _resolve_gpu_tuning_profile(args: argparse.Namespace) -> str | None:
    if args.tune_gpu and args.auto_tune_gpu:
        raise SystemExit("--tune-gpu and --auto-tune-gpu cannot be used together")
    if args.tune_gpu:
        return args.tune_gpu
    if args.auto_tune_gpu:
        if not recommend_nvidia_tuning("nvidia-throughput").get("supported"):
            return None
        return "nvidia-throughput"
    return None


def _command_after_separator(command: list[str], usage: str) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit(usage)
    return command


if __name__ == "__main__":
    raise SystemExit(main())
