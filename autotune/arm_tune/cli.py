from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.config import default_local_target, load_target_spec
from autotune.arm_tune.errors import ArmTuneError
from autotune.arm_tune.schemas import DoctorStatuses, ProductionValidation, StatusCheck, TargetSpec
from autotune.arm_tune.targets.local import LocalArmTarget
from autotune.arm_tune.targets.ssh import SshArmTarget
from autotune.arm_tune.targets.worker import main as worker_main
from autotune.resource.executor_capabilities import collect_executor_capabilities


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except ArmTuneError as exc:
        print(f"ArmTune error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotuneai arm", description="ArmTune target inspection and validation.")
    subparsers = parser.add_subparsers(dest="arm_command", metavar="{doctor,inspect-host}")

    doctor = subparsers.add_parser("doctor", help="Validate ArmTune controller and target configuration.")
    doctor.add_argument("--target-config")
    doctor.add_argument("--require-production", action="store_true")
    doctor.set_defaults(handler=_cmd_doctor)

    inspect_host = subparsers.add_parser("inspect-host", help="Inspect an ArmTune local or SSH target host.")
    inspect_host.add_argument("--target-config")
    inspect_host.set_defaults(handler=_cmd_inspect_host)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job-spec", required=True)
    worker.add_argument("--run-dir", required=True)
    worker.set_defaults(handler=_cmd_worker)
    _hide_subparser_help(subparsers, "worker")

    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    target_configured = bool(getattr(args, "target_config", None))
    spec = _target_spec_from_args(args)
    local_validation = LocalArmTarget(default_local_target()).validate_production_target()
    configured_validation: ProductionValidation | None = None
    configured_prerequisites: dict[str, Any] | None = None
    configured_error: str | None = None
    if target_configured:
        target = _target_from_spec(spec)
        try:
            if spec.type == "ssh":
                prerequisites = target.check_prerequisites()
                configured_prerequisites = prerequisites.to_dict()
                configured_validation = prerequisites.production_validation
            else:
                configured_validation = target.validate_production_target()
        except ArmTuneError as exc:
            configured_error = str(exc)
    statuses = DoctorStatuses(
        controller_status=StatusCheck(
            status="ready",
            reason="controller diagnostics completed; controller architecture is not production benchmark evidence",
        ),
        local_target_status=_status_from_validation(local_validation),
        configured_target_status=_configured_status(
            configured=target_configured,
            validation=configured_validation,
            error=configured_error,
        ),
        production_benchmark_ready=bool(target_configured and configured_validation and configured_validation.valid),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "armtune_doctor",
        "target": spec.sanitized_dict(),
        "controller_status": statuses.controller_status.to_dict(),
        "local_target_status": statuses.local_target_status.to_dict(),
        "configured_target_status": statuses.configured_target_status.to_dict(),
        "production_benchmark_ready": statuses.production_benchmark_ready,
        "controller_diagnostics": {
            "executor_capabilities": collect_executor_capabilities(),
        },
        "configured_target_prerequisites": configured_prerequisites,
        "production_validation": configured_validation.to_dict() if configured_validation else None,
        "notes": _doctor_notes(spec, statuses),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_production and not statuses.production_benchmark_ready:
        return 2
    return 0


def _cmd_inspect_host(args: argparse.Namespace) -> int:
    spec = _target_spec_from_args(args)
    target = _target_from_spec(spec)
    result = target.inspect()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_worker(args: argparse.Namespace) -> int:
    return worker_main(["--job-spec", args.job_spec, "--run-dir", args.run_dir])


def _target_spec_from_args(args: argparse.Namespace) -> TargetSpec:
    if getattr(args, "target_config", None):
        return load_target_spec(args.target_config)
    return default_local_target()


def _target_from_spec(spec: TargetSpec):
    if spec.type == "local":
        return LocalArmTarget(spec)
    if spec.type == "ssh":
        return SshArmTarget(spec)
    raise ArmTuneError(f"unsupported target type: {spec.type}")


def _hide_subparser_help(subparsers: argparse._SubParsersAction, name: str) -> None:
    subparsers._choices_actions = [action for action in subparsers._choices_actions if action.dest != name]


def _status_from_validation(validation: ProductionValidation) -> StatusCheck:
    return StatusCheck(
        status="ready" if validation.valid else "unsupported",
        reason=validation.reason,
    )


def _configured_status(
    *,
    configured: bool,
    validation: ProductionValidation | None,
    error: str | None,
) -> StatusCheck:
    if not configured:
        return StatusCheck(status="not_configured", reason="no target config was supplied")
    if error:
        return StatusCheck(status="error", reason=error)
    if validation is None:
        return StatusCheck(status="error", reason="configured target validation did not produce a result")
    return _status_from_validation(validation)


def _doctor_notes(spec: TargetSpec, statuses: DoctorStatuses) -> list[str]:
    notes = [
        "Host capabilities and workload execution evidence are separate ArmTune schemas.",
        "No benchmark or workload execution evidence is collected by arm doctor.",
        "Controller readiness is separate from production benchmark readiness.",
    ]
    if spec.type == "ssh" and spec.strict_host_key_checking:
        notes.append("SSH strict host-key checking is enabled.")
    if not statuses.production_benchmark_ready:
        notes.append("Production benchmark execution requires a configured Linux Arm64 target.")
    return notes
