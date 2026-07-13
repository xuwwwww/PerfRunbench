from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autotune.arm_tune.errors import ArmTuneError, ArmTuneProtocolError
from autotune.arm_tune.platform.environment import collect_host_inspection
from autotune.arm_tune.schemas import RemotePrerequisiteResult, WorkerResult
from autotune.arm_tune.targets.protocol import atomic_write_json, read_json, validate_worker_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autotuneai arm worker", description="Internal ArmTune target worker.")
    parser.add_argument("--job-spec", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_worker(args.job_spec, args.run_dir)
    except ArmTuneError as exc:
        payload = WorkerResult(operation="unknown", status="error", error=str(exc)).to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "ok" else 1


def run_worker(job_spec_path: str | Path, run_dir: str | Path) -> WorkerResult:
    job = read_json(job_spec_path)
    validate_worker_job(job)
    operation = str(job["operation"])
    target_dir = Path(run_dir)
    _ensure_writable_run_dir(target_dir)
    if operation == "check_prerequisites":
        inspection = collect_host_inspection(target_type=str(job.get("target_type", "ssh")))
        result = RemotePrerequisiteResult(
            python_executable=sys.executable,
            python_available=True,
            module=str(job.get("remote_module", "autotune.cli")),
            module_importable=True,
            worker_protocol_compatible=True,
            remote_workdir_writable=True,
            linux=inspection.host.architecture.is_linux,
            arm64=inspection.host.architecture.is_arm64,
            host=inspection.host,
            production_validation=inspection.production_validation,
        )
        payload = result.to_dict()
        atomic_write_json(target_dir / "prerequisite_result.json", payload)
        return WorkerResult(operation=operation, status="ok", result=payload)
    if operation == "inspect_host":
        inspection = collect_host_inspection(target_type=str(job.get("target_type", "local")))
        payload = inspection.to_dict()
        atomic_write_json(target_dir / "inspect_host_result.json", payload)
        return WorkerResult(operation=operation, status="ok", result=payload)
    raise ArmTuneProtocolError(f"unsupported worker operation: {operation}")


def _ensure_writable_run_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe.json"
        atomic_write_json(probe, {"schema_version": "1.0", "kind": "armtune_write_probe"})
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise ArmTuneProtocolError(f"remote run directory is not writable: {path}") from exc
