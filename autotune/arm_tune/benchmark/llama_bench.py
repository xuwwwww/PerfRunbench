from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest, write_text_artifact
from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.benchmark.parser import parse_llama_bench_streams, required_metric_values
from autotune.arm_tune.benchmark.provenance import decide_benchmark_provenance
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.platform.environment import collect_host_inspection
from autotune.arm_tune.schemas import (
    ArmBenchmarkResult,
    BenchmarkArtifactManifest,
    BenchmarkFileManifest,
    BenchmarkTrialResult,
    LlamaBenchInspection,
    LlamaBenchInvocation,
    LlamaBenchJobSpec,
    LlamaBenchParseResult,
    utc_now,
)
from autotune.arm_tune.targets.protocol import atomic_write_json


BENCHMARK_OPERATION = "benchmark.llama_bench"
PROTOCOL_VERSION = "1.0"
REQUIRED_FLAGS = {
    "model": ["-m", "--model"],
    "threads": ["-t", "--threads"],
    "prompt_tokens": ["-p", "--n-prompt"],
    "generation_tokens": ["-n", "--n-gen"],
    "repetitions": ["-r", "--repetitions"],
}
OPTIONAL_FLAGS = {
    "threads_batch": ["--threads-batch"],
    "batch_size": ["--batch-size"],
    "ubatch_size": ["--ubatch-size"],
}


def run_llama_bench_worker(job: LlamaBenchJobSpec, run_dir: str | Path) -> ArmBenchmarkResult:
    benchmark_dir = Path(run_dir)
    raw_dir = benchmark_dir / "raw"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(benchmark_dir / "benchmark_job.json", job.to_dict())
    host_inspection = collect_host_inspection(target_type=job.target_type)
    provenance = decide_benchmark_provenance(job, host_inspection)
    if provenance.result_origin != "test_fixture" and not host_inspection.production_validation.valid:
        artifacts = _collect_artifacts(benchmark_dir, raw_dir)
        return _write_result(
            benchmark_dir,
            ArmBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                execution_status="not_run",
                parse_status="no_metrics",
                benchmark_status="failed",
                host=host_inspection.host,
                artifacts=BenchmarkArtifactManifest(artifacts=artifacts),
                provenance=provenance,
                artifact_directory=str(benchmark_dir),
                message=host_inspection.production_validation.reason,
            ),
        )
    validation = _validate_files(job)
    executable = validation.get("executable")
    model = validation.get("model")
    _write_file_manifests(benchmark_dir, executable, model)
    provenance = decide_benchmark_provenance(job, host_inspection, executable=executable)
    if validation["status"] != "passed":
        artifacts = _collect_artifacts(benchmark_dir, raw_dir)
        return _write_result(
            benchmark_dir,
            ArmBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                execution_status="not_run",
                parse_status="no_metrics",
                benchmark_status="failed",
                host=host_inspection.host,
                executable=executable,
                model=model,
                artifacts=BenchmarkArtifactManifest(artifacts=artifacts),
                provenance=provenance,
                artifact_directory=str(benchmark_dir),
                message=validation["message"],
            ),
        )
    inspection = inspect_llama_bench(job, benchmark_dir)
    atomic_write_json(benchmark_dir / "inspection.json", inspection.to_dict())
    provenance = decide_benchmark_provenance(job, host_inspection, executable=executable, inspection=inspection)
    if provenance.result_origin == "unverified":
        artifacts = _collect_artifacts(benchmark_dir, raw_dir)
        return _write_result(
            benchmark_dir,
            ArmBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                execution_status="not_run",
                parse_status="no_metrics",
                benchmark_status="failed",
                host=host_inspection.host,
                executable=executable,
                model=model,
                inspection=inspection,
                artifacts=BenchmarkArtifactManifest(artifacts=artifacts),
                provenance=provenance,
                artifact_directory=str(benchmark_dir),
                message=provenance.decision_reason,
            ),
        )
    flags = select_required_flags(inspection.supported_flags)
    missing = [name for name, value in flags.items() if value is None]
    if missing:
        artifacts = _collect_artifacts(benchmark_dir, raw_dir)
        return _write_result(
            benchmark_dir,
            ArmBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                execution_status="not_run",
                parse_status="no_metrics",
                benchmark_status="failed",
                host=host_inspection.host,
                executable=executable,
                model=model,
                inspection=inspection,
                artifacts=BenchmarkArtifactManifest(artifacts=artifacts),
                provenance=provenance,
                artifact_directory=str(benchmark_dir),
                message=f"llama-bench help is missing required flags: {', '.join(missing)}",
            ),
        )
    additional_flags = select_additional_flags(job.benchmark.additional_supported_options, inspection.supported_flags)
    unsupported_additional = [name for name, value in additional_flags.items() if value is None]
    if unsupported_additional:
        artifacts = _collect_artifacts(benchmark_dir, raw_dir)
        return _write_result(
            benchmark_dir,
            ArmBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                execution_status="not_run",
                parse_status="no_metrics",
                benchmark_status="failed",
                host=host_inspection.host,
                executable=executable,
                model=model,
                inspection=inspection,
                artifacts=BenchmarkArtifactManifest(artifacts=artifacts),
                provenance=provenance,
                artifact_directory=str(benchmark_dir),
                message=f"llama-bench help is missing configured optional flags: {', '.join(unsupported_additional)}",
            ),
        )
    invocation = execute_llama_bench(
        job,
        benchmark_dir,
        {key: value for key, value in flags.items() if value is not None},
        {key: value for key, value in additional_flags.items() if value is not None},
    )
    atomic_write_json(benchmark_dir / "invocation.json", invocation.to_dict())
    stdout_text = Path(invocation.raw_stdout_path).read_text(encoding="utf-8", errors="replace")
    stderr_text = Path(invocation.raw_stderr_path).read_text(encoding="utf-8", errors="replace")
    parse_result = parse_llama_bench_streams(stdout_text, stderr_text)
    atomic_write_json(benchmark_dir / "parse_result.json", parse_result.to_dict())
    execution_status = "timeout" if invocation.timed_out else ("completed" if invocation.exit_code == 0 else "failed")
    prefill, decode = required_metric_values(parse_result)
    required_metrics_present = prefill is not None and decode is not None
    benchmark_status = "passed" if execution_status == "completed" and parse_result.parse_status == "parsed" and required_metrics_present else "failed"
    trial = BenchmarkTrialResult(
        trial_index=1,
        validation_status="passed",
        execution_status=execution_status,
        parse_status=parse_result.parse_status,
        benchmark_status=benchmark_status,
        message="benchmark completed" if benchmark_status == "passed" else "benchmark did not produce required metrics",
        invocation=invocation,
        parse_result=parse_result,
    )
    atomic_write_json(benchmark_dir / "trials" / "0001" / "trial_result.json", trial.to_dict())
    artifacts = _collect_artifacts(benchmark_dir, raw_dir)
    result = ArmBenchmarkResult(
        run_id=job.run_id,
        operation=job.operation,
        target_type=job.target_type,
        result_origin=provenance.result_origin,
        validation_status="passed",
        execution_status=execution_status,
        parse_status=parse_result.parse_status,
        benchmark_status=benchmark_status,
        host=host_inspection.host,
        executable=executable,
        model=model,
        inspection=inspection,
        trial=trial,
        artifacts=BenchmarkArtifactManifest(artifacts=artifacts),
        provenance=provenance,
        artifact_directory=str(benchmark_dir),
        message=trial.message,
    )
    return _write_result(benchmark_dir, result)


def inspect_llama_bench(job: LlamaBenchJobSpec, benchmark_dir: Path) -> LlamaBenchInspection:
    raw_dir = benchmark_dir / "raw"
    executable = job.executable.path
    version = _run_command([executable, "--version"], timeout_seconds=min(10.0, job.benchmark.timeout_seconds))
    help_result = _run_command([executable, "--help"], timeout_seconds=min(10.0, job.benchmark.timeout_seconds))
    version_stdout = write_text_artifact(raw_dir / "inspection_version_stdout.txt", version.stdout)
    version_stderr = write_text_artifact(raw_dir / "inspection_version_stderr.txt", version.stderr)
    help_stdout = write_text_artifact(raw_dir / "inspection_help_stdout.txt", help_result.stdout)
    help_stderr = write_text_artifact(raw_dir / "inspection_help_stderr.txt", help_result.stderr)
    inspection_text = version.stdout + "\n" + version.stderr + "\n" + help_result.stdout + "\n" + help_result.stderr
    supported_flags = sorted(_extract_supported_flags(help_result.stdout + "\n" + help_result.stderr))
    return LlamaBenchInspection(
        version_command=[executable, "--version"],
        help_command=[executable, "--help"],
        version_exit_code=version.return_code,
        help_exit_code=help_result.return_code,
        version_status="passed" if version.return_code == 0 else "failed",
        help_status="passed" if help_result.return_code == 0 else "failed",
        raw_version_stdout_path=str(version_stdout),
        raw_version_stderr_path=str(version_stderr),
        raw_help_stdout_path=str(help_stdout),
        raw_help_stderr_path=str(help_stderr),
        supported_flags=supported_flags,
        source_commit=_extract_commit(version.stdout + "\n" + version.stderr),
        fixture_marker_detected=_contains_fixture_marker(inspection_text),
    )


def execute_llama_bench(
    job: LlamaBenchJobSpec,
    benchmark_dir: Path,
    flags: dict[str, str],
    additional_flags: dict[str, str],
) -> LlamaBenchInvocation:
    raw_dir = benchmark_dir / "raw"
    command = [
        job.executable.path,
        flags["model"],
        job.model.path,
        flags["threads"],
        str(job.benchmark.threads),
        flags["prompt_tokens"],
        str(job.benchmark.prompt_tokens),
        flags["generation_tokens"],
        str(job.benchmark.generation_tokens),
        flags["repetitions"],
        str(job.benchmark.repetitions),
    ]
    for key, value in sorted(job.benchmark.additional_supported_options.items()):
        command.extend([additional_flags[key], str(value)])
    started = utc_now()
    start_monotonic = time.monotonic()
    result = _run_command(command, timeout_seconds=job.benchmark.timeout_seconds, cpu_affinity=job.benchmark.cpu_affinity)
    ended = utc_now()
    duration = time.monotonic() - start_monotonic
    stdout_path = write_text_artifact(raw_dir / "benchmark_stdout.txt", result.stdout)
    stderr_path = write_text_artifact(raw_dir / "benchmark_stderr.txt", result.stderr)
    return LlamaBenchInvocation(
        command=command,
        started_at_utc=started,
        ended_at_utc=ended,
        duration_seconds=round(duration, 6),
        exit_code=result.return_code,
        timed_out=result.timed_out,
        raw_stdout_path=str(stdout_path),
        raw_stderr_path=str(stderr_path),
    )


def select_required_flags(supported_flags: list[str]) -> dict[str, str | None]:
    supported = set(supported_flags)
    selected: dict[str, str | None] = {}
    for name, candidates in REQUIRED_FLAGS.items():
        selected[name] = next((candidate for candidate in candidates if candidate in supported), None)
    return selected


def select_additional_flags(options: dict[str, int], supported_flags: list[str]) -> dict[str, str | None]:
    supported = set(supported_flags)
    selected: dict[str, str | None] = {}
    for name in options:
        selected[name] = next((candidate for candidate in OPTIONAL_FLAGS.get(name, []) if candidate in supported), None)
    return selected


def _validate_files(job: LlamaBenchJobSpec) -> dict[str, Any]:
    executable = _file_manifest(job.executable.path, expected_sha256=job.executable.expected_sha256, require_executable=True)
    if executable.validation_status != "passed":
        return {"status": "failed", "message": executable.validation_message, "executable": executable, "model": None}
    model = _file_manifest(job.model.path, expected_sha256=job.model.expected_sha256, require_executable=False)
    if model.validation_status != "passed":
        return {"status": "failed", "message": model.validation_message, "executable": executable, "model": model}
    return {"status": "passed", "message": "validation passed", "executable": executable, "model": model}


def _file_manifest(path_text: str, *, expected_sha256: str | None, require_executable: bool) -> BenchmarkFileManifest:
    path = Path(path_text)
    basename = path.name
    if not _is_absolute_target_path(path_text):
        return _invalid_manifest(path_text, basename, expected_sha256, "path must be absolute")
    if not path.exists():
        return _invalid_manifest(path_text, basename, expected_sha256, "file does not exist")
    if not path.is_file():
        return _invalid_manifest(path_text, basename, expected_sha256, "path is not a regular file")
    if require_executable and not _is_executable(path):
        return _invalid_manifest(path_text, basename, expected_sha256, "file is not executable")
    try:
        digest = sha256_file(path)
        size = path.stat().st_size
    except OSError as exc:
        return _invalid_manifest(path_text, basename, expected_sha256, f"file is not readable: {exc}")
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        return BenchmarkFileManifest(
            path=path_text,
            basename=basename,
            size_bytes=size,
            sha256=digest,
            expected_sha256=expected_sha256,
            validation_status="failed",
            validation_message="sha256 mismatch",
        )
    return BenchmarkFileManifest(
        path=path_text,
        basename=basename,
        size_bytes=size,
        sha256=digest,
        expected_sha256=expected_sha256,
        validation_status="passed",
        validation_message="validation passed",
    )


def _invalid_manifest(path: str, basename: str, expected_sha256: str | None, message: str) -> BenchmarkFileManifest:
    return BenchmarkFileManifest(
        path=path,
        basename=basename,
        size_bytes=0,
        sha256="",
        expected_sha256=expected_sha256,
        validation_status="failed",
        validation_message=message,
    )


def _is_absolute_target_path(path: str) -> bool:
    return Path(path).is_absolute() or path.startswith("/")


def _is_executable(path: Path) -> bool:
    if os.name == "nt":
        return path.suffix.lower() in {".exe", ".bat", ".cmd", ".com"}
    return os.access(path, os.X_OK)


def _run_command(command: list[str], *, timeout_seconds: float, cpu_affinity: list[int] | None = None):
    preexec_fn = None
    if cpu_affinity and os.name != "nt" and hasattr(os, "sched_setaffinity"):
        affinity = set(cpu_affinity)

        def _set_child_affinity() -> None:
            os.sched_setaffinity(0, affinity)

        preexec_fn = _set_child_affinity
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            preexec_fn=preexec_fn,
        )
        return replaceable_command_result(command, completed.returncode, completed.stdout or "", completed.stderr or "", False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return replaceable_command_result(command, 124, stdout, stderr, True)
    except OSError as exc:
        return replaceable_command_result(command, 127, "", str(exc), False)


def replaceable_command_result(command: list[str], return_code: int, stdout: str, stderr: str, timed_out: bool):
    from autotune.arm_tune.schemas import CommandResult

    return CommandResult(command=command, return_code=return_code, stdout=stdout, stderr=stderr, timed_out=timed_out)


def _extract_supported_flags(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)(?:--[A-Za-z][A-Za-z0-9-]*|-[A-Za-z])", text))


def _extract_commit(text: str) -> str:
    match = re.search(r"(?:commit|build)[:= ]+([0-9a-f]{7,40})", text, flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def _contains_fixture_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("fake", "fixture", "synthetic"))


def _collect_artifacts(benchmark_dir: Path, raw_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts = {
        "benchmark_job": artifact_record(benchmark_dir / "benchmark_job.json", label="benchmark_job", root=benchmark_dir),
        "executable_manifest": artifact_record(benchmark_dir / "executable_manifest.json", label="executable_manifest", root=benchmark_dir),
        "model_manifest": artifact_record(benchmark_dir / "model_manifest.json", label="model_manifest", root=benchmark_dir),
        "inspection": artifact_record(benchmark_dir / "inspection.json", label="inspection", root=benchmark_dir),
        "invocation": artifact_record(benchmark_dir / "invocation.json", label="invocation", root=benchmark_dir),
        "parse_result": artifact_record(benchmark_dir / "parse_result.json", label="parse_result", root=benchmark_dir),
        "trial_result": artifact_record(benchmark_dir / "trials" / "0001" / "trial_result.json", label="trial_result", root=benchmark_dir),
        "benchmark_stdout": artifact_record(raw_dir / "benchmark_stdout.txt", label="benchmark_stdout", root=benchmark_dir),
        "benchmark_stderr": artifact_record(raw_dir / "benchmark_stderr.txt", label="benchmark_stderr", root=benchmark_dir),
        "inspection_version_stdout": artifact_record(raw_dir / "inspection_version_stdout.txt", label="inspection_version_stdout", root=benchmark_dir),
        "inspection_version_stderr": artifact_record(raw_dir / "inspection_version_stderr.txt", label="inspection_version_stderr", root=benchmark_dir),
        "inspection_help_stdout": artifact_record(raw_dir / "inspection_help_stdout.txt", label="inspection_help_stdout", root=benchmark_dir),
        "inspection_help_stderr": artifact_record(raw_dir / "inspection_help_stderr.txt", label="inspection_help_stderr", root=benchmark_dir),
    }
    write_artifact_manifest(benchmark_dir / "artifact_manifest.json", artifacts)
    return artifacts


def _write_file_manifests(
    benchmark_dir: Path,
    executable: BenchmarkFileManifest | None,
    model: BenchmarkFileManifest | None,
) -> None:
    if executable is not None:
        atomic_write_json(benchmark_dir / "executable_manifest.json", executable.to_dict())
    if model is not None:
        atomic_write_json(benchmark_dir / "model_manifest.json", model.to_dict())


def _write_result(benchmark_dir: Path, result: ArmBenchmarkResult) -> ArmBenchmarkResult:
    atomic_write_json(benchmark_dir / "benchmark_result.json", result.to_dict())
    return result


def ensure_benchmark_operation(job: LlamaBenchJobSpec) -> None:
    if job.operation != BENCHMARK_OPERATION:
        raise ArmTuneConfigError(f"unsupported benchmark operation: {job.operation}")
    if job.protocol_version != PROTOCOL_VERSION:
        raise ArmTuneConfigError(f"unsupported benchmark protocol_version: {job.protocol_version}")
