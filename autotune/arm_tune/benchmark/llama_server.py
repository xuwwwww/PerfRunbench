from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest, write_text_artifact
from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.benchmark.provenance import FIXTURE_MARKERS
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.platform.environment import collect_host_inspection
from autotune.arm_tune.schemas import (
    ArtifactValidationResult,
    BenchmarkArtifactManifest,
    BenchmarkFileManifest,
    BenchmarkProvenance,
    GenerationMetricDistribution,
    GenerationRequestResult,
    HostInspectionResult,
    LlamaServerBenchmarkResult,
    LlamaServerConfiguration,
    LlamaServerInspection,
    LlamaServerJobSpec,
    PromptSuiteEntry,
    ServerLifecycleResult,
    StreamingEventRecord,
    utc_now,
)
from autotune.arm_tune.targets.protocol import atomic_write_json


SERVER_BENCHMARK_OPERATION = "benchmark.llama_server"
PROTOCOL_VERSION = "1.0"
REQUIRED_SERVER_FLAGS = {
    "model": ["-m", "--model"],
    "host": ["--host"],
    "port": ["--port"],
}
OPTIONAL_SERVER_FLAGS = {
    "threads": ["-t", "--threads"],
    "context_size": ["-c", "--ctx-size"],
    "batch_size": ["--batch-size"],
    "ubatch_size": ["--ubatch-size"],
    "seed": ["--seed"],
    "temperature": ["--temp", "--temperature"],
}


class _StartedServer:
    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        stdout_handle: Any,
        stderr_handle: Any,
        stdout_path: Path,
        stderr_path: Path,
        port: int,
        command: list[str],
        start_monotonic: float,
    ) -> None:
        self.process = process
        self.stdout_handle = stdout_handle
        self.stderr_handle = stderr_handle
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path
        self.port = port
        self.command = command
        self.start_monotonic = start_monotonic


def run_llama_server_worker(job: LlamaServerJobSpec, run_dir: str | Path) -> LlamaServerBenchmarkResult:
    server_dir = Path(run_dir)
    raw_dir = server_dir / "raw"
    requests_dir = server_dir / "requests"
    server_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    requests_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(server_dir / "server_job.json", job.to_dict())
    host_inspection = collect_host_inspection(target_type=job.target_type)
    provenance = decide_server_provenance(job, host_inspection)
    if provenance.result_origin != "test_fixture" and not host_inspection.production_validation.valid:
        return _write_final_result(
            server_dir,
            LlamaServerBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                inspection_status="not_run",
                startup_status="not_started",
                readiness_status="not_checked",
                request_status="not_run",
                stream_parse_status="not_parsed",
                cleanup_status="not_started",
                benchmark_status="failed",
                quality_gate_status="not_evaluated",
                host=host_inspection.host,
                artifacts=BenchmarkArtifactManifest(artifacts=_collect_artifacts(server_dir)),
                provenance=provenance,
                artifact_directory=str(server_dir),
                message=host_inspection.production_validation.reason,
            ),
        )

    validation = _validate_files(job)
    executable = validation.get("executable")
    model = validation.get("model")
    _write_file_manifests(server_dir, executable, model)
    provenance = decide_server_provenance(job, host_inspection, executable=executable)
    if validation["status"] != "passed":
        return _write_final_result(
            server_dir,
            LlamaServerBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                inspection_status="not_run",
                startup_status="not_started",
                readiness_status="not_checked",
                request_status="not_run",
                stream_parse_status="not_parsed",
                cleanup_status="not_started",
                benchmark_status="failed",
                quality_gate_status="not_evaluated",
                host=host_inspection.host,
                executable=executable,
                model=model,
                artifacts=BenchmarkArtifactManifest(artifacts=_collect_artifacts(server_dir)),
                provenance=provenance,
                artifact_directory=str(server_dir),
                message=validation["message"],
            ),
        )

    inspection = inspect_llama_server(job, server_dir)
    atomic_write_json(server_dir / "inspection.json", inspection.to_dict())
    provenance = decide_server_provenance(job, host_inspection, executable=executable, inspection=inspection)
    if provenance.result_origin == "unverified":
        return _write_final_result(
            server_dir,
            LlamaServerBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                inspection_status="passed" if inspection.help_status == "passed" else "failed",
                startup_status="not_started",
                readiness_status="not_checked",
                request_status="not_run",
                stream_parse_status="not_parsed",
                cleanup_status="not_started",
                benchmark_status="failed",
                quality_gate_status="not_evaluated",
                host=host_inspection.host,
                executable=executable,
                model=model,
                inspection=inspection,
                artifacts=BenchmarkArtifactManifest(artifacts=_collect_artifacts(server_dir)),
                provenance=provenance,
                artifact_directory=str(server_dir),
                message=provenance.decision_reason,
            ),
        )

    flag_selection = select_server_flags(inspection.supported_flags)
    missing = [name for name in REQUIRED_SERVER_FLAGS if flag_selection.get(name) is None]
    if missing:
        return _write_final_result(
            server_dir,
            LlamaServerBenchmarkResult(
                run_id=job.run_id,
                operation=job.operation,
                target_type=job.target_type,
                result_origin=provenance.result_origin,
                validation_status="failed",
                inspection_status="passed",
                startup_status="not_started",
                readiness_status="not_checked",
                request_status="not_run",
                stream_parse_status="not_parsed",
                cleanup_status="not_started",
                benchmark_status="failed",
                quality_gate_status="not_evaluated",
                host=host_inspection.host,
                executable=executable,
                model=model,
                inspection=inspection,
                artifacts=BenchmarkArtifactManifest(artifacts=_collect_artifacts(server_dir)),
                provenance=provenance,
                artifact_directory=str(server_dir),
                message=f"llama-server help is missing required flags: {', '.join(missing)}",
            ),
        )

    prompts = load_prompt_suite(job.workload.prompt_suite_path)
    port = _select_port(job.server.port)
    command, used_flags, unsupported_optional = build_server_command(job, flag_selection, port)
    configuration = LlamaServerConfiguration(
        host=job.server.host,
        port=port,
        command=command,
        supported_flags_used=used_flags,
        unsupported_optional_fields=unsupported_optional,
        affinity_mode=job.server.affinity_mode,
        cpu_affinity=list(job.server.cpu_affinity) if job.server.cpu_affinity is not None else None,
        affinity_applied=_can_apply_affinity(job.server.cpu_affinity),
    )
    atomic_write_json(server_dir / "server_configuration.json", configuration.to_dict())

    started: _StartedServer | None = None
    lifecycle = ServerLifecycleResult(selected_port=port)
    requests: list[GenerationRequestResult] = []
    try:
        started = start_server(command, raw_dir, port, cpu_affinity=job.server.cpu_affinity)
        lifecycle = wait_until_ready(started, job)
        if lifecycle.readiness_status == "ready":
            requests = run_prompt_suite_requests(job, prompts, requests_dir, port)
        else:
            requests = []
    finally:
        cleanup = cleanup_server(started, job) if started else ServerLifecycleResult(
            process_started=False,
            startup_status=lifecycle.startup_status,
            readiness_status=lifecycle.readiness_status,
            cleanup_status="not_started",
            selected_port=port,
            port_released=_port_released(job.server.host, port),
        )
        lifecycle = replace(
            lifecycle,
            cleanup_status=cleanup.cleanup_status,
            cleanup_mode=cleanup.cleanup_mode,
            cleanup_message=cleanup.cleanup_message,
            port_released=cleanup.port_released,
        )
        atomic_write_json(server_dir / "lifecycle.json", lifecycle.to_dict())

    measured_requests = [item for item in requests if item.phase == "measured"]
    failed_requests = [item for item in measured_requests if item.request_status != "completed" or item.stream_parse_status != "parsed"]
    request_status = "completed" if measured_requests and not failed_requests else ("failed" if measured_requests else "not_run")
    stream_parse_status = "parsed" if measured_requests and not failed_requests else ("failed" if measured_requests else "not_parsed")
    benchmark_status = (
        "passed"
        if validation["status"] == "passed"
        and lifecycle.startup_status == "ready"
        and lifecycle.readiness_status == "ready"
        and request_status == "completed"
        and lifecycle.cleanup_status in {"passed", "forced"}
        else "failed"
    )
    if lifecycle.cleanup_status == "failed":
        benchmark_status = "failed"
    result = LlamaServerBenchmarkResult(
        run_id=job.run_id,
        operation=job.operation,
        target_type=job.target_type,
        result_origin=provenance.result_origin,
        validation_status="passed",
        inspection_status="passed" if inspection.help_status == "passed" else "failed",
        startup_status=lifecycle.startup_status,
        readiness_status=lifecycle.readiness_status,
        request_status=request_status,
        stream_parse_status=stream_parse_status,
        cleanup_status=lifecycle.cleanup_status,
        benchmark_status=benchmark_status,
        quality_gate_status="not_evaluated",
        host=host_inspection.host,
        executable=executable,
        model=model,
        inspection=inspection,
        server_configuration=configuration,
        lifecycle=lifecycle,
        requests=requests,
        distributions=summarize_generation_metrics(requests),
        provenance=provenance,
        artifact_directory=str(server_dir),
        message=_result_message(benchmark_status, lifecycle, failed_requests),
    )
    return _write_final_result(server_dir, result)


def inspect_llama_server(job: LlamaServerJobSpec, server_dir: Path) -> LlamaServerInspection:
    raw_dir = server_dir / "raw"
    executable = job.executable.path
    version = _run_command([executable, "--version"], timeout_seconds=min(10.0, job.server.startup_timeout_seconds))
    help_result = _run_command([executable, "--help"], timeout_seconds=min(10.0, job.server.startup_timeout_seconds))
    version_stdout = write_text_artifact(raw_dir / "inspection_version_stdout.txt", version.stdout)
    version_stderr = write_text_artifact(raw_dir / "inspection_version_stderr.txt", version.stderr)
    help_stdout = write_text_artifact(raw_dir / "inspection_help_stdout.txt", help_result.stdout)
    help_stderr = write_text_artifact(raw_dir / "inspection_help_stderr.txt", help_result.stderr)
    inspection_text = version.stdout + "\n" + version.stderr + "\n" + help_result.stdout + "\n" + help_result.stderr
    return LlamaServerInspection(
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
        supported_flags=sorted(_extract_supported_flags(help_result.stdout + "\n" + help_result.stderr)),
        source_commit=_extract_commit(version.stdout + "\n" + version.stderr),
        runtime_build=_extract_runtime_build(version.stdout + "\n" + version.stderr),
        fixture_marker_detected=_contains_fixture_marker(inspection_text),
    )


def select_server_flags(supported_flags: list[str]) -> dict[str, str | None]:
    supported = set(supported_flags)
    selected: dict[str, str | None] = {}
    for name, candidates in {**REQUIRED_SERVER_FLAGS, **OPTIONAL_SERVER_FLAGS}.items():
        selected[name] = next((candidate for candidate in candidates if candidate in supported), None)
    return selected


def build_server_command(
    job: LlamaServerJobSpec,
    flags: dict[str, str | None],
    port: int,
) -> tuple[list[str], dict[str, str], list[str]]:
    used: dict[str, str] = {}
    unsupported_optional: list[str] = []
    command = [
        job.executable.path,
        _required_flag(flags, "model"),
        job.model.path,
        _required_flag(flags, "host"),
        job.server.host,
        _required_flag(flags, "port"),
        str(port),
    ]
    for field_name, value in [
        ("threads", job.server.threads),
        ("context_size", job.server.context_size),
        ("batch_size", job.server.batch_size),
        ("ubatch_size", job.server.ubatch_size),
        ("seed", job.server.seed),
        ("temperature", job.server.temperature),
    ]:
        if value is None:
            continue
        flag = flags.get(field_name)
        if flag is None:
            unsupported_optional.append(field_name)
            continue
        used[field_name] = flag
        command.extend([flag, str(value)])
    used["model"] = _required_flag(flags, "model")
    used["host"] = _required_flag(flags, "host")
    used["port"] = _required_flag(flags, "port")
    return command, used, unsupported_optional


def start_server(command: list[str], raw_dir: Path, port: int, *, cpu_affinity: list[int] | None = None) -> _StartedServer:
    stdout_path = raw_dir / "server_stdout.txt"
    stderr_path = raw_dir / "server_stderr.txt"
    stdout_handle = stdout_path.open("w", encoding="utf-8", errors="replace", newline="\n")
    stderr_handle = stderr_path.open("w", encoding="utf-8", errors="replace", newline="\n")
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        start_new_session = True
    try:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            text=True,
            shell=False,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
        _apply_process_affinity(process.pid, cpu_affinity)
    except Exception:
        try:
            if "process" in locals() and process.poll() is None:
                process.terminate()
        except Exception:
            pass
        stdout_handle.close()
        stderr_handle.close()
        raise
    return _StartedServer(
        process=process,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        port=port,
        command=command,
        start_monotonic=time.monotonic(),
    )


def wait_until_ready(started: _StartedServer, job: LlamaServerJobSpec) -> ServerLifecycleResult:
    deadline = time.monotonic() + job.server.startup_timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        exit_code = started.process.poll()
        if exit_code is not None:
            return ServerLifecycleResult(
                process_started=True,
                startup_status="exited_before_ready",
                readiness_status="failed",
                cleanup_status="not_started",
                pid=started.process.pid,
                selected_port=started.port,
                startup_duration_seconds=round(time.monotonic() - started.start_monotonic, 6),
                readiness_message=f"server exited before readiness with code {exit_code}",
            )
        try:
            with urllib.request.urlopen(f"http://{job.server.host}:{started.port}/health", timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return ServerLifecycleResult(
                        process_started=True,
                        startup_status="ready",
                        readiness_status="ready",
                        cleanup_status="not_started",
                        pid=started.process.pid,
                        selected_port=started.port,
                        startup_duration_seconds=round(time.monotonic() - started.start_monotonic, 6),
                        readiness_message=f"health endpoint returned HTTP {response.status}",
                    )
                last_error = f"health endpoint returned HTTP {response.status}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.05)
    return ServerLifecycleResult(
        process_started=True,
        startup_status="startup_timeout",
        readiness_status="timeout",
        cleanup_status="not_started",
        pid=started.process.pid,
        selected_port=started.port,
        startup_duration_seconds=round(time.monotonic() - started.start_monotonic, 6),
        readiness_message=last_error or "startup timeout",
    )


def run_prompt_suite_requests(
    job: LlamaServerJobSpec,
    prompts: list[PromptSuiteEntry],
    requests_dir: Path,
    port: int,
) -> list[GenerationRequestResult]:
    results: list[GenerationRequestResult] = []
    for prompt in prompts:
        for phase, count in [("warmup", job.workload.warmup_requests), ("measured", job.workload.repetitions)]:
            for index in range(1, count + 1):
                result = run_generation_request(job, prompt, requests_dir, port, phase=phase, repetition_index=index)
                results.append(result)
                if job.workload.cooldown_seconds > 0:
                    time.sleep(job.workload.cooldown_seconds)
    return results


def run_generation_request(
    job: LlamaServerJobSpec,
    prompt: PromptSuiteEntry,
    requests_dir: Path,
    port: int,
    *,
    phase: str,
    repetition_index: int,
) -> GenerationRequestResult:
    request_dir = requests_dir / prompt.prompt_id / phase / f"{repetition_index:04d}"
    request_dir.mkdir(parents=True, exist_ok=True)
    max_tokens = prompt.max_predict_tokens or job.workload.predict_tokens
    request_payload = {
        "schema_version": "1.0",
        "prompt_id": prompt.prompt_id,
        "phase": phase,
        "repetition_index": repetition_index,
        "request": {
            "model": "local",
            "messages": [{"role": "user", "content": prompt.prompt}],
            "max_tokens": max_tokens,
            "temperature": job.server.temperature,
            "seed": job.server.seed,
            "stream": job.workload.streaming,
        },
    }
    atomic_write_json(request_dir / "request.json", request_payload)
    started = time.monotonic()
    url = f"http://{job.server.host}:{port}/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload["request"]).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    events: list[StreamingEventRecord] = []
    generated_parts: list[str] = []
    finish_reason: str | None = None
    first_content_time: float | None = None
    http_status: int | None = None
    response_headers: dict[str, str] = {}
    error = ""
    request_status = "failed"
    stream_parse_status = "failed"
    try:
        with urllib.request.urlopen(request, timeout=job.server.request_timeout_seconds) as response:
            http_status = response.status
            response_headers = _selected_headers(dict(response.headers.items()))
            for raw in response:
                now = time.monotonic()
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                event = parse_streaming_line(line, len(events) + 1, received_offset_ms=round((now - started) * 1000, 3))
                events.append(event)
                if event.valid_generated_content:
                    if first_content_time is None:
                        first_content_time = now
                    generated_parts.append(event.content_delta)
                if event.finish_reason:
                    finish_reason = event.finish_reason
                if event.event_type == "done":
                    break
            request_status = "completed" if http_status and 200 <= http_status < 300 else "failed"
            stream_parse_status = _stream_status(events, require_content=True)
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        response_headers = _selected_headers(dict(exc.headers.items())) if exc.headers else {}
        error = f"HTTP {exc.code}"
        request_status = "failed"
        stream_parse_status = "http_error"
    except TimeoutError as exc:
        error = str(exc) or "request timeout"
        request_status = "timeout"
        stream_parse_status = "timeout"
    except Exception as exc:
        error = str(exc)
        request_status = "failed"
        stream_parse_status = "failed"
    completed = time.monotonic()
    generated_text = "".join(generated_parts)
    _write_stream_jsonl(request_dir / "stream.jsonl", events)
    write_text_artifact(request_dir / "output.txt", generated_text)
    result = GenerationRequestResult(
        prompt_id=prompt.prompt_id,
        phase=phase,
        repetition_index=repetition_index,
        request_status=request_status,
        stream_parse_status=stream_parse_status,
        http_status=http_status,
        ttft_ms=round((first_content_time - started) * 1000, 3) if first_content_time is not None else None,
        end_to_end_latency_ms=round((completed - started) * 1000, 3) if request_status == "completed" else None,
        generated_token_count=None,
        generated_text=generated_text,
        finish_reason=finish_reason,
        response_headers=response_headers,
        error=error,
        request_artifact_dir=str(request_dir),
        stream_event_count=len(events),
    )
    atomic_write_json(request_dir / "result.json", result.to_dict())
    return result


def parse_streaming_line(line: str, event_index: int, *, received_offset_ms: float) -> StreamingEventRecord:
    if not line:
        return StreamingEventRecord(
            event_index=event_index,
            raw_line=line,
            event_type="empty",
            received_offset_ms=received_offset_ms,
            parse_status="ignored",
        )
    if not line.startswith("data:"):
        return StreamingEventRecord(
            event_index=event_index,
            raw_line=line,
            event_type="other",
            received_offset_ms=received_offset_ms,
            parse_status="ignored",
        )
    data = line[5:].strip()
    if data == "[DONE]":
        return StreamingEventRecord(
            event_index=event_index,
            raw_line=line,
            event_type="done",
            received_offset_ms=received_offset_ms,
        )
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        return StreamingEventRecord(
            event_index=event_index,
            raw_line=line,
            event_type="malformed",
            received_offset_ms=received_offset_ms,
            parse_status="malformed",
            message=str(exc),
        )
    content, finish_reason = _content_from_payload(payload)
    return StreamingEventRecord(
        event_index=event_index,
        raw_line=line,
        event_type="content" if content else "metadata",
        content_delta=content,
        finish_reason=finish_reason,
        valid_generated_content=bool(content),
        received_offset_ms=received_offset_ms,
    )


def cleanup_server(started: _StartedServer, job: LlamaServerJobSpec) -> ServerLifecycleResult:
    process = started.process
    cleanup_mode = "already_exited"
    cleanup_message = ""
    if process.poll() is None:
        cleanup_mode = "terminated"
        try:
            _terminate_process(process)
            process.wait(timeout=job.server.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            cleanup_mode = "forced"
            _kill_process(process)
            try:
                process.wait(timeout=job.server.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                cleanup_message = "process did not exit after forced termination"
        except OSError as exc:
            cleanup_message = str(exc)
    for handle in [started.stdout_handle, started.stderr_handle]:
        try:
            handle.close()
        except OSError:
            pass
    port_released = _wait_for_port_release(job.server.host, started.port, timeout_seconds=max(0.5, job.server.shutdown_timeout_seconds))
    cleanup_status = "passed" if not cleanup_message and port_released else "failed"
    if cleanup_mode == "forced" and cleanup_status == "passed":
        cleanup_status = "forced"
    return ServerLifecycleResult(
        process_started=True,
        startup_status="ready",
        readiness_status="ready",
        cleanup_status=cleanup_status,
        cleanup_mode=cleanup_mode,
        cleanup_message=cleanup_message,
        port_released=port_released,
        pid=process.pid,
        selected_port=started.port,
    )


def summarize_generation_metrics(requests: list[GenerationRequestResult]) -> dict[str, GenerationMetricDistribution]:
    measured = [request for request in requests if request.phase == "measured"]
    failed = sum(1 for request in measured if request.request_status != "completed" or request.stream_parse_status != "parsed")
    return {
        "ttft_ms": _distribution("ttft_ms", [request.ttft_ms for request in measured if request.ttft_ms is not None], failed),
        "end_to_end_latency_ms": _distribution(
            "end_to_end_latency_ms",
            [request.end_to_end_latency_ms for request in measured if request.end_to_end_latency_ms is not None],
            failed,
        ),
    }


def load_prompt_suite(path_text: str) -> list[PromptSuiteEntry]:
    path = Path(path_text)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("prompt suite must be an object")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ArmTuneConfigError("prompt suite must contain prompts")
    entries: list[PromptSuiteEntry] = []
    seen: set[str] = set()
    for item in prompts:
        if not isinstance(item, dict):
            raise ArmTuneConfigError("prompt suite entries must be objects")
        prompt_id = str(item.get("prompt_id", "")).strip()
        if not prompt_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", prompt_id):
            raise ArmTuneConfigError("prompt_id must be non-empty and filesystem-safe")
        if prompt_id in seen:
            raise ArmTuneConfigError(f"duplicate prompt_id: {prompt_id}")
        seen.add(prompt_id)
        prompt = str(item.get("prompt", ""))
        if not prompt:
            raise ArmTuneConfigError(f"prompt {prompt_id} must not be empty")
        max_tokens = item.get("max_predict_tokens")
        if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1):
            raise ArmTuneConfigError("max_predict_tokens must be a positive integer")
        deterministic = item.get("deterministic_settings", {})
        if deterministic is None:
            deterministic = {}
        if not isinstance(deterministic, dict):
            raise ArmTuneConfigError("deterministic_settings must be an object")
        entries.append(
            PromptSuiteEntry(
                prompt_id=prompt_id,
                prompt=prompt,
                expected_response_mode=str(item.get("expected_response_mode", "text")),
                max_predict_tokens=max_tokens,
                deterministic_settings=dict(deterministic),
            )
        )
    return entries


def decide_server_provenance(
    job: LlamaServerJobSpec,
    host_inspection: HostInspectionResult,
    *,
    executable: BenchmarkFileManifest | None = None,
    inspection: LlamaServerInspection | None = None,
) -> BenchmarkProvenance:
    requested = job.result_origin
    host_ready = host_inspection.production_validation.valid
    fake_marker = _has_fake_server_marker(job, executable, inspection)
    test_override = _has_test_only_host_override(host_inspection)
    evidence = [f"requested_origin={requested}", f"host_production_ready={'true' if host_ready else 'false'}"]
    if fake_marker:
        evidence.append("fake_or_fixture_executable_marker=true")
    if test_override:
        evidence.append("test_only_host_override=true")
    if requested == "test_fixture" or fake_marker or test_override:
        return BenchmarkProvenance(
            result_origin="test_fixture",
            requested_origin=requested,
            decision_reason="fixture provenance marker overrides target architecture",
            evidence=evidence,
            host_production_ready=host_ready,
            fake_executable_marker=fake_marker,
            test_only_host_override=test_override,
        )
    if requested == "measured" and host_ready:
        return BenchmarkProvenance(
            result_origin="measured",
            requested_origin=requested,
            decision_reason="measured origin requires Linux Arm64 target and no fixture markers",
            evidence=evidence,
            host_production_ready=host_ready,
            fake_executable_marker=False,
            test_only_host_override=False,
        )
    return BenchmarkProvenance(
        result_origin="unverified",
        requested_origin=requested,
        decision_reason="missing measured provenance or unsupported production target",
        evidence=evidence,
        host_production_ready=host_ready,
        fake_executable_marker=fake_marker,
        test_only_host_override=test_override,
    )


def apply_server_artifact_validation(
    result: LlamaServerBenchmarkResult,
    validation: ArtifactValidationResult,
    artifact_directory: str | Path,
) -> LlamaServerBenchmarkResult:
    benchmark_status = result.benchmark_status
    message = result.message
    if validation.validation_status != "passed" and benchmark_status == "passed":
        benchmark_status = "failed"
        message = f"{message}; artifact validation failed" if message else "artifact validation failed"
    return replace(
        result,
        benchmark_status=benchmark_status,
        artifact_validation=validation,
        artifact_directory=str(artifact_directory),
    )


def finalize_server_result_artifacts(
    server_dir: str | Path,
    result: LlamaServerBenchmarkResult,
) -> LlamaServerBenchmarkResult:
    return _write_final_result(Path(server_dir), result)


def ensure_server_benchmark_operation(job: LlamaServerJobSpec) -> None:
    if job.operation != SERVER_BENCHMARK_OPERATION:
        raise ArmTuneConfigError(f"unsupported benchmark operation: {job.operation}")
    if job.protocol_version != PROTOCOL_VERSION:
        raise ArmTuneConfigError(f"unsupported benchmark protocol_version: {job.protocol_version}")


def _validate_files(job: LlamaServerJobSpec) -> dict[str, Any]:
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


def _write_file_manifests(
    server_dir: Path,
    executable: BenchmarkFileManifest | None,
    model: BenchmarkFileManifest | None,
) -> None:
    if executable is not None:
        atomic_write_json(server_dir / "executable_manifest.json", executable.to_dict())
    if model is not None:
        atomic_write_json(server_dir / "model_manifest.json", model.to_dict())


def _write_final_result(server_dir: Path, result: LlamaServerBenchmarkResult) -> LlamaServerBenchmarkResult:
    atomic_write_json(server_dir / "benchmark_result.json", result.to_dict())
    artifacts = _collect_artifacts(server_dir)
    final = replace(result, artifacts=BenchmarkArtifactManifest(artifacts=artifacts))
    atomic_write_json(server_dir / "benchmark_result.json", final.to_dict())
    artifacts = _collect_artifacts(server_dir)
    final = replace(final, artifacts=BenchmarkArtifactManifest(artifacts=artifacts))
    atomic_write_json(server_dir / "benchmark_result.json", final.to_dict())
    return final


def _collect_artifacts(server_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    labels = {
        "server_job": server_dir / "server_job.json",
        "executable_manifest": server_dir / "executable_manifest.json",
        "model_manifest": server_dir / "model_manifest.json",
        "inspection": server_dir / "inspection.json",
        "server_configuration": server_dir / "server_configuration.json",
        "lifecycle": server_dir / "lifecycle.json",
        "inspection_version_stdout": server_dir / "raw" / "inspection_version_stdout.txt",
        "inspection_version_stderr": server_dir / "raw" / "inspection_version_stderr.txt",
        "inspection_help_stdout": server_dir / "raw" / "inspection_help_stdout.txt",
        "inspection_help_stderr": server_dir / "raw" / "inspection_help_stderr.txt",
        "server_stdout": server_dir / "raw" / "server_stdout.txt",
        "server_stderr": server_dir / "raw" / "server_stderr.txt",
    }
    for label, path in labels.items():
        if path.exists():
            artifacts[label] = artifact_record(path, label=label, root=server_dir)
    for path in sorted((server_dir / "requests").rglob("*")) if (server_dir / "requests").exists() else []:
        if path.is_file():
            label = "request_" + path.relative_to(server_dir).as_posix().replace("/", "_").replace(".", "_")
            artifacts[label] = artifact_record(path, label=label, root=server_dir)
    write_artifact_manifest(server_dir / "artifact_manifest.json", artifacts)
    return artifacts


def _run_command(command: list[str], *, timeout_seconds: float):
    from autotune.arm_tune.schemas import CommandResult

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        return CommandResult(command=command, return_code=completed.returncode, stdout=completed.stdout or "", stderr=completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(command=command, return_code=124, stdout=stdout, stderr=stderr, timed_out=True)
    except OSError as exc:
        return CommandResult(command=command, return_code=127, stderr=str(exc))


def _required_flag(flags: dict[str, str | None], name: str) -> str:
    value = flags.get(name)
    if value is None:
        raise ArmTuneConfigError(f"llama-server help is missing required flag: {name}")
    return value


def _select_port(port: int | str) -> int:
    if port != "auto":
        return int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _can_apply_affinity(cpu_affinity: list[int] | None) -> bool | None:
    if cpu_affinity is None:
        return None
    return os.name != "nt" and hasattr(os, "sched_setaffinity")


def _apply_process_affinity(pid: int, cpu_affinity: list[int] | None) -> None:
    if cpu_affinity is None:
        return
    if os.name == "nt" or not hasattr(os, "sched_setaffinity"):
        return
    os.sched_setaffinity(pid, set(cpu_affinity))


def _write_stream_jsonl(path: Path, events: list[StreamingEventRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def _stream_status(events: list[StreamingEventRecord], *, require_content: bool) -> str:
    if any(event.parse_status == "malformed" for event in events):
        return "malformed"
    if not any(event.event_type == "done" for event in events):
        return "incomplete"
    if require_content and not any(event.valid_generated_content for event in events):
        return "no_content"
    return "parsed"


def _content_from_payload(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    first = choices[0]
    if not isinstance(first, dict):
        return "", None
    finish_reason = first.get("finish_reason")
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content, str(finish_reason) if finish_reason else None
    text = first.get("text")
    if isinstance(text, str) and text:
        return text, str(finish_reason) if finish_reason else None
    return "", str(finish_reason) if finish_reason else None


def _selected_headers(headers: dict[str, str]) -> dict[str, str]:
    keep = {"content-type", "x-request-id", "server"}
    return {key: value for key, value in headers.items() if key.lower() in keep}


def _distribution(name: str, values: list[float], failed_count: int) -> GenerationMetricDistribution:
    if not values:
        return GenerationMetricDistribution(metric_name=name, valid_count=0, failed_count=failed_count)
    ordered = sorted(values)
    median = _percentile(ordered, 50)
    p95 = _percentile(ordered, 95) if len(ordered) >= 5 else None
    return GenerationMetricDistribution(
        metric_name=name,
        valid_count=len(ordered),
        failed_count=failed_count,
        median=round(median, 6),
        p95=round(p95, 6) if p95 is not None else None,
        min=round(ordered[0], 6),
        max=round(ordered[-1], 6),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if os.name == "nt" and process.pid:
        subprocess.run(["taskkill", "/T", "/PID", str(process.pid)], check=False, capture_output=True, text=True)
        return
    if os.name != "nt" and process.pid:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return
        except OSError:
            pass
    process.terminate()


def _kill_process(process: subprocess.Popen[str]) -> None:
    if os.name == "nt" and process.pid:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], check=False, capture_output=True, text=True)
        return
    if os.name != "nt" and process.pid:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    process.kill()


def _wait_for_port_release(host: str, port: int, *, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _port_released(host, port):
            return True
        time.sleep(0.05)
    return _port_released(host, port)


def _port_released(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _result_message(
    benchmark_status: str,
    lifecycle: ServerLifecycleResult,
    failed_requests: list[GenerationRequestResult],
) -> str:
    if benchmark_status == "passed":
        return "server benchmark completed; quality gate not evaluated"
    if lifecycle.readiness_status != "ready":
        return lifecycle.readiness_message or "server was not ready"
    if failed_requests:
        return f"{len(failed_requests)} measured request(s) failed"
    if lifecycle.cleanup_status == "failed":
        return "server cleanup failed"
    return "server benchmark failed"


def _has_fake_server_marker(
    job: LlamaServerJobSpec,
    executable: BenchmarkFileManifest | None,
    inspection: LlamaServerInspection | None,
) -> bool:
    texts = [job.executable.path]
    if executable is not None:
        texts.extend([executable.basename, executable.path])
    if inspection is not None and inspection.fixture_marker_detected:
        return True
    return any(marker in text.lower() for text in texts for marker in FIXTURE_MARKERS)


def _has_test_only_host_override(host_inspection: HostInspectionResult) -> bool:
    host = host_inspection.host
    texts = [
        host.architecture.kernel_release,
        host.cpu_identity.source,
        host.host_capabilities.evidence_source,
        host.topology.source,
        host.os_release,
    ]
    return any("fixture" in text.lower() or "test" in text.lower() for text in texts)


def _extract_supported_flags(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)(?:--[A-Za-z][A-Za-z0-9-]*|-[A-Za-z])", text))


def _extract_commit(text: str) -> str:
    match = re.search(r"(?:commit|build)[:= ]+([0-9a-f]{7,40})", text, flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def _extract_runtime_build(text: str) -> str:
    stripped = " ".join(part.strip() for part in text.splitlines() if part.strip())
    return stripped or "unknown"


def _contains_fixture_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in FIXTURE_MARKERS)


def _is_absolute_target_path(path: str) -> bool:
    return Path(path).is_absolute() or path.startswith("/")


def _is_executable(path: Path) -> bool:
    if os.name == "nt":
        return path.suffix.lower() in {".exe", ".bat", ".cmd", ".com"}
    return os.access(path, os.X_OK)
