from __future__ import annotations

import io
import json
import os
import socket
import stat
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autotune.arm_tune.benchmark.artifact_validation import validate_artifact_manifest
from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.benchmark.llama_server import SERVER_BENCHMARK_OPERATION, decide_server_provenance
from autotune.arm_tune.config import parse_benchmark_config
from autotune.arm_tune.errors import ArmTuneConfigError, ArmTuneProtocolError
from autotune.arm_tune.schemas import (
    ArchitectureInfo,
    ArmCpuIdentity,
    ArmHostCapabilities,
    BenchmarkArtifactManifest,
    BenchmarkFileManifest,
    CommandResult,
    CpuTopology,
    HostInfo,
    HostInspectionResult,
    LlamaServerBenchmarkResult,
    LlamaServerJobSpec,
    ProductionValidation,
    ServerLifecycleResult,
    TargetSpec,
    WorkerResult,
)
from autotune.arm_tune.targets.local import LocalArmTarget
from autotune.arm_tune.targets.protocol import validate_worker_job
from autotune.arm_tune.targets.ssh import SshArmTarget
from autotune.cli import main


FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "arm_tune" / "fake_llama_server.py"


class FakeRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.commands: list[list[str]] = []

    def run(self, command: list[str], *, timeout_seconds: int) -> CommandResult:
        self.commands.append(command)
        if len(self.commands) == 3:
            return CommandResult(command=command, return_code=0, stdout=self.stdout)
        return CommandResult(command=command, return_code=0)


class ArmTuneLlamaServerTest(unittest.TestCase):
    def test_server_config_rejects_public_bind_address(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(_fake_executable(root), _model_file(root), _prompt_suite(root))
            config["server"]["host"] = "0.0.0.0"

            with self.assertRaisesRegex(ArmTuneConfigError, "loopback-only"):
                parse_benchmark_config(config)

    def test_worker_protocol_rejects_server_protocol_mismatch_and_public_bind(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, job = parse_benchmark_config(_config(_fake_executable(root), _model_file(root), _prompt_suite(root)))
            payload = job.to_dict()
            payload["protocol_version"] = "9.9"

            with self.assertRaisesRegex(ArmTuneProtocolError, "protocol_version"):
                validate_worker_job(payload)

            public_payload = job.to_dict()
            public_payload["server"]["host"] = "0.0.0.0"
            with self.assertRaisesRegex(ArmTuneProtocolError, "loopback-only"):
                validate_worker_job(public_payload)

    def test_hash_mismatch_does_not_start_server(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _config(_fake_executable(root), _model_file(root), _prompt_suite(root))
            config["executable"]["expected_sha256"] = "0" * 64
            _, job = parse_benchmark_config(config)
            result = LocalArmTarget().run_server_benchmark(
                replace(job, run_id="server-hash-mismatch"),
                root / "run" / "armtune" / "server_benchmark",
            )

            self.assertEqual(result.validation_status, "failed")
            self.assertEqual(result.startup_status, "not_started")
            self.assertFalse((Path(result.artifact_directory) / "raw" / "server_stdout.txt").exists())

    def test_missing_required_server_flag_fails_before_startup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, job = parse_benchmark_config(_config(_fake_executable(root), _model_file(root), _prompt_suite(root)))

            with patch.dict(os.environ, {"FAKE_LLAMA_SERVER_HELP_MODE": "missing_port"}):
                result = LocalArmTarget().run_server_benchmark(
                    replace(job, run_id="server-missing-flag"),
                    root / "run" / "armtune" / "server_benchmark",
                )

            self.assertEqual(result.validation_status, "failed")
            self.assertEqual(result.inspection_status, "passed")
            self.assertEqual(result.startup_status, "not_started")
            self.assertIn("port", result.message)

    def test_fake_server_e2e_records_streaming_metrics_and_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args_path = root / "args.json"
            _, job = parse_benchmark_config(
                _config(_fake_executable(root), _model_file(root), _prompt_suite(root, prompt_count=3))
            )

            with patch.dict(os.environ, {"FAKE_LLAMA_SERVER_ARGS_PATH": str(args_path)}):
                result = LocalArmTarget().run_server_benchmark(
                    replace(job, run_id="server-e2e"),
                    root / "run" / "armtune" / "server_benchmark",
                )

            measured = [request for request in result.requests if request.phase == "measured"]
            first_stream = _read_jsonl(root / "run" / "armtune" / "server_benchmark" / "requests" / "short_completion" / "measured" / "0001" / "stream.jsonl")

            self.assertEqual(result.result_origin, "test_fixture")
            self.assertEqual(result.benchmark_status, "passed")
            self.assertEqual(result.quality_gate_status, "not_evaluated")
            self.assertEqual(result.startup_status, "ready")
            self.assertEqual(result.readiness_status, "ready")
            self.assertIn(result.cleanup_status, {"passed", "forced"})
            self.assertTrue(result.lifecycle.port_released)
            self.assertEqual(len(measured), 6)
            self.assertTrue(all(request.ttft_ms is not None for request in measured))
            self.assertTrue(all(request.end_to_end_latency_ms is not None for request in measured))
            self.assertEqual(measured[0].generated_text, "Hello world")
            self.assertGreater(measured[0].ttft_ms, 0)
            self.assertEqual(first_stream[0]["event_type"], "metadata")
            first_valid_index = next(index for index, event in enumerate(first_stream) if event["valid_generated_content"])
            self.assertGreater(first_valid_index, 0)
            self.assertTrue(all(not event["valid_generated_content"] for event in first_stream[:first_valid_index]))
            self.assertEqual(first_stream[first_valid_index]["event_type"], "content")
            self.assertTrue(args_path.exists())
            self.assertIn("--host", json.loads(args_path.read_text(encoding="utf-8")))
            validation = validate_artifact_manifest(root / "run" / "armtune" / "server_benchmark")
            self.assertEqual(validation.validation_status, "passed")

    def test_startup_exit_and_timeout_are_distinct(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, job = parse_benchmark_config(_config(_fake_executable(root), _model_file(root), _prompt_suite(root)))

            with patch.dict(os.environ, {"FAKE_LLAMA_SERVER_MODE": "exit_before_ready"}):
                exited = LocalArmTarget().run_server_benchmark(
                    replace(job, run_id="server-exit-before-ready"),
                    root / "exit" / "armtune" / "server_benchmark",
                )

            timeout_job = replace(job, run_id="server-startup-timeout", server=replace(job.server, startup_timeout_seconds=0.3))
            with patch.dict(os.environ, {"FAKE_LLAMA_SERVER_MODE": "startup_timeout"}):
                timed_out = LocalArmTarget().run_server_benchmark(
                    timeout_job,
                    root / "timeout" / "armtune" / "server_benchmark",
                )

            self.assertEqual(exited.startup_status, "exited_before_ready")
            self.assertEqual(exited.request_status, "not_run")
            self.assertEqual(timed_out.startup_status, "startup_timeout")
            self.assertEqual(timed_out.readiness_status, "timeout")
            self.assertEqual(timed_out.request_status, "not_run")

    def test_request_failures_are_preserved_without_zero_latency(self) -> None:
        cases = {
            "http_error": "http_error",
            "malformed_sse": "malformed",
            "mid_stream_disconnect": "incomplete",
            "empty_content": "no_content",
            "request_timeout": "timeout",
        }
        for mode, expected_stream_status in cases.items():
            with self.subTest(mode=mode), TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                _, job = parse_benchmark_config(
                    _config(
                        _fake_executable(root),
                        _model_file(root),
                        _prompt_suite(root),
                        request_timeout_seconds=0.25,
                    )
                )

                with patch.dict(os.environ, {"FAKE_LLAMA_SERVER_MODE": mode}):
                    result = LocalArmTarget().run_server_benchmark(
                        replace(job, run_id=f"server-{mode}"),
                        root / "run" / "armtune" / "server_benchmark",
                    )

                measured = [request for request in result.requests if request.phase == "measured"]
                self.assertEqual(result.benchmark_status, "failed")
                self.assertTrue(measured)
                self.assertEqual(measured[0].stream_parse_status, expected_stream_status)
                if expected_stream_status in {"http_error", "timeout", "no_content", "malformed"}:
                    self.assertIsNone(measured[0].ttft_ms)

    def test_measured_origin_cannot_be_forced_without_production_host(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, job = parse_benchmark_config(_config(_fake_executable(root), _model_file(root), _prompt_suite(root), result_origin="measured"))
            realistic_job = replace(job, executable=replace(job.executable, path="/opt/llama-server"))
            provenance = decide_server_provenance(realistic_job, _inspection(valid=False, system="Linux", machine="x86_64"))

            self.assertEqual(provenance.result_origin, "unverified")

    def test_ssh_server_benchmark_uses_worker_protocol_shape(self) -> None:
        result = _server_result()
        payload = WorkerResult(operation=SERVER_BENCHMARK_OPERATION, status="ok", result=result.to_dict()).to_dict()
        runner = FakeRunner(json.dumps(payload))
        spec = TargetSpec(type="ssh", host="arm-host")
        _, job = parse_benchmark_config(
            _config("/opt/llama-server", "/models/fake.gguf", "/prompts/suite.json", target={"type": "ssh", "host": "arm-host"})
        )
        job = replace(job, run_id="fixture-run", target_run_directory="~/armtune-runs/fixture-run/armtune/server_benchmark")

        with TemporaryDirectory() as temp_dir:
            returned = SshArmTarget(spec, runner=runner).run_server_benchmark(job, Path(temp_dir) / "local-server-artifacts")

        self.assertEqual(returned.run_id, "fixture-run")
        self.assertEqual(len(runner.commands), 4)
        self.assertEqual(runner.commands[0][-3:], ["mkdir", "-p", "~/armtune-runs/fixture-run/armtune/server_benchmark"])
        self.assertIn("scp", runner.commands[1][0])
        self.assertIn("worker", runner.commands[2])
        self.assertIn("~/armtune-runs/fixture-run/armtune/server_benchmark/server_job.json", runner.commands[2])
        self.assertIn("-r", runner.commands[3])

    def test_public_cli_fake_server_e2e_summary(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, _config(_fake_executable(root), _model_file(root), _prompt_suite(root)))

            with _cwd(root):
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["arm", "benchmark", "--config", str(config_path)])

            text = output.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertIn("server_startup_status: ready", text)
            self.assertIn("quality_gate_status: not_evaluated", text)
            self.assertIn("median_ttft_ms:", text)
            self.assertIn("median_end_to_end_latency_ms:", text)

    def test_port_released_after_cleanup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, job = parse_benchmark_config(_config(_fake_executable(root), _model_file(root), _prompt_suite(root)))

            result = LocalArmTarget().run_server_benchmark(
                replace(job, run_id="server-port-release"),
                root / "run" / "armtune" / "server_benchmark",
            )

            self.assertTrue(result.lifecycle.port_released)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", result.lifecycle.selected_port))


def _config(
    executable_path: str | Path,
    model_path: str | Path,
    prompt_suite_path: str | Path,
    *,
    target: dict | None = None,
    result_origin: str = "test_fixture",
    request_timeout_seconds: float = 2.0,
) -> dict:
    return {
        "schema_version": "1.0",
        "result_origin": result_origin,
        "target": target or {"type": "local"},
        "runtime": {"adapter": "llama_server"},
        "executable": {"path": str(executable_path)},
        "model": {"path": str(model_path)},
        "server": {
            "host": "127.0.0.1",
            "port": "auto",
            "startup_timeout_seconds": 2.0,
            "request_timeout_seconds": request_timeout_seconds,
            "shutdown_timeout_seconds": 1.0,
            "threads": 1,
            "context_size": 256,
            "batch_size": 32,
            "seed": 42,
            "temperature": 0.0,
        },
        "workload": {
            "mode": "server",
            "prompt_suite_path": str(prompt_suite_path),
            "predict_tokens": 32,
            "repetitions": 2,
            "warmup_requests": 1,
            "streaming": True,
            "cooldown_seconds": 0.0,
        },
        "artifacts": {
            "preserve_server_stdout": True,
            "preserve_server_stderr": True,
            "preserve_request_stream": True,
            "preserve_generated_output": True,
        },
    }


def _prompt_suite(root: Path, *, prompt_count: int = 1) -> Path:
    prompts = [
        {
            "prompt_id": "short_completion",
            "prompt": "Complete this sentence: ArmTune measures",
            "expected_response_mode": "text",
            "max_predict_tokens": 16,
            "deterministic_settings": {"temperature": 0.0, "seed": 42},
        },
        {
            "prompt_id": "structured_json",
            "prompt": "Return a JSON object with status and note.",
            "expected_response_mode": "json_requested_not_scored",
            "max_predict_tokens": 24,
            "deterministic_settings": {"temperature": 0.0, "seed": 42},
        },
        {
            "prompt_id": "short_summary",
            "prompt": "Summarize this synthetic event: startup ready, stream parsed, cleanup passed.",
            "expected_response_mode": "summary_not_scored",
            "max_predict_tokens": 24,
            "deterministic_settings": {"temperature": 0.0, "seed": 42},
        },
    ][:prompt_count]
    path = root / "prompt_suite.json"
    path.write_text(json.dumps({"schema_version": "1.0", "prompts": prompts}, indent=2), encoding="utf-8")
    return path.resolve()


def _fake_executable(root: Path) -> Path:
    if os.name == "nt":
        wrapper = root / "fake_llama_server.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{FIXTURE_SCRIPT}" %*\r\n', encoding="utf-8")
    else:
        wrapper = root / "fake_llama_server"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FIXTURE_SCRIPT}" "$@"\n', encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper.resolve()


def _model_file(root: Path) -> Path:
    path = root / "fixture-model.gguf"
    path.write_bytes(b"fixture model, not real weights\n")
    return path.resolve()


def _write_config(root: Path, payload: dict) -> Path:
    path = root / "server_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path.resolve()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _server_result() -> LlamaServerBenchmarkResult:
    host = HostInfo(
        architecture=ArchitectureInfo(system="Linux", machine="aarch64", normalized_machine="aarch64", kernel_release="fixture", is_linux=True, is_arm64=True),
        cpu_identity=ArmCpuIdentity(model_name="fixture Neoverse", source="fixture"),
        host_capabilities=ArmHostCapabilities(features=["asimd"], neon=True, evidence_source="fixture"),
        topology=CpuTopology(logical_cpus=1, source="fixture"),
    )
    return LlamaServerBenchmarkResult(
        run_id="fixture-run",
        operation=SERVER_BENCHMARK_OPERATION,
        target_type="ssh",
        result_origin="test_fixture",
        validation_status="passed",
        inspection_status="passed",
        startup_status="ready",
        readiness_status="ready",
        request_status="completed",
        stream_parse_status="parsed",
        cleanup_status="passed",
        benchmark_status="passed",
        quality_gate_status="not_evaluated",
        host=host,
        executable=BenchmarkFileManifest(path="/opt/llama-server", basename="llama-server", size_bytes=1, sha256=sha256_file(FIXTURE_SCRIPT)),
        model=BenchmarkFileManifest(path="/models/fake.gguf", basename="fake.gguf", size_bytes=1, sha256="f" * 64),
        lifecycle=ServerLifecycleResult(process_started=True, startup_status="ready", readiness_status="ready", cleanup_status="passed", port_released=True, selected_port=12345),
        artifacts=BenchmarkArtifactManifest(artifacts={}),
        artifact_directory="~/armtune-runs/fixture-run/armtune/server_benchmark",
        message="fixture server result",
    )


def _inspection(*, valid: bool, system: str, machine: str) -> HostInspectionResult:
    host = HostInfo(
        architecture=ArchitectureInfo(
            system=system,
            machine=machine,
            normalized_machine=machine,
            kernel_release="6.8.0",
            is_linux=system == "Linux",
            is_arm64=machine == "aarch64",
        ),
        cpu_identity=ArmCpuIdentity(model_name="Neoverse", source="manual"),
        host_capabilities=ArmHostCapabilities(evidence_source="manual"),
        topology=CpuTopology(logical_cpus=1, source="manual"),
    )
    return HostInspectionResult(
        target_type="local",
        host=host,
        production_validation=ProductionValidation(valid=valid, reason="fixture", observed_system=system, observed_machine=machine),
    )


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
