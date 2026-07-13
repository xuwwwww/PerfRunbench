from __future__ import annotations

import io
import json
import os
import stat
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.config import parse_benchmark_config
from autotune.arm_tune.schemas import (
    ArchitectureInfo,
    ArmBenchmarkResult,
    ArmCpuIdentity,
    ArmHostCapabilities,
    BenchmarkArtifactManifest,
    BenchmarkFileManifest,
    CommandResult,
    CpuTopology,
    HostInfo,
    LlamaBenchJobSpec,
    ProductionValidation,
    TargetSpec,
    WorkerResult,
)
from autotune.arm_tune.targets.local import LocalArmTarget
from autotune.arm_tune.targets.ssh import SshArmTarget
from autotune.cli import main


FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "arm_tune" / "fake_llama_bench.py"


class FakeRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.commands: list[list[str]] = []

    def run(self, command: list[str], *, timeout_seconds: int) -> CommandResult:
        self.commands.append(command)
        if len(self.commands) == 3:
            return CommandResult(command=command, return_code=0, stdout=self.stdout)
        return CommandResult(command=command, return_code=0)


class ArmTuneBenchmarkTest(unittest.TestCase):
    def test_local_target_vertical_slice_preserves_artifacts_and_metrics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            _, job = parse_benchmark_config(_config(executable, model))
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.result_origin, "test_fixture")
            self.assertEqual(result.validation_status, "passed")
            self.assertEqual(result.execution_status, "completed")
            self.assertEqual(result.parse_status, "parsed")
            self.assertEqual(result.benchmark_status, "passed")
            self.assertTrue((benchmark_dir / "benchmark_job.json").exists())
            self.assertTrue((benchmark_dir / "executable_manifest.json").exists())
            self.assertTrue((benchmark_dir / "model_manifest.json").exists())
            self.assertTrue((benchmark_dir / "raw" / "benchmark_stdout.txt").exists())
            self.assertTrue((benchmark_dir / "raw" / "benchmark_stderr.txt").exists())
            self.assertTrue((benchmark_dir / "benchmark_result.json").exists())
            self.assertIn("benchmark_stdout", result.artifacts.artifacts)

    def test_expected_hash_mismatch_does_not_execute_benchmark(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            config = _config(executable, model)
            config["executable"]["expected_sha256"] = "0" * 64
            _, job = parse_benchmark_config(config)
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.validation_status, "failed")
            self.assertEqual(result.execution_status, "not_run")
            self.assertEqual(result.benchmark_status, "failed")
            self.assertTrue((benchmark_dir / "executable_manifest.json").exists())
            self.assertFalse((benchmark_dir / "raw" / "benchmark_stdout.txt").exists())

    def test_missing_model_is_validation_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            missing_model = root / "missing.gguf"
            _, job = parse_benchmark_config(_config(executable, missing_model))
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.validation_status, "failed")
            self.assertIn("file does not exist", result.message)

    def test_non_executable_binary_is_validation_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = root / "not_executable.txt"
            executable.write_text("not executable", encoding="utf-8")
            model = _model_file(root)
            _, job = parse_benchmark_config(_config(executable, model))
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.validation_status, "failed")
            self.assertIn("not executable", result.message)

    def test_unsupported_required_flag_fails_before_execution(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            _, job = parse_benchmark_config(_config(executable, model))
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_HELP_MODE": "missing_generation"}):
                result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.validation_status, "failed")
            self.assertEqual(result.execution_status, "not_run")
            self.assertIn("generation_tokens", result.message)

    def test_unsupported_configured_optional_flag_fails_before_execution(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            config = _config(executable, model)
            config["benchmark"]["additional_supported_options"] = {"batch_size": 8}
            _, job = parse_benchmark_config(config)
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.validation_status, "failed")
            self.assertEqual(result.execution_status, "not_run")
            self.assertIn("batch_size", result.message)
            self.assertFalse((benchmark_dir / "raw" / "benchmark_stdout.txt").exists())

    def test_command_arguments_are_constructed_without_shell_string(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            args_path = root / "args.json"
            _, job = parse_benchmark_config(_config(executable, model))
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_ARGS_PATH": str(args_path)}):
                result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            args = json.loads(args_path.read_text(encoding="utf-8"))
            self.assertEqual(result.benchmark_status, "passed")
            self.assertIn("-m", args)
            self.assertIn(str(model), args)
            self.assertIn("-t", args)
            self.assertIn("1", args)

    def test_timeout_preserves_partial_raw_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            config = _config(executable, model)
            config["benchmark"]["timeout_seconds"] = 1
            _, job = parse_benchmark_config(config)
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_MODE": "timeout"}):
                result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.execution_status, "timeout")
            self.assertEqual(result.benchmark_status, "failed")
            self.assertTrue((benchmark_dir / "raw" / "benchmark_stdout.txt").exists())
            self.assertTrue((benchmark_dir / "raw" / "benchmark_stderr.txt").exists())

    def test_nonzero_process_exit_is_failed_benchmark(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            _, job = parse_benchmark_config(_config(executable, model))
            benchmark_dir = root / "run" / "armtune" / "benchmark"
            job = replace(job, run_id="fixture-run", target_run_directory=str(benchmark_dir))

            with patch.dict(os.environ, {"FAKE_LLAMA_BENCH_MODE": "nonzero"}):
                result = LocalArmTarget().run_benchmark(job, benchmark_dir)

            self.assertEqual(result.execution_status, "failed")
            self.assertEqual(result.benchmark_status, "failed")
            self.assertEqual(result.trial.invocation.exit_code, 42)

    def test_cli_exit_codes_for_success_missing_metrics_and_validation_failure(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            executable = _fake_executable(root)
            model = _model_file(root)
            config_path = _write_config(root, _config(executable, model))
            with _cwd(root):
                output = io.StringIO()
                with redirect_stdout(output):
                    success_code = main(["arm", "benchmark", "--config", str(config_path)])

            no_metrics_config = _write_config(root, _config(executable, model), name="no_metrics.json")
            with _cwd(root), patch.dict(os.environ, {"FAKE_LLAMA_BENCH_MODE": "no_metrics"}):
                output = io.StringIO()
                with redirect_stdout(output):
                    missing_metrics_code = main(["arm", "benchmark", "--config", str(no_metrics_config)])

            mismatch = _config(executable, model)
            mismatch["model"]["expected_sha256"] = "0" * 64
            mismatch_config = _write_config(root, mismatch, name="mismatch.json")
            with _cwd(root):
                output = io.StringIO()
                with redirect_stdout(output):
                    validation_code = main(["arm", "benchmark", "--config", str(mismatch_config)])

        self.assertEqual(success_code, 0)
        self.assertEqual(missing_metrics_code, 1)
        self.assertEqual(validation_code, 2)

    def test_ssh_target_benchmark_uses_worker_protocol_and_downloads_artifacts(self) -> None:
        result = _benchmark_result()
        payload = WorkerResult(operation="benchmark.llama_bench", status="ok", result=result.to_dict()).to_dict()
        runner = FakeRunner(json.dumps(payload))
        spec = TargetSpec(type="ssh", host="arm-host")
        _, job = parse_benchmark_config(_config("/opt/llama-bench", "/models/fake.gguf", target={"type": "ssh", "host": "arm-host"}))
        job = replace(job, run_id="fixture-run", target_run_directory="~/armtune-runs/fixture-run/armtune/benchmark")

        returned = SshArmTarget(spec, runner=runner).run_benchmark(job, Path("local-artifacts"))

        self.assertEqual(returned.run_id, "fixture-run")
        self.assertEqual(len(runner.commands), 4)
        self.assertEqual(runner.commands[0][-3:], ["mkdir", "-p", "~/armtune-runs/fixture-run/armtune/benchmark"])
        self.assertIn("scp", runner.commands[1][0])
        self.assertIn("-m", runner.commands[2])
        self.assertIn("autotune.cli", runner.commands[2])
        self.assertIn("-r", runner.commands[3])


def _config(executable_path: str | Path, model_path: str | Path, *, target: dict | None = None) -> dict:
    return {
        "schema_version": "1.0",
        "result_origin": "test_fixture",
        "target": target or {"type": "local"},
        "executable": {"path": str(executable_path)},
        "model": {"path": str(model_path)},
        "benchmark": {
            "threads": 1,
            "prompt_tokens": 16,
            "generation_tokens": 8,
            "repetitions": 1,
            "timeout_seconds": 5,
        },
        "artifacts": {"preserve_raw_stdout": True, "preserve_raw_stderr": True},
    }


def _fake_executable(root: Path) -> Path:
    if os.name == "nt":
        wrapper = root / "fake_llama_bench.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{FIXTURE_SCRIPT}" %*\r\n', encoding="utf-8")
    else:
        wrapper = root / "fake_llama_bench"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FIXTURE_SCRIPT}" "$@"\n', encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper.resolve()


def _model_file(root: Path) -> Path:
    path = root / "fixture-model.gguf"
    path.write_bytes(b"fixture model, not real weights\n")
    return path.resolve()


def _write_config(root: Path, payload: dict, *, name: str = "benchmark.json") -> Path:
    path = root / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path.resolve()


def _benchmark_result() -> ArmBenchmarkResult:
    host = HostInfo(
        architecture=ArchitectureInfo(system="Linux", machine="aarch64", normalized_machine="aarch64", kernel_release="fixture", is_linux=True, is_arm64=True),
        cpu_identity=ArmCpuIdentity(model_name="fixture Neoverse", source="fixture"),
        host_capabilities=ArmHostCapabilities(features=["asimd"], neon=True, evidence_source="fixture"),
        topology=CpuTopology(logical_cpus=1, source="fixture"),
    )
    return ArmBenchmarkResult(
        run_id="fixture-run",
        operation="benchmark.llama_bench",
        target_type="ssh",
        result_origin="test_fixture",
        validation_status="passed",
        execution_status="completed",
        parse_status="parsed",
        benchmark_status="passed",
        host=host,
        executable=BenchmarkFileManifest(path="/opt/llama-bench", basename="llama-bench", size_bytes=1, sha256=sha256_file(FIXTURE_SCRIPT)),
        model=BenchmarkFileManifest(path="/models/fake.gguf", basename="fake.gguf", size_bytes=1, sha256="f" * 64),
        artifacts=BenchmarkArtifactManifest(artifacts={}),
        artifact_directory="~/armtune-runs/fixture-run/armtune/benchmark",
        message="fixture result",
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
