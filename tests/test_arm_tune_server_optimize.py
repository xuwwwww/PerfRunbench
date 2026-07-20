from __future__ import annotations

import io
import json
import os
import socket
import stat
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from autotune.arm_tune.benchmark.artifact_validation import validate_artifact_manifest
from autotune.arm_tune.config import parse_optimize_config
from autotune.arm_tune.schemas import (
    ArchitectureInfo,
    ArmCpuIdentity,
    ArmHostCapabilities,
    CpuTopology,
    HostInfo,
    HostInspectionResult,
    ProductionValidation,
)
from autotune.arm_tune.targets.local import LocalArmTarget
from autotune.cli import main


FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "arm_tune" / "fake_llama_server.py"


class ArmTuneServerOptimizeTest(unittest.TestCase):
    def test_parse_server_optimize_config_accepts_generic_quality_gate_and_server_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, job, search = parse_optimize_config(_optimize_config(root, suite_path=_generic_quality_suite(root)))

        self.assertEqual(job.server.threads, 2)
        self.assertEqual(job.server.ubatch_size, 128)
        self.assertEqual(search.max_candidates, 3)
        self.assertEqual(search.affinity_modes, ["scheduler-default", "compact"])
        self.assertTrue(search.quality_gate_enabled)
        self.assertIsNotNone(search.quality_normalization_policy)
        self.assertTrue(search.quality_normalization_policy["allow_markdown_fence"])

    def test_public_fake_server_default_optimize_does_not_require_quality_gate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, _optimize_config(root, quality_enabled=False))
            output = io.StringIO()
            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "1",
                    },
                ):
                    with redirect_stdout(output):
                        exit_code = main(["arm", "optimize", "--config", str(config_path)])

            result_path = next((root / ".autotuneai" / "runs").glob("*/armtune/server_optimize/server_optimize_result.json"))
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["quality_gate_status"], "not_evaluated")
            self.assertEqual(payload["quality_suite_manifest"]["suite_id"], "not_evaluated")
            self.assertGreaterEqual(payload["eligible_candidate_count"], 1)
            self.assertIn(payload["recommendation"]["status"], {"recommended", "inconclusive"})
            self.assertTrue(all(item["quality_result"]["aggregate"]["quality_gate_status"] == "not_evaluated" for item in payload["candidates"]))
            self.assertIn("quality_suite_id: not_evaluated", output.getvalue())
            self.assertTrue(_candidate_ports_released(payload, root))

    def test_public_fake_server_optional_generic_quality_gate_optimize_e2e(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, _optimize_config(root, suite_path=_generic_quality_suite(root)))
            output = io.StringIO()
            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "1",
                    },
                ):
                    with redirect_stdout(output):
                        exit_code = main(["arm", "optimize", "--config", str(config_path)])

            text = output.getvalue()
            result_paths = list((root / ".autotuneai" / "runs").glob("*/armtune/server_optimize/server_optimize_result.json"))
            self.assertEqual(exit_code, 0)
            self.assertIn("adapter: llama_server", text)
            self.assertIn("quality_suite_id: generic_server_output_v1", text)
            self.assertEqual(len(result_paths), 1)
            payload = json.loads(result_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(payload["result_origin"], "test_fixture")
            self.assertEqual(payload["quality_suite_manifest"]["suite_id"], "generic_server_output_v1")
            self.assertGreaterEqual(payload["eligible_candidate_count"], 1)
            self.assertTrue(all(item["quality_result"]["aggregate"]["quality_gate_status"] == "passed" for item in payload["candidates"]))
            validation = validate_artifact_manifest(result_paths[0].parent)
            self.assertEqual(validation.validation_status, "passed")
            self.assertTrue(_candidate_ports_released(payload, root))

    def test_generic_quality_gate_failure_returns_no_eligible_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = _generic_quality_suite(root, required_text="not-present")
            config_path = _write_config(root, _optimize_config(root, suite_path=suite, max_candidates=1))
            output = io.StringIO()
            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "1",
                    },
                ):
                    with redirect_stdout(output):
                        exit_code = main(["arm", "optimize", "--config", str(config_path)])

            result_path = next((root / ".autotuneai" / "runs").glob("*/armtune/server_optimize/server_optimize_result.json"))
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(payload["recommendation"]["status"], "no_eligible_candidate")
            self.assertEqual(payload["eligible_candidate_count"], 0)
            self.assertIn("recommendation_state: no_eligible_candidate", output.getvalue())


def _optimize_config(root: Path, *, max_candidates: int = 3, suite_path: Path | None = None, quality_enabled: bool = True) -> dict:
    prompt_suite = root / "unused_prompt_suite.json"
    prompt_suite.write_text(json.dumps({"schema_version": "1.0", "prompts": [{"prompt_id": "unused", "prompt": "unused"}]}), encoding="utf-8")
    quality_gate = {"enabled": quality_enabled}
    if suite_path is not None:
        quality_gate["suite_path"] = str(suite_path)
    return {
        "schema_version": "1.0",
        "result_origin": "test_fixture",
        "target": {"type": "local"},
        "runtime": {"adapter": "llama_server"},
        "executable": {"path": str(_fake_executable(root))},
        "model": {"path": str(_model_file(root))},
        "server": {
            "host": "127.0.0.1",
            "port": "auto",
            "startup_timeout_seconds": 2.0,
            "request_timeout_seconds": 2.0,
            "shutdown_timeout_seconds": 0.2,
            "threads": 2,
            "context_size": 256,
            "batch_size": 128,
            "ubatch_size": 128,
            "seed": 42,
            "temperature": 0.0,
        },
        "workload": {
            "mode": "server",
            "prompt_suite_path": str(prompt_suite.resolve()),
            "predict_tokens": 64,
            "repetitions": 1,
            "warmup_requests": 0,
            "streaming": True,
            "cooldown_seconds": 0.0,
        },
        "search": {
            "warmup_runs": 0,
            "search_runs": 1,
            "confirmation_runs": 1,
            "cooldown_seconds": 0.0,
            "random_seed": 42,
            "insert_paired_baseline_every": 0,
            "noise_band_percent": 2.0,
            "max_candidates": max_candidates,
            "affinity_modes": ["scheduler-default", "compact", "compact"],
        },
        "quality_gate": quality_gate,
        "normalization": {
            "schema_version": "1.0",
            "enabled": True,
            "strict": False,
            "allow_markdown_fence": True,
            "allow_prose_outside_fence": False,
        },
        "artifacts": {
            "preserve_server_stdout": True,
            "preserve_server_stderr": True,
            "preserve_request_stream": True,
            "preserve_generated_output": True,
        },
    }


def _generic_quality_suite(root: Path, *, required_text: str = "Hello") -> Path:
    path = root / "generic_server_output_suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "suite_id": "generic_server_output_v1",
                "description": "Generic server response correctness suite.",
                "cases": [
                    {
                        "case_id": "generic_server_output",
                        "category": "generic_output",
                        "prompt": "Return a short greeting.",
                        "response_format": "text",
                        "required": True,
                        "maximum_predicted_tokens": 32,
                        "expected": {},
                        "validators": [
                            {"type": "nonempty_output"},
                            {"type": "required_substrings", "expected": [required_text]},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
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
    path = root / "server_optimize_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path.resolve()


def _inspection() -> HostInspectionResult:
    host = HostInfo(
        architecture=ArchitectureInfo(system="Linux", machine="aarch64", normalized_machine="aarch64", kernel_release="fixture", is_linux=True, is_arm64=True),
        cpu_identity=ArmCpuIdentity(model_name="fixture Neoverse", source="fixture"),
        host_capabilities=ArmHostCapabilities(features=["asimd"], neon=True, evidence_source="fixture"),
        topology=CpuTopology(logical_cpus=2, online_cpus="0-1", cores_per_socket=2, sockets=1, numa_nodes=1, source="fixture"),
    )
    return HostInspectionResult(
        target_type="local",
        host=host,
        production_validation=ProductionValidation(valid=True, reason="fixture", observed_system="Linux", observed_machine="aarch64"),
    )


def _candidate_ports_released(payload: dict, cwd: Path) -> bool:
    for candidate in payload["candidates"]:
        for run in candidate["benchmark_runs"]:
            artifact_dir = Path(run["artifact_directory"])
            if not artifact_dir.is_absolute():
                artifact_dir = cwd / artifact_dir
            lifecycle = artifact_dir / "lifecycle.json"
            if not lifecycle.exists():
                return False
            data = json.loads(lifecycle.read_text(encoding="utf-8"))
            if not data.get("port_released"):
                return False
            port = data.get("selected_port")
            if port:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        sock.bind(("127.0.0.1", int(port)))
                    except OSError:
                        return False
    return True


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
