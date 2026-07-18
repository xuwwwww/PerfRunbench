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


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "arm_tune" / "fake_llama_server.py"
QUALITY_SUITE = ROOT / "configs" / "armtune" / "quality" / "boottrace_suite.json"


class ArmTuneServerOptimizeTest(unittest.TestCase):
    def test_parse_server_optimize_config_accepts_quality_gate_and_server_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _, job, search = parse_optimize_config(_optimize_config(root))

        self.assertEqual(job.server.threads, 2)
        self.assertEqual(job.server.ubatch_size, 128)
        self.assertEqual(search.max_candidates, 3)
        self.assertEqual(search.affinity_modes, ["scheduler-default", "compact"])
        self.assertTrue(search.quality_gate_enabled)
        self.assertEqual(search.quality_suite_path, str(QUALITY_SUITE))

    def test_public_fake_server_quality_aware_optimize_e2e(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, _optimize_config(root))
            output = io.StringIO()
            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_QUALITY_MODE": "boottrace",
                        "FAKE_LLAMA_SERVER_DELAY_PROFILE": "quality",
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "3",
                    },
                ):
                    with redirect_stdout(output):
                        exit_code = main(["arm", "optimize", "--config", str(config_path)])

            text = output.getvalue()
            result_paths = list((root / ".autotuneai" / "runs").glob("*/armtune/server_optimize/server_optimize_result.json"))
            self.assertEqual(exit_code, 0)
            self.assertIn("adapter: llama_server", text)
            self.assertIn("quality_suite_id: synthetic_boottrace_quality_v1", text)
            self.assertIn("recommendation_state: recommended", text)
            self.assertEqual(len(result_paths), 1)
            payload = json.loads(result_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(payload["result_origin"], "test_fixture")
            self.assertEqual(payload["candidate_count"], 3)
            self.assertEqual(payload["eligible_candidate_count"], 2)
            self.assertEqual(payload["recommendation"]["status"], "recommended")
            self.assertEqual(payload["selected_candidate_id"], "threads-1")
            rejected = {item["candidate_id"]: item for item in payload["rejected_candidates"]}
            self.assertIn("server-batch256-ubatch128", rejected)
            self.assertEqual(rejected["server-batch256-ubatch128"]["eligibility_status"], "rejected_quality")
            selected = next(item for item in payload["candidates"] if item["candidate"]["candidate_id"] == "threads-1")
            self.assertEqual(selected["quality_result"]["aggregate"]["quality_gate_status"], "passed")
            self.assertLess(
                selected["distributions"]["ttft_ms"]["median"],
                next(item for item in payload["candidates"] if item["candidate"]["candidate_id"] == "baseline")["distributions"]["ttft_ms"]["median"],
            )
            validation = validate_artifact_manifest(result_paths[0].parent)
            self.assertEqual(validation.validation_status, "passed")
            self.assertTrue(_candidate_ports_released(payload, root))

    def test_quality_gate_failure_returns_no_eligible_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, _optimize_config(root, max_candidates=1))
            output = io.StringIO()
            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_QUALITY_MODE": "boottrace_fail_all",
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "3",
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


def _optimize_config(root: Path, *, max_candidates: int = 3) -> dict:
    prompt_suite = root / "unused_prompt_suite.json"
    prompt_suite.write_text(json.dumps({"schema_version": "1.0", "prompts": [{"prompt_id": "unused", "prompt": "unused"}]}), encoding="utf-8")
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
        "quality_gate": {
            "enabled": True,
            "suite_path": str(QUALITY_SUITE),
        },
        "artifacts": {
            "preserve_server_stdout": True,
            "preserve_server_stderr": True,
            "preserve_request_stream": True,
            "preserve_generated_output": True,
        },
    }


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
