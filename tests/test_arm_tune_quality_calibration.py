from __future__ import annotations

import io
import json
import os
import stat
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
CALIBRATION_SUITE = ROOT / "configs" / "armtune" / "quality" / "boottrace_calibration_suite.json"


class ArmTuneQualityCalibrationTest(unittest.TestCase):
    def test_public_cli_fenced_output_is_unsuitable_in_strict_mode(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(root, _calibration_config(root, normalization={"enabled": False, "strict": True}))
            output = io.StringIO()

            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_QUALITY_MODE": "boottrace_fenced",
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "5",
                    },
                ):
                    with redirect_stdout(output):
                        exit_code = main(["arm", "quality-calibrate", "--config", str(config_path)])

            result_path = next((root / ".autotuneai" / "runs").glob("*/armtune/quality_calibration/quality_calibration_result.json"))
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(payload["suitability_state"], "unsuitable")
            self.assertEqual(payload["quality_gate_status"], "failed")
            self.assertEqual(payload["artifact_validation"]["validation_status"], "passed")
            self.assertIn("markdown_fence_present", payload["failure_category_histogram"])
            self.assertIn("model_profile_suitability: unsuitable", output.getvalue())

    def test_public_cli_normalization_can_calibrate_fenced_json_as_suitable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(
                root,
                _calibration_config(
                    root,
                    normalization={
                        "schema_version": "1.0",
                        "enabled": True,
                        "strict": False,
                        "allow_markdown_fence": True,
                        "allow_prose_outside_fence": False,
                    },
                ),
            )
            output = io.StringIO()

            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_QUALITY_MODE": "boottrace_fenced",
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "5",
                    },
                ):
                    with redirect_stdout(output):
                        exit_code = main(["arm", "quality-calibrate", "--config", str(config_path)])

            result_path = next((root / ".autotuneai" / "runs").glob("*/armtune/quality_calibration/quality_calibration_result.json"))
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            case = payload["quality_result"]["cases"][0]
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["suitability_state"], "suitable")
            self.assertEqual(payload["quality_gate_status"], "passed")
            self.assertEqual(payload["artifact_validation"]["validation_status"], "passed")
            self.assertEqual(payload["required_quality_cases_passed"], 5)
            self.assertEqual(case["validation_input"], "normalized")
            self.assertTrue(case["normalization_applied"])
            self.assertEqual(payload["structured_capability"]["seed"], "supported")
            self.assertEqual(payload["structured_capability"]["temperature"], "supported")
            self.assertIn("model_profile_suitability: suitable", output.getvalue())

    def test_prompt_contract_version_is_recorded_and_used_for_requests(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_config(
                root,
                _calibration_config(
                    root,
                    normalization={"enabled": False, "strict": True},
                    quality_gate={
                        "prompt_contract_version": "taxonomy-rules-and-disjoint-few-shot",
                        "calibration_suite_path": str(CALIBRATION_SUITE),
                    },
                ),
            )

            with _cwd(root), patch.object(LocalArmTarget, "inspect", return_value=_inspection()):
                with patch.dict(
                    os.environ,
                    {
                        "FAKE_LLAMA_SERVER_QUALITY_MODE": "boottrace_success",
                        "FAKE_LLAMA_SERVER_EVENT_DELAY": "0.001",
                        "FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS": "5",
                    },
                ):
                    exit_code = main(["arm", "quality-calibrate", "--config", str(config_path)])

            result_dir = next((root / ".autotuneai" / "runs").glob("*/armtune/quality_calibration"))
            prompt_contract = json.loads((result_dir / "prompt_contract.json").read_text(encoding="utf-8"))
            prompt_suite = json.loads((result_dir / "prompt_suite_snapshot.json").read_text(encoding="utf-8"))
            effective_config = json.loads((result_dir / "effective_config.json").read_text(encoding="utf-8"))
            request = json.loads(
                next((result_dir / "benchmark" / "requests").glob("boot_success/measured/0001/request.json")).read_text(encoding="utf-8")
            )
            rendered_prompt = request["request"]["messages"][0]["content"]

            self.assertEqual(exit_code, 0)
            self.assertEqual(prompt_contract["contract_version"], "taxonomy-rules-and-disjoint-few-shot")
            self.assertEqual(prompt_suite["prompt_contract_version"], "taxonomy-rules-and-disjoint-few-shot")
            self.assertEqual(effective_config["quality_gate"]["prompt_contract_version"], "taxonomy-rules-and-disjoint-few-shot")
            self.assertIn("Disjoint calibration examples", rendered_prompt)
            self.assertIn("dev_no_boot_device_failure", rendered_prompt)
            self.assertTrue((result_dir / "calibration_suite_manifest.json").exists())


def _calibration_config(root: Path, *, normalization: dict, quality_gate: dict | None = None) -> dict:
    prompt_suite = root / "unused_prompt_suite.json"
    prompt_suite.write_text(json.dumps({"schema_version": "1.0", "prompts": [{"prompt_id": "unused", "prompt": "unused"}]}), encoding="utf-8")
    quality_gate_payload = {
        "enabled": True,
        "suite_path": str(QUALITY_SUITE),
    }
    if quality_gate:
        quality_gate_payload.update(quality_gate)
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
            "predict_tokens": 96,
            "repetitions": 1,
            "warmup_requests": 0,
            "streaming": True,
            "cooldown_seconds": 0.0,
        },
        "quality_gate": quality_gate_payload,
        "normalization": normalization,
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
    path = root / "quality_calibration_config.json"
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
