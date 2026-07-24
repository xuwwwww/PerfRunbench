from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from autotune.cli import main
from autotune.platform_validation import validate_platform


class PlatformValidationTest(unittest.TestCase):
    def test_quick_cli_writes_sanitized_compatibility_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output_path = Path(temp_dir) / "validation.json"
            stdout = io.StringIO()
            with (
                patch("autotune.platform_validation._probe_torch", return_value={"availability": "unavailable"}),
                patch("autotune.platform_validation._probe_onnxruntime", return_value={"availability": "unavailable"}),
                patch(
                    "autotune.platform_validation._detect_nvidia",
                    return_value={"detected": False, "detection_source": "test", "device_names": []},
                ),
                redirect_stdout(stdout),
            ):
                code = main(["platform", "validate", "--preset", "quick", "--output", str(output_path)])

            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["output"], "validation.json")
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "1.0")
            self.assertEqual(report["preset_budget_minutes"], 5.0)
            self.assertFalse(report["privacy"]["hostname_collected"])
            self.assertFalse(report["privacy"]["machine_identifier_collected"])
            self.assertTrue(all(item["performance_status"] == "unavailable" for item in report["validations"]))
            self.assertNotIn(str(Path.home()), output_path.read_text(encoding="utf-8"))

    def test_cuda_smoke_is_not_run_when_cuda_is_unavailable(self) -> None:
        torch_probe = {
            "availability": "available",
            "version": "test-version",
            "cuda_available": False,
            "cuda_device_count": 0,
        }
        cpu_result = {
            "schema_version": "1.0",
            "runtime": "pytorch",
            "accelerator": "CPU",
            "validation_type": "compatibility_smoke",
            "compatibility_status": "compatibility_verified",
            "performance_status": "unavailable",
            "evidence_origin": "local_bounded_validation",
            "runtime_version": "test-version",
            "details": {},
            "claim_boundary": "Compatibility only; no performance improvement is claimed.",
        }
        with (
            patch("autotune.platform_validation._probe_torch", return_value=torch_probe),
            patch("autotune.platform_validation._torch_cpu_smoke", return_value=cpu_result),
            patch("autotune.platform_validation._torch_cuda_smoke", side_effect=AssertionError("must not run")),
            patch("autotune.platform_validation._probe_onnxruntime", return_value={"availability": "unavailable"}),
            patch(
                "autotune.platform_validation._detect_nvidia",
                return_value={"detected": False, "detection_source": "test", "device_names": []},
            ),
        ):
            report = validate_platform(preset="quick")

        cuda = next(item for item in report["validations"] if item["accelerator"] == "NVIDIA CUDA")
        self.assertEqual(cuda["compatibility_status"], "unavailable")
        self.assertEqual(cuda["performance_status"], "unavailable")

    def test_installed_runtime_failure_marks_report_failed(self) -> None:
        with (
            patch(
                "autotune.platform_validation._probe_torch",
                return_value={"availability": "failed", "error": "broken runtime at C:\\private\\user\\torch.dll"},
            ),
            patch("autotune.platform_validation._probe_onnxruntime", return_value={"availability": "unavailable"}),
            patch(
                "autotune.platform_validation._detect_nvidia",
                return_value={"detected": False, "detection_source": "test", "device_names": []},
            ),
        ):
            report = validate_platform(preset="quick")

        self.assertEqual(report["overall_status"], "failed")
        serialized = json.dumps(report)
        self.assertNotIn("C:\\\\private\\\\user", serialized)


if __name__ == "__main__":
    unittest.main()
