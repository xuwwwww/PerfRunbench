from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from autotune.arm_tune.schemas import ProductionValidation
from autotune.cli import main
from autotune.cli import build_parser


class ArmTuneCliTest(unittest.TestCase):
    def test_arm_doctor_prints_versioned_payload(self) -> None:
        output = io.StringIO()
        with patch("autotune.arm_tune.cli.collect_executor_capabilities", return_value={"recommended_executor": "local"}):
            with redirect_stdout(output):
                code = main(["arm", "doctor"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["kind"], "armtune_doctor")
        self.assertEqual(payload["target"]["type"], "local")
        self.assertIn("controller_status", payload)
        self.assertIn("local_target_status", payload)
        self.assertIn("configured_target_status", payload)
        self.assertIn("production_benchmark_ready", payload)

    def test_arm_doctor_controller_ready_but_local_target_unsupported(self) -> None:
        output = io.StringIO()
        validation = ProductionValidation(
            valid=False,
            reason="production target must be Linux Arm64",
            observed_system="Windows",
            observed_machine="amd64",
        )
        with patch("autotune.arm_tune.cli.collect_executor_capabilities", return_value={"recommended_executor": "local"}):
            with patch("autotune.arm_tune.cli.LocalArmTarget.validate_production_target", return_value=validation):
                with redirect_stdout(output):
                    code = main(["arm", "doctor"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["controller_status"]["status"], "ready")
        self.assertEqual(payload["local_target_status"]["status"], "unsupported")
        self.assertEqual(payload["configured_target_status"]["status"], "not_configured")
        self.assertFalse(payload["production_benchmark_ready"])

    def test_arm_doctor_require_production_returns_nonzero_when_not_ready(self) -> None:
        output = io.StringIO()
        validation = ProductionValidation(
            valid=False,
            reason="production target must be Linux Arm64",
            observed_system="Windows",
            observed_machine="amd64",
        )
        with patch("autotune.arm_tune.cli.collect_executor_capabilities", return_value={"recommended_executor": "local"}):
            with patch("autotune.arm_tune.cli.LocalArmTarget.validate_production_target", return_value=validation):
                with redirect_stdout(output):
                    code = main(["arm", "doctor", "--require-production"])

        self.assertEqual(code, 2)

    def test_arm_doctor_require_production_accepts_native_linux_arm64_local_target(self) -> None:
        output = io.StringIO()
        validation = ProductionValidation(
            valid=True,
            reason="target is Linux Arm64",
            observed_system="Linux",
            observed_machine="aarch64",
        )
        with patch("autotune.arm_tune.cli.collect_executor_capabilities", return_value={"recommended_executor": "local"}):
            with patch("autotune.arm_tune.cli.LocalArmTarget.validate_production_target", return_value=validation):
                with redirect_stdout(output):
                    code = main(["arm", "doctor", "--require-production"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["controller_status"]["status"], "ready")
        self.assertEqual(payload["local_target_status"]["status"], "ready")
        self.assertEqual(payload["configured_target_status"]["status"], "not_configured")
        self.assertTrue(payload["production_benchmark_ready"])

    def test_top_level_arm_parser_preserves_remaining_args(self) -> None:
        args = build_parser().parse_args(["arm", "doctor", "--require-production"])

        self.assertEqual(args.command, "arm")
        self.assertEqual(args.arm_args, ["doctor", "--require-production"])

    def test_top_level_arm_parser_forwards_help(self) -> None:
        args = build_parser().parse_args(["arm", "--help"])

        self.assertEqual(args.command, "arm")
        self.assertTrue(args.arm_help)
        self.assertEqual(args.arm_args, [])

    def test_arm_inspect_host_keeps_workload_evidence_separate(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["arm", "inspect-host"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertIn("host_capabilities", payload["host"])
        self.assertIn("workload_execution_evidence", payload)
        self.assertEqual(payload["workload_execution_evidence"]["status"], "not_collected")

    def test_existing_top_level_help_behavior_remains(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main([])

        self.assertEqual(code, 2)
        self.assertIn("PerfRunbench resource and tuning toolkit", output.getvalue())


if __name__ == "__main__":
    unittest.main()
