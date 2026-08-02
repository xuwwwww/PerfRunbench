from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
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
        self.assertIn("PerfRunbench reproducible benchmarking", output.getvalue())

    def test_release_arm_commands_are_registered_without_removed_calibration_command(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["arm", "--help"])

        self.assertEqual(code, 0)
        text = output.getvalue()
        for command in [
            "init",
            "guide",
            "explain",
            "build-doctor",
            "build-plan",
            "build-profile",
            "tune-build",
            "inspect-build",
            "profile-compare",
            "compare-builds",
            "passport",
            "verify-passport",
        ]:
            self.assertIn(command, text)
        self.assertNotIn("quality-calibrate", text)

    def test_arm_init_creates_versioned_guarded_starter_bundle(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "armtune"
            with redirect_stdout(output):
                code = main(
                    [
                        "arm",
                        "init",
                        "--output",
                        str(project),
                        "--target",
                        "ssh",
                        "--workload",
                        "llama-bench",
                        "--preset",
                        "quick",
                        "--json",
                    ]
                )

            result = json.loads(output.getvalue())
            manifest = json.loads((project / "armtune-project.json").read_text(encoding="utf-8"))
            optimize = json.loads((project / "optimize.json").read_text(encoding="utf-8"))

            self.assertEqual(code, 0)
            self.assertEqual(result["schema_version"], "1.0")
            self.assertEqual(manifest["kind"], "armtune_starter_project")
            self.assertEqual(manifest["result_origin"], "unverified")
            self.assertEqual(optimize["result_origin"], "unverified")
            self.assertEqual(optimize["search"]["max_candidates"], 4)
            self.assertEqual(optimize["search"]["noise_band_percent"], 5.0)
            self.assertTrue(optimize["target"]["strict_host_key_checking"])

    def test_arm_guide_reports_placeholders_without_target_access(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            project = Path(temp_dir) / "armtune"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["arm", "init", "--output", str(project)]), 0)
            with redirect_stdout(output):
                code = main(["arm", "guide", "--project", str(project), "--json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["readiness"], "needs_configuration")
        self.assertGreater(len(payload["placeholder_fields"]), 0)
        self.assertIn("--require-production", payload["workflow"][0]["command"])

    def test_arm_explain_makes_decode_heavy_human_readable(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["arm", "explain", "decode-heavy", "--json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(payload["kind"], "armtune_concepts")
        self.assertIn("longer response", payload["concepts"]["decode-heavy"]["meaning"])
        self.assertIn("latency regression guards", payload["concepts"]["decode-heavy"]["primary_metric"])

    def test_arm_doctor_and_inspect_host_offer_readable_summaries(self) -> None:
        doctor_output = io.StringIO()
        inspect_output = io.StringIO()
        with patch("autotune.arm_tune.cli.collect_executor_capabilities", return_value={"recommended_executor": "local"}):
            with redirect_stdout(doctor_output):
                doctor_code = main(["arm", "doctor", "--summary"])
        with redirect_stdout(inspect_output):
            inspect_code = main(["arm", "inspect-host", "--summary"])

        self.assertEqual(doctor_code, 0)
        self.assertEqual(inspect_code, 0)
        self.assertIn("production_benchmark_ready:", doctor_output.getvalue())
        self.assertIn("workload_execution_evidence: not_collected", inspect_output.getvalue())


if __name__ == "__main__":
    unittest.main()
