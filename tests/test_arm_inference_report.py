from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from autotune.cli import main
from autotune.report.arm_inference_report import generate_arm_inference_report


class ArmInferenceReportTest(unittest.TestCase):
    def test_generates_protocol_separated_visual_evidence(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            source = root / "summary.json"
            source.write_text(json.dumps(_result()), encoding="utf-8")

            outputs = generate_arm_inference_report(source, root / "visuals")

            for path in outputs.values():
                self.assertTrue(Path(path).exists())
            summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "1.0")
            self.assertEqual(summary["result_origin"], "measured")
            self.assertEqual(len(summary["thread_scaling"]), 4)
            self.assertEqual(len(summary["serving_comparisons"]), 2)
            self.assertEqual(
                summary["screening_rejections"],
                [
                    {
                        "client_limited": True,
                        "concurrency": 8,
                        "failure_count": 12,
                        "profile_id": "slots8-cont-on",
                        "quality_failures": 12,
                        "workload_id": "prefill-heavy",
                    }
                ],
            )
            html = Path(outputs["report"]).read_text(encoding="utf-8")
            self.assertIn("Realistic Arm inference tuning", html)
            self.assertIn("Hardware scaling percentages and runtime-tuning percentages", html)
            self.assertIn("<code>summary.json</code>", html)
            self.assertIn("Failed and client-limited measurements remain visible", html)
            self.assertIn("eligible preliminary", html)
            self.assertNotIn(">eligible_preliminary<", html)
            self.assertNotIn(str(root), html)
            charts = Path(outputs["charts"])
            self.assertIn("llama-bench thread scaling", (charts / "thread_scaling.svg").read_text(encoding="utf-8"))
            self.assertIn("5% noise guard", (charts / "paired_confirmation.svg").read_text(encoding="utf-8"))

    def test_rejects_unrelated_result_kind(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            source = Path(temp_dir) / "result.json"
            source.write_text(json.dumps({"schema_version": "1.0", "kind": "other"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_arm_inference_report(source)

    def test_cli_exports_visual_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            source = root / "summary.json"
            source.write_text(json.dumps(_result()), encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "report-arm-inference",
                            "--input",
                            str(source),
                            "--output-dir",
                            str(root / "visuals"),
                        ]
                    ),
                    0,
                )
            payload = json.loads(stdout.getvalue())
            self.assertTrue(Path(payload["report"]).exists())


def _comparison(workload: str, baseline: float, candidate: float) -> dict:
    improvement = (candidate / baseline - 1.0) * 100.0
    return {
        "schema_version": "1.0",
        "workload_id": workload,
        "pair_count": 3,
        "baseline_output_tokens_per_second_by_pair": {"1": baseline - 1, "2": baseline, "3": baseline + 1},
        "candidate_output_tokens_per_second_by_pair": {"1": candidate - 1, "2": candidate, "3": candidate + 1},
        "baseline_output_tokens_per_second_median": baseline,
        "candidate_output_tokens_per_second_median": candidate,
        "ratio_of_medians_improvement_percent": improvement,
        "paired_improvement_percent": {
            "count": 3,
            "median": improvement,
            "p95": improvement + 0.5,
            "p99": improvement + 0.6,
            "min": improvement - 0.5,
            "max": improvement + 0.5,
        },
        "baseline_ttft_p95_ms_block_median": 500.0,
        "candidate_ttft_p95_ms_block_median": 250.0,
        "ttft_p95_reduction_percent": 50.0,
        "bootstrap": {
            "supported": True,
            "p05_percent": improvement - 1.0,
            "median_percent": improvement,
            "p95_percent": improvement + 1.0,
            "excludes_zero": True,
        },
        "guards_passed": True,
        "cleanup_passed": True,
        "throughput_eligible": True,
        "latency_eligible": True,
    }


def _result() -> dict:
    return {
        "schema_version": "1.0",
        "kind": "c7g4xlarge_realistic_arm_inference_validation",
        "status": "passed",
        "result_origin": "measured",
        "target": {
            "instance_class": "c7g.4xlarge",
            "cpu": "Neoverse-V1",
            "logical_cpus": 16,
            "topology": "single_host_loopback",
        },
        "source_commit": "a" * 40,
        "llama_cpp_commit": "b" * 40,
        "model_sha256": "c" * 64,
        "candidate_count_including_baseline": 4,
        "server_startups": 12,
        "measured_http_requests": 360,
        "llama_bench_scaling": {
            "rows": [
                {"threads": 1, "test": "pp512", "avg_tokens_per_second": 10.0, "stddev_tokens_per_second": 0.1},
                {"threads": 1, "test": "tg128", "avg_tokens_per_second": 3.0, "stddev_tokens_per_second": 0.1},
                {"threads": 8, "test": "pp512", "avg_tokens_per_second": 40.0, "stddev_tokens_per_second": 0.2},
                {"threads": 8, "test": "tg128", "avg_tokens_per_second": 18.0, "stddev_tokens_per_second": 0.2},
            ]
        },
        "screening_summaries": [
            {
                "profile_id": "slots8-cont-on",
                "workload_id": "prefill-heavy",
                "concurrency": 8,
                "failure_count": 12,
                "quality_failures": 12,
                "client_limited": True,
            }
        ],
        "capacity_matched_7cpu": {
            "comparisons": [_comparison("interactive-short", 16.0, 30.0)]
        },
        "full_capacity": {
            "4": {
                "comparisons": [_comparison("decode-heavy", 18.0, 34.0)]
            }
        },
        "recommendations": [{"recommendation_id": "example"}],
        "performance_passport_status": "eligible_preliminary",
        "cleanup_passed": True,
        "remote_artifacts_checked": 100,
        "remote_artifacts_failed": 0,
        "claim_boundary": (
            "Results are limited to one target, model, binary, and declared protocols."
        ),
    }
