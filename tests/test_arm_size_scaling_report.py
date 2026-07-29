from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from autotune.cli import main
from autotune.report.arm_size_scaling_report import (
    build_arm_size_scaling_summary,
    generate_arm_size_scaling_report,
)


class ArmSizeScalingReportTest(unittest.TestCase):
    def test_generates_sanitized_cross_instance_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            inputs = []
            revalidations = []
            for index, (instance, cpus) in enumerate(
                (("c7g.xlarge", 4), ("c7g.2xlarge", 8), ("c7g.4xlarge", 16))
            ):
                result_path = root / f"result-{index}.json"
                validation_path = root / f"validation-{index}.json"
                result_path.write_text(
                    json.dumps(_result(instance, cpus)),
                    encoding="utf-8",
                )
                validation_path.write_text(
                    json.dumps(_revalidation(index)),
                    encoding="utf-8",
                )
                inputs.append(result_path)
                revalidations.append(validation_path)

            outputs = generate_arm_size_scaling_report(
                inputs,
                revalidations,
                root / "public",
            )

            for path in outputs.values():
                self.assertTrue(Path(path).exists())
            summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], "1.0")
            self.assertEqual(summary["result_origin"], "measured")
            self.assertEqual([item["memory_gib"] for item in summary["targets"]], [8, 16, 32])
            self.assertEqual(summary["evidence_totals"]["measured_http_requests"], 1047)
            self.assertEqual(summary["evidence_totals"]["controller_artifacts_revalidated"], 303)
            self.assertEqual(len(summary["serving_comparisons"]), 6)
            self.assertNotIn("private", json.dumps(summary).lower())
            self.assertNotIn("\\\\", json.dumps(summary))
            html = Path(outputs["report"]).read_text(encoding="utf-8")
            self.assertIn("One workload, three C7g sizes, guarded tuning", html)
            self.assertIn("Prefill-heavy was not recommended", html)
            self.assertIn("Hardware-size scaling and runtime-tuning percentages", html)
            for chart in (
                "thread_scaling_across_sizes.svg",
                "serving_before_after_by_size.svg",
                "absolute_throughput_by_size.svg",
                "ttft_before_after_by_size.svg",
                "workload_guard_matrix.svg",
                "parallel_efficiency.svg",
            ):
                self.assertTrue((Path(outputs["charts"]) / chart).exists())

    def test_rejects_identity_mismatch(self) -> None:
        left = _result("c7g.xlarge", 4)
        right = _result("c7g.2xlarge", 8)
        right["model_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "model_sha256"):
            build_arm_size_scaling_summary(
                [left, right],
                [_revalidation(0), _revalidation(1)],
            )

    def test_rejects_failed_controller_validation(self) -> None:
        validation = _revalidation(0)
        validation["failed"] = 1
        validation["failures"] = ["digest mismatch"]
        with self.assertRaisesRegex(ValueError, "revalidation failed"):
            build_arm_size_scaling_summary(
                [_result("c7g.xlarge", 4), _result("c7g.2xlarge", 8)],
                [validation, _revalidation(1)],
            )

    def test_rejects_unqualified_formal_comparison(self) -> None:
        left = _result("c7g.xlarge", 4)
        left["paired_confirmation"]["groups"][0]["analysis"]["comparisons"][0][
            "bootstrap"
        ]["excludes_zero"] = False
        with self.assertRaisesRegex(ValueError, "formal paired comparison"):
            build_arm_size_scaling_summary(
                [left, _result("c7g.2xlarge", 8)],
                [_revalidation(0), _revalidation(1)],
            )

    def test_cli_exports_cross_instance_report(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            inputs = []
            revalidations = []
            for index, (instance, cpus) in enumerate(
                (("c7g.xlarge", 4), ("c7g.2xlarge", 8))
            ):
                result_path = root / f"result-{index}.json"
                validation_path = root / f"validation-{index}.json"
                result_path.write_text(json.dumps(_result(instance, cpus)), encoding="utf-8")
                validation_path.write_text(json.dumps(_revalidation(index)), encoding="utf-8")
                inputs.extend(["--input", str(result_path)])
                revalidations.extend(["--controller-revalidation", str(validation_path)])
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "report-arm-size-scaling",
                            *inputs,
                            *revalidations,
                            "--output-dir",
                            str(root / "public"),
                        ]
                    ),
                    0,
                )
            payload = json.loads(stdout.getvalue())
            self.assertTrue(Path(payload["report"]).exists())


def _result(instance: str, cpus: int) -> dict:
    threads = cpus - 1
    profile = f"slots4-cont-on-threads{threads}"
    baseline = float(cpus)
    candidate = baseline * 1.8
    return {
        "schema_version": "1.0",
        "kind": "c7g_size_scaling_comparative_result",
        "status": "passed",
        "result_origin": "measured",
        "target": {
            "instance_class": instance,
            "cpu": "Neoverse-V1",
            "logical_cpus": cpus,
            "topology": "single_host_loopback",
        },
        "source_commit": "a" * 40,
        "llama_cpp_commit": "b" * 40,
        "model_sha256": "c" * 64,
        "runtime_bundle_sha256": "d" * 64,
        "bench_binary_sha256": "e" * 64,
        "server_binary_sha256": "1" * 64,
        "candidate_count_including_baseline": 3,
        "measured_http_requests": 349,
        "server_startups": 10,
        "cleanup_passed": True,
        "host_restored": True,
        "remote_artifacts_checked": 100,
        "remote_artifacts_failed": 0,
        "failures": [],
        "performance_passport_status": "eligible_preliminary",
        "recommendations": [
            {
                "profile_id": profile,
                "eligible_workloads": ["decode-heavy", "interactive-short"],
            }
        ],
        "llama_bench_scaling": {
            "measured_repetitions": 5,
            "rows": [
                {
                    "threads": 1,
                    "test": "pp512",
                    "avg_tokens_per_second": 6.7,
                    "stddev_tokens_per_second": 0.01,
                },
                {
                    "threads": 1,
                    "test": "tg128",
                    "avg_tokens_per_second": 2.7,
                    "stddev_tokens_per_second": 0.01,
                },
                {
                    "threads": threads,
                    "test": "pp512",
                    "avg_tokens_per_second": 6.7 * threads * 0.75,
                    "stddev_tokens_per_second": 0.04,
                },
                {
                    "threads": threads,
                    "test": "tg128",
                    "avg_tokens_per_second": 2.7 * threads * 0.72,
                    "stddev_tokens_per_second": 0.04,
                },
            ],
        },
        "screening_decisions": [
            _screening("interactive-short", profile, 70.0, True),
            _screening("decode-heavy", profile, 80.0, True),
            _screening("prefill-heavy", profile, -30.0, False),
        ],
        "paired_confirmation": {
            "groups": [
                {
                    "profile_id": profile,
                    "analysis": {
                        "comparisons": [
                            _comparison("interactive-short", baseline, candidate),
                            _comparison("decode-heavy", baseline + 1, candidate + 1),
                        ]
                    },
                }
            ]
        },
    }


def _screening(workload: str, profile: str, delta: float, eligible: bool) -> dict:
    return {
        "workload_id": workload,
        "concurrency": 4,
        "candidates": [
            {
                "profile_id": profile.replace("slots4", "slots2"),
                "throughput_improvement_percent": 1.0 if eligible else -15.0,
                "screening_eligible": False,
                "guard_reasons": [] if eligible else ["ttft_regression"],
            },
            {
                "profile_id": profile,
                "throughput_improvement_percent": delta,
                "screening_eligible": eligible,
                "guard_reasons": [] if eligible else ["ttft_regression"],
            },
        ],
    }


def _comparison(workload: str, baseline: float, candidate: float) -> dict:
    improvement = (candidate / baseline - 1.0) * 100.0
    return {
        "workload_id": workload,
        "pair_count": 3,
        "baseline_output_tokens_per_second_median": baseline,
        "candidate_output_tokens_per_second_median": candidate,
        "ratio_of_medians_improvement_percent": improvement,
        "paired_improvement_percent": {"median": improvement},
        "bootstrap": {
            "p05_percent": improvement - 1,
            "p95_percent": improvement + 1,
            "excludes_zero": True,
        },
        "baseline_ttft_p95_ms_block_median": 1000.0,
        "candidate_ttft_p95_ms_block_median": 250.0,
        "ttft_p95_reduction_percent": 75.0,
        "guards_passed": True,
        "cleanup_passed": True,
        "throughput_eligible": True,
        "latency_eligible": True,
    }


def _revalidation(index: int) -> dict:
    return {
        "schema_version": "1.0",
        "checked": 100 + index,
        "failed": 0,
        "failures": [],
        "archive_sha256": f"{index + 2:x}" * 64,
    }
