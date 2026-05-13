from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.report.run_report import generate_run_report
from autotune.report.comparison_report import generate_comparison_report
from autotune.resource.run_state import write_json


class RunReportTest(unittest.TestCase):
    def test_generate_run_report_writes_markdown(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_dir = Path(temp_dir)
            run_dir = runs_dir / "run1"
            run_dir.mkdir()
            write_json(
                run_dir / "manifest.json",
                {
                    "run_id": "run1",
                    "status": "completed",
                    "return_code": 0,
                    "command": ["python", "train.py"],
                    "budget": {"reserve_cores": 1, "allowed_threads": 3},
                    "notes": ["selected_executor=local"],
                },
            )
            write_json(
                run_dir / "resource_summary.json",
                {
                    "peak_rss_mb": 128,
                    "memory_budget_exceeded": False,
                    "peak_child_cpu_percent": 75,
                    "system_cpu_percent_p95": 50,
                    "per_cpu_peak_max_percent": 80,
                },
            )
            write_json(run_dir / "resource_timeline.json", [{"system_cpu_percent": 50, "available_memory_mb": 1024}])

            report_path = generate_run_report("run1", runs_dir=runs_dir)

            self.assertEqual(report_path, run_dir / "report.md")
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("PerfRunbench Run Report: run1", report)
            self.assertIn("## Before / After", report)
            self.assertIn("## Visual Summary", report)
            self.assertIn("<svg", report)
            self.assertIn("Available memory at start", report)
            self.assertIn("## CPU", report)
            self.assertIn("Observed system CPU p95 percent", report)
            self.assertIn("## Memory", report)
            self.assertIn("## GPU Tuning", report)
            self.assertIn("## Diagnostics", report)

    def test_generate_run_report_surfaces_gpu_tuning_status(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_dir = Path(temp_dir)
            run_dir = runs_dir / "run1"
            run_dir.mkdir()
            write_json(
                run_dir / "manifest.json",
                {
                    "run_id": "run1",
                    "status": "completed",
                    "return_code": 0,
                    "command": ["python", "train.py"],
                    "budget": {},
                    "notes": ["selected_executor=local"],
                },
            )
            write_json(run_dir / "resource_summary.json", {"memory_budget_exceeded": False})
            write_json(run_dir / "resource_timeline.json", [])
            write_json(run_dir / "gpu_tuning_plan.json", {"profile": "nvidia-guard"})
            write_json(
                run_dir / "gpu_tuning_diff.json",
                [
                    {"key": "persistence_mode", "command": ["nvidia-smi", "-pm", "1"], "return_code": 0},
                    {"key": "power.limit", "command": ["nvidia-smi", "-pl", "60"], "return_code": 1},
                ],
            )
            write_json(run_dir / "gpu_tuning_restore_after.json", {"changes": [{"key": "power.limit", "return_code": 0}]})

            report_path = generate_run_report("run1", run_dir / "report.html", runs_dir=runs_dir)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("GPU Tuning", report)
            self.assertIn("nvidia-guard", report)
            self.assertIn("1/2", report)
            self.assertIn("power.limit", report)

    def test_generate_run_report_explains_minimal_mode_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_dir = Path(temp_dir)
            run_dir = runs_dir / "run1"
            run_dir.mkdir()
            write_json(
                run_dir / "manifest.json",
                {
                    "run_id": "run1",
                    "status": "completed",
                    "return_code": 0,
                    "command": ["python", "train.py"],
                    "budget": {},
                    "notes": ["selected_executor=systemd"],
                },
            )
            write_json(
                run_dir / "resource_summary.json",
                {"samples": 0, "monitoring_mode": "none", "memory_budget_exceeded": False},
            )
            write_json(run_dir / "training_metrics.json", {"samples_per_second": 100.0})

            report_path = generate_run_report("run1", run_dir / "report.html", runs_dir=runs_dir)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Monitoring mode", report)
            self.assertIn("not collected in minimal mode", report)

    def test_generate_comparison_report_writes_visual_markdown(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "comparison.json"
            write_json(
                path,
                {
                    "tuned_profile": "linux-throughput",
                    "baseline": {"run_id": "b1", "benchmark_duration_seconds": 10, "lifecycle_duration_seconds": 11, "peak_memory_mb": 100},
                    "tuned": {"run_id": "t1", "benchmark_duration_seconds": 8, "lifecycle_duration_seconds": 9, "peak_memory_mb": 90},
                    "deltas": {"benchmark_duration_percent": -20, "peak_memory_percent": -10},
                },
            )

            report_path = generate_comparison_report(path)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("PerfRunbench Tuning Comparison", report)
            self.assertIn("Performance Deltas", report)
            self.assertIn("GPU Tuning Effectiveness", report)
            self.assertIn("<svg", report)

    def test_generate_comparison_report_writes_html_when_requested(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "comparison.json"
            output = Path(temp_dir) / "comparison.html"
            write_json(
                path,
                {
                    "tuned_profile": "linux-low-latency",
                    "baseline": {"run_id": "b1", "benchmark_duration_seconds": 10, "lifecycle_duration_seconds": 11, "peak_memory_mb": 100},
                    "tuned": {"run_id": "t1", "benchmark_duration_seconds": 9, "lifecycle_duration_seconds": 10, "peak_memory_mb": 95},
                    "deltas": {"benchmark_duration_percent": -10, "peak_memory_percent": -5},
                },
            )

            report_path = generate_comparison_report(path, output)

            report = report_path.read_text(encoding="utf-8")
            self.assertEqual(report_path, output)
            self.assertIn("<!doctype html>", report.lower())
            self.assertIn("Tuning Comparison", report)
            self.assertIn("<svg", report)

    def test_generate_profile_summary_report_writes_html(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "profile_summary.json"
            output = Path(temp_dir) / "profile_summary.html"
            write_json(
                path,
                {
                    "kind": "profile_comparison_summary",
                    "repeat": 3,
                    "best_profile": "linux-performance",
                    "best_profile_beats_baseline": True,
                    "comparisons": [
                        {
                            "profile": "linux-performance",
                            "output": "results/reports/linux_performance_comparison.json",
                            "samples_per_second_percent": 5.5,
                            "benchmark_duration_percent": -1.2,
                            "peak_memory_percent": 0.0,
                        }
                    ],
                },
            )

            report_path = generate_comparison_report(path, output)

            report = report_path.read_text(encoding="utf-8")
            self.assertEqual(report_path, output)
            self.assertIn("Profile Comparison Summary", report)
            self.assertIn("linux-performance", report)

    def test_generate_auto_recommendation_report_writes_html(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "auto_recommendation.json"
            output = Path(temp_dir) / "auto_recommendation.html"
            write_json(
                path,
                {
                    "kind": "auto_recommendation",
                    "best_label": "unbounded:linux-performance",
                    "cache_path": ".autotuneai/recommendations/latest.json",
                    "fingerprint": "abc",
                    "repeat": 1,
                    "warmup_runs": 1,
                    "decision": {
                        "status": "meaningful-speedup",
                        "baseline_label": "unbounded:baseline",
                        "recommended_label": "unbounded:linux-performance",
                        "noise_band_percent": 2.0,
                        "primary_speed_delta_percent": 20.0,
                        "samples_per_second_delta_percent": 20.0,
                        "step_time_p95_delta_percent": -20.0,
                        "gpu_tflops_delta_percent": 25.0,
                        "within_noise_band": False,
                        "recommendation_reason": ["system_profile=linux-performance", "samples_per_second=120"],
                        "interpretation": "Recommended profile beat baseline outside the noise band.",
                    },
                    "recommendation": {
                        "label": "unbounded:linux-performance",
                        "guard_mode": "unbounded",
                        "system_profile": "linux-performance",
                        "runtime_profile": "runtime-pytorch-max-performance",
                        "gpu_profile": "nvidia-performance",
                        "metrics": {
                            "samples_per_second": 120,
                            "duration_seconds": 9,
                            "step_time_p95_seconds": 0.08,
                            "step_time_p99_seconds": 0.09,
                            "gpu_tflops_estimate": 10,
                        },
                    },
                    "candidates": [
                        {
                            "label": "unbounded:baseline",
                            "status": "completed",
                            "metrics": {
                                "samples_per_second": 100,
                                "duration_seconds": 10,
                                "step_time_p95_seconds": 0.1,
                                "step_time_p99_seconds": 0.12,
                                "gpu_tflops_estimate": 8,
                            },
                        },
                        {
                            "label": "unbounded:linux-performance",
                            "status": "completed",
                            "system_profile": "linux-performance",
                            "runtime_profile": "runtime-pytorch-max-performance",
                            "gpu_profile": "nvidia-performance",
                            "metrics": {
                                "samples_per_second": 120,
                                "duration_seconds": 9,
                                "step_time_p95_seconds": 0.08,
                                "step_time_p99_seconds": 0.09,
                                "gpu_tflops_estimate": 10,
                            },
                        },
                    ],
                },
            )

            report_path = generate_comparison_report(path, output)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Auto Recommendation", report)
            self.assertIn("Current vs Recommended", report)
            self.assertIn("Decision", report)
            self.assertIn("Why This Recommendation", report)
            self.assertIn("meaningful-speedup", report)
            self.assertIn("Step p95", report)
            self.assertIn("Logical CPU count", report)
            self.assertIn("Busiest core peak %", report)
            self.assertIn("unbounded:linux-performance", report)


if __name__ == "__main__":
    unittest.main()
