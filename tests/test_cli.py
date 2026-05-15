from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from autotune.cli import main


class CliTest(unittest.TestCase):
    def test_executors_command_prints_capabilities(self) -> None:
        with patch("autotune.cli.collect_executor_capabilities") as collect:
            collect.return_value = {"recommended_executor": "local", "executors": {}}
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["executors"])
        self.assertEqual(code, 0)
        self.assertIn("recommended_executor", output.getvalue())

    def test_analyze_command_prints_report(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.analyze_run") as analyze, redirect_stdout(output):
            analyze.return_value = {
                "run_id": "run1",
                "status": "completed",
                "return_code": 0,
                "executor": {},
                "cpu": {},
                "memory": {},
                "cgroup": {},
                "diagnostics": [],
            }
            code = main(["analyze", "--run-id", "run1"])
        self.assertEqual(code, 0)
        self.assertIn("Run run1", output.getvalue())

    def test_report_command_writes_report(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.generate_run_report") as generate, redirect_stdout(output):
            generate.return_value = "report.md"
            code = main(["report", "--run-id", "run1"])
        self.assertEqual(code, 0)
        generate.assert_called_once_with("run1", None)
        self.assertIn("Wrote run report", output.getvalue())

    def test_report_comparison_command_writes_report(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.generate_comparison_report") as generate, redirect_stdout(output):
            generate.return_value = "comparison.md"
            code = main(["report-comparison", "--input", "comparison.json"])
        self.assertEqual(code, 0)
        generate.assert_called_once_with("comparison.json", None)
        self.assertIn("Wrote tuning comparison report", output.getvalue())

    def test_calibrate_memory_command_writes_summary(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.calibrate_memory") as calibrate, redirect_stdout(output):
            calibrate.return_value = {"records": [], "recommendations": []}
            code = main(["calibrate-memory", "--budget-gb", "-5", "--workload-memory-mb", "256"])
        self.assertEqual(code, 0)
        calibrate.assert_called_once()
        self.assertIn("Wrote memory calibration", output.getvalue())

    def test_tune_batch_accepts_generic_key(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.tune_batch_size") as tune, redirect_stdout(output):
            tune.return_value = {"recommended_value": 2}
            code = main(["tune-batch", "--file", "train.yaml", "--key", "num_workers", "--values", "0", "2", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(tune.call_args.args[1], "num_workers")
        self.assertIn("Wrote training tuning summary", output.getvalue())

    def test_tune_training_accepts_multiple_knobs(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.tune_training_knobs") as tune, redirect_stdout(output):
            tune.return_value = {"final_recommendation": {"batch_size": 32}}
            code = main([
                "tune-training",
                "--file",
                "train.yaml",
                "--knob",
                "batch_size=16,32",
                "--knob",
                "dataloader_workers=0,2",
                "--min-final-accuracy",
                "0.9",
                "--",
                "python",
                "train.py",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(tune.call_args.args[1], {"batch_size": [16, 32], "dataloader_workers": [0, 2]})
        self.assertEqual(tune.call_args.kwargs["min_final_accuracy"], 0.9)
        self.assertIn("Wrote training plan summary", output.getvalue())

    def test_compare_tuning_command_runs_comparison(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.compare_tuning") as compare, patch("autotune.cli.generate_comparison_report") as report, redirect_stdout(output):
            compare.return_value = {"deltas": {}}
            report.return_value = Path("results/reports/tuning_comparison.html")
            code = main([
                "compare-tuning",
                "--profile",
                "linux-throughput",
                "--runtime-profile",
                "runtime-cpu-performance",
                "--repeat",
                "3",
                "--",
                "python",
                "train.py",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs["tuned_profile"], "linux-throughput")
        self.assertEqual(compare.call_args.kwargs["tuned_runtime_env_profile"], "runtime-cpu-performance")
        self.assertEqual(compare.call_args.kwargs["repeat"], 3)
        self.assertTrue(compare.call_args.kwargs["alternate_order"])
        report.assert_called_once()
        self.assertIn("Wrote tuning comparison", output.getvalue())

    def test_compare_profiles_command_runs_summary(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.compare_profiles") as compare, patch("autotune.cli.generate_comparison_report") as report, redirect_stdout(output):
            compare.return_value = {
                "best_profile": "linux-low-latency",
                "comparisons": [
                    {"profile": "linux-throughput", "output": "results/reports/linux_throughput_comparison.json"},
                    {"profile": "linux-low-latency", "output": "results/reports/linux_low_latency_comparison.json"},
                ],
            }
            report.return_value = Path("results/reports/profile_summary.html")
            code = main(["compare-profiles", "--profiles", "linux-throughput", "linux-low-latency", "--repeat", "2", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs["profiles"], ["linux-throughput", "linux-low-latency"])
        self.assertEqual(compare.call_args.kwargs["repeat"], 2)
        self.assertEqual(report.call_count, 3)
        self.assertIn("Wrote profile comparison summary", output.getvalue())

    def test_compare_budgets_command_runs_summary(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.compare_budget_modes") as compare, patch("autotune.cli.generate_comparison_report") as report, redirect_stdout(output):
            compare.return_value = {"kind": "budget_mode_comparison"}
            report.return_value = Path("results/reports/budget_comparison.html")
            code = main([
                "compare-budgets",
                "--memory-budget-gb",
                "-3",
                "--profile",
                "linux-low-latency",
                "--repeat",
                "2",
                "--",
                "python",
                "train.py",
            ])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs["tuned_profile"], "linux-low-latency")
        self.assertEqual(compare.call_args.args[1].memory_budget_gb, -3)
        report.assert_called_once()
        self.assertIn("Wrote budget comparison", output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Windows")
    def test_compare_tuning_auto_selects_windows_profile(self, _system) -> None:
        output = io.StringIO()
        with patch("autotune.cli.compare_tuning") as compare, patch("autotune.cli.generate_comparison_report"), redirect_stdout(output):
            compare.return_value = {"deltas": {}}
            code = main(["compare-tuning", "--workload-profile", "throughput", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs["tuned_profile"], "windows-throughput")

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Windows")
    def test_compare_tuning_auto_selects_windows_performance_profile(self, _system) -> None:
        output = io.StringIO()
        with patch("autotune.cli.compare_tuning") as compare, patch("autotune.cli.generate_comparison_report"), redirect_stdout(output):
            compare.return_value = {"deltas": {}}
            code = main(["compare-tuning", "--workload-profile", "performance", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs["tuned_profile"], "windows-performance")

    def test_compare_runs_writes_output(self) -> None:
        output = io.StringIO()
        with patch("autotune.resource.comparison_runner.build_comparison_result") as compare, redirect_stdout(output):
            compare.return_value = {"baseline": {}, "tuned": {}, "deltas": {}}
            code = main(["compare-runs", "--baseline-run-id", "run1", "--tuned-run-id", "run2"])
        self.assertEqual(code, 0)
        self.assertIn("Wrote run comparison", output.getvalue())

    def test_optimize_command_writes_recommendation(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.optimize_recommendation") as optimize, patch("autotune.cli.generate_comparison_report") as report, redirect_stdout(output):
            optimize.return_value = {"recommendation": {"label": "best"}}
            report.return_value = Path("results/reports/auto_recommendation.html")
            code = main(["optimize", "--max-candidates", "2", "--warmup-runs", "1", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(optimize.call_args.kwargs["max_candidates"], 2)
        self.assertEqual(optimize.call_args.kwargs["warmup_runs"], 1)
        self.assertEqual(optimize.call_args.kwargs["optimization_target"], "auto")
        self.assertEqual(optimize.call_args.kwargs["output"], "results/reports/auto_recommendation.json")
        self.assertEqual(optimize.call_args.kwargs["cache_path"], str(Path(".autotuneai") / "recommendations" / "latest.json"))
        report.assert_called_once()
        self.assertIn("Cached recommendation", output.getvalue())

    def test_optimize_performance_command_uses_performance_mode(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.optimize_recommendation") as optimize, patch("autotune.cli.generate_comparison_report") as report, redirect_stdout(output):
            optimize.return_value = {"recommendation": {"label": "performance:baseline"}}
            report.return_value = Path("results/reports/performance_recommendation.html")
            code = main(["optimize-performance", "--max-candidates", "2", "--time-budget-hours", "8", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(optimize.call_args.kwargs["optimization_mode"], "performance")
        self.assertEqual(optimize.call_args.kwargs["optimization_target"], "auto")
        self.assertEqual(optimize.call_args.kwargs["monitor_mode"], "minimal")
        self.assertEqual(optimize.call_args.kwargs["time_budget_hours"], 8)
        self.assertTrue(optimize.call_args.kwargs["thermal_control"])
        self.assertFalse(optimize.call_args.args[1].enforce)
        self.assertEqual(optimize.call_args.kwargs["sample_interval_seconds"], 5.0)
        self.assertFalse(optimize.call_args.kwargs["hard_kill"])
        self.assertEqual(optimize.call_args.kwargs["output"], "results/reports/performance_recommendation.json")
        self.assertEqual(optimize.call_args.kwargs["cache_path"], str(Path(".autotuneai") / "recommendations" / "latest.json"))
        report.assert_called_once()
        self.assertIn("Best performance recommendation", output.getvalue())

    def test_optimize_performance_command_accepts_target(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.optimize_recommendation") as optimize, patch("autotune.cli.generate_comparison_report") as report, redirect_stdout(output):
            optimize.return_value = {"recommendation": {"label": "performance:runtime-cpu"}}
            report.return_value = Path("results/reports/performance_recommendation.html")
            code = main(["optimize-performance", "--target", "cpu", "--max-candidates", "2", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(optimize.call_args.kwargs["optimization_target"], "cpu")
        self.assertEqual(optimize.call_args.kwargs["output"], str(Path("results/reports/performance_recommendation_cpu.json")))
        self.assertEqual(optimize.call_args.kwargs["cache_path"], str(Path(".autotuneai") / "recommendations" / "latest_performance_cpu.json"))
        self.assertEqual(report.call_args.args[0], Path("results/reports/performance_recommendation_cpu.json"))
        self.assertIn("performance_recommendation_cpu.json", output.getvalue())

    def test_optimize_command_uses_target_specific_defaults(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.optimize_recommendation") as optimize, patch("autotune.cli.generate_comparison_report") as report, redirect_stdout(output):
            optimize.return_value = {"recommendation": {"label": "budgeted:gpu-guard"}}
            report.return_value = Path("results/reports/auto_recommendation_gpu.html")
            code = main(["optimize", "--target", "gpu", "--max-candidates", "2", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(optimize.call_args.kwargs["optimization_target"], "gpu")
        self.assertEqual(optimize.call_args.kwargs["output"], str(Path("results/reports/auto_recommendation_gpu.json")))
        self.assertEqual(optimize.call_args.kwargs["cache_path"], str(Path(".autotuneai") / "recommendations" / "latest_guarded_gpu.json"))
        self.assertEqual(report.call_args.args[0], Path("results/reports/auto_recommendation_gpu.json"))

    @patch("autotune.cli.launch_performance")
    @patch("autotune.cli.generate_run_report")
    def test_launch_performance_applies_cached_recommendation_without_budget(self, generate_report, launch_performance) -> None:
        launch_performance.return_value = (0, ".autotuneai/runs/run1")
        generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
        cached = {
            "recommendation": {
                "label": "performance:linux-performance",
                "system_profile": "linux-performance",
                "runtime_profile": "runtime-pytorch-max-performance",
                "gpu_profile": "nvidia-performance",
                "budget": {"memory_budget_gb": -3, "resource_budget_enforced": True},
            }
        }
        output = io.StringIO()
        with patch("autotune.cli.load_recommendation", return_value=cached), redirect_stdout(output):
            code = main(["launch-performance", "--apply-recommendation", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertFalse(launch_performance.call_args.args[1].enforce)
        self.assertEqual(launch_performance.call_args.kwargs["tune_system_profile"], "linux-performance")
        self.assertEqual(launch_performance.call_args.kwargs["runtime_env_profile"], "runtime-pytorch-max-performance")
        self.assertEqual(launch_performance.call_args.kwargs["tune_gpu_profile"], "nvidia-performance")
        self.assertIn("without resource monitoring", output.getvalue())

    @patch("autotune.cli.run_with_budget")
    @patch("autotune.cli.generate_run_report")
    def test_run_command_applies_cached_recommendation(self, generate_report, run_with_budget) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
        cached = {
            "recommendation": {
                "system_profile": "linux-throughput",
                "runtime_profile": "runtime-pytorch-max-performance",
                "gpu_profile": "nvidia-performance",
                "budget": {
                    "memory_budget_gb": -3,
                    "reserve_memory_gb": 0.0,
                    "reserve_cores": 1,
                    "cpu_quota_percent": 90,
                    "resource_budget_enforced": True,
                },
            }
        }
        with patch("autotune.cli.load_recommendation", return_value=cached), redirect_stdout(io.StringIO()):
            code = main(["run", "--apply-recommendation", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.args[1].memory_budget_gb, -3)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-throughput")
        self.assertEqual(run_with_budget.call_args.kwargs["runtime_env_profile"], "runtime-pytorch-max-performance")
        self.assertEqual(run_with_budget.call_args.kwargs["tune_gpu_profile"], "nvidia-performance")

    @patch("autotune.cli.run_with_budget")
    def test_demo_run_executes_builtin_workload(self, run_with_budget) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/demo1")
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["demo"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.args[0], ["python", "examples/dummy_train.py"])
        self.assertIn('"scenario": "run"', output.getvalue())

    @patch("autotune.cli.tune_batch_size")
    def test_demo_tune_batch_uses_builtin_config(self, tune_batch_size) -> None:
        tune_batch_size.return_value = {"recommended_value": 64}
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["demo", "--scenario", "tune-batch"])
        self.assertEqual(code, 0)
        self.assertEqual(tune_batch_size.call_args.args[0], "examples/train_config.yaml")
        self.assertEqual(tune_batch_size.call_args.args[3], ["python", "examples/dummy_train.py"])
        self.assertIn('"recommended_value": 64', output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Linux")
    @patch("autotune.cli.compare_tuning")
    def test_demo_compare_tuning_runs_on_linux(self, compare_tuning, _system) -> None:
        compare_tuning.return_value = {"tuned_profile": "linux-training-safe"}
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["demo", "--scenario", "compare-tuning"])
        self.assertEqual(code, 0)
        self.assertEqual(compare_tuning.call_args.args[0], ["python", "examples/dummy_train.py"])
        self.assertIn('"tuned_profile": "linux-training-safe"', output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Windows")
    @patch("autotune.cli.compare_tuning")
    def test_demo_compare_tuning_runs_on_windows(self, compare_tuning, _system) -> None:
        compare_tuning.return_value = {"tuned_profile": "windows-training-safe"}
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["demo", "--scenario", "compare-tuning"])
        self.assertEqual(code, 0)
        self.assertEqual(compare_tuning.call_args.kwargs["tuned_profile"], "windows-training-safe")

    def test_tune_gpu_command_prints_recommendations(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.recommend_nvidia_tuning") as recommend, redirect_stdout(output):
            recommend.return_value = {"supported": False}
            code = main(["tune-gpu"])
        self.assertEqual(code, 0)
        recommend.assert_called_once_with("nvidia-throughput")
        self.assertIn("supported", output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Windows")
    def test_tune_system_defaults_to_platform_profile(self, _system) -> None:
        output = io.StringIO()
        with patch("autotune.cli.recommend_system_tuning") as recommend, redirect_stdout(output):
            recommend.return_value = {"profile": "windows-training-safe"}
            code = main(["tune-system"])
        self.assertEqual(code, 0)
        recommend.assert_called_once_with("windows-training-safe")

    def test_tune_system_recommend_all_prints_supported_profiles(self) -> None:
        output = io.StringIO()
        with patch("autotune.cli.recommend_system_tuning") as recommend, redirect_stdout(output):
            recommend.side_effect = lambda profile: {"profile": profile, "supported": profile.startswith("linux-")}
            code = main(["tune-system", "--recommend-all"])
        self.assertEqual(code, 0)
        self.assertIn("linux-training-safe", output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Linux")
    @patch("autotune.cli.run_with_budget")
    @patch("autotune.cli.generate_run_report")
    def test_run_command_auto_tunes_system_on_linux(self, generate_report, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["run", "--auto-tune-system", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-training-safe")
        generate_report.assert_called_once()
        self.assertIn("Run directory", output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Linux")
    @patch("autotune.cli.run_with_budget")
    @patch("autotune.cli.generate_run_report")
    def test_run_command_auto_tunes_memory_profile_when_budgeted(self, generate_report, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
        with redirect_stdout(io.StringIO()):
            code = main(["run", "--auto-tune-system", "--memory-budget-gb", "-3", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-memory-conservative")

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Windows")
    @patch("autotune.cli.run_with_budget")
    @patch("autotune.cli.generate_run_report")
    def test_run_command_auto_tunes_system_on_windows(self, generate_report, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
        with redirect_stdout(io.StringIO()):
            code = main(["run", "--auto-tune-system", "--workload-profile", "throughput", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "windows-throughput")

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Linux")
    @patch("autotune.cli.run_with_budget")
    @patch("autotune.cli.generate_run_report")
    def test_run_command_auto_tunes_performance_profile(self, generate_report, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
        with redirect_stdout(io.StringIO()):
            code = main(["run", "--auto-tune-system", "--workload-profile", "performance", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-performance")

    @patch("autotune.cli.recommend_nvidia_tuning")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_gpu(self, run_with_budget, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": True}
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with patch("autotune.cli.generate_run_report") as generate_report, redirect_stdout(io.StringIO()):
            generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
            code = main(["run", "--auto-tune-gpu", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_gpu_profile"], "nvidia-performance")

    @patch("autotune.cli.recommend_nvidia_tuning")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_gpu_guard_when_budgeted(self, run_with_budget, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": True}
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with patch("autotune.cli.generate_run_report") as generate_report, redirect_stdout(io.StringIO()):
            generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
            code = main(["run", "--auto-tune-gpu", "--memory-budget-gb", "-3", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        recommend_gpu.assert_called_once_with("nvidia-guard")
        self.assertEqual(run_with_budget.call_args.kwargs["tune_gpu_profile"], "nvidia-guard")

    @patch("autotune.cli.run_with_budget")
    def test_run_command_applies_runtime_profile(self, run_with_budget) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with patch("autotune.cli.generate_run_report") as generate_report, redirect_stdout(io.StringIO()):
            generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
            code = main(["run", "--runtime-profile", "runtime-pytorch-max-performance", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["runtime_env_profile"], "runtime-pytorch-max-performance")

    def test_run_command_requires_confirmation_for_advanced_tuning(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run", "--runtime-profile", "runtime-pytorch-aggressive", "--", "python", "train.py"])

    @patch("autotune.cli.run_with_budget")
    def test_run_command_passes_advanced_options(self, run_with_budget) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with patch("autotune.cli.generate_run_report") as generate_report, redirect_stdout(io.StringIO()):
            generate_report.return_value = Path(".autotuneai/runs/run1/report.html")
            code = main([
                "run",
                "--confirm-advanced-tuning",
                "--numa-node",
                "0",
                "--extra-env",
                "CUDNN_BENCHMARK=1",
                "--",
                "python",
                "train.py",
            ])
        self.assertEqual(code, 0)
        advanced = run_with_budget.call_args.kwargs["advanced_options"]
        self.assertEqual(advanced.numa_node, 0)
        self.assertEqual(advanced.extra_env, {"CUDNN_BENCHMARK": "1"})

    def test_tune_runtime_command_prints_env(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["tune-runtime", "--profile", "runtime-cpu-performance"])
        self.assertEqual(code, 0)
        self.assertIn("OMP_NUM_THREADS", output.getvalue())

    def test_run_command_rejects_conflicting_system_tuning_options(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run", "--auto-tune-system", "--tune-system", "linux-training-safe", "--", "python", "train.py"])

    def test_run_command_requires_workload(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run"])

    @patch("autotune.cli.list_runs")
    @patch("autotune.cli.restore_system_tuning")
    @patch("autotune.cli.restore_nvidia_tuning")
    def test_restore_latest_uses_most_recent_run(self, restore_gpu, restore_system, list_runs_mock) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_dir = Path(temp_dir)
            run_dir = runs_dir / "latest1"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text('{"changed_files": []}', encoding="utf-8")
            list_runs_mock.return_value = [{"run_id": "latest1"}]
            restore_system.return_value = []
            restore_gpu.return_value = []
            output = io.StringIO()
            with patch("autotune.cli.RUNS_DIR", runs_dir), redirect_stdout(output):
                code = main(["restore", "--latest"])
        self.assertEqual(code, 0)
        restore_system.assert_called_once()
        self.assertIn("latest1", output.getvalue())

    @patch("autotune.cli.restore_system_tuning")
    @patch("autotune.cli.restore_nvidia_tuning")
    def test_restore_active_uses_active_tuning_state(self, restore_gpu, restore_system) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_dir = Path(temp_dir) / "runs"
            runs_dir.mkdir()
            run_dir = runs_dir / "active1"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text('{"changed_files": []}', encoding="utf-8")
            active_state_path = Path(temp_dir) / "active_tuning_state.json"
            active_state_path.write_text('{"run_id": "active1", "system_active": true}', encoding="utf-8")
            restore_system.return_value = []
            restore_gpu.return_value = []
            output = io.StringIO()
            with (
                patch("autotune.cli.RUNS_DIR", runs_dir),
                patch("autotune.cli.ACTIVE_TUNING_STATE", active_state_path),
                patch("autotune.resource.run_state.ACTIVE_TUNING_STATE", active_state_path),
                redirect_stdout(output),
            ):
                code = main(["restore", "--active"])
        self.assertEqual(code, 0)
        restore_system.assert_called_once()
        self.assertIn("Cleared active tuning state", output.getvalue())


if __name__ == "__main__":
    unittest.main()
