from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
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
        with patch("autotune.cli.compare_tuning") as compare, redirect_stdout(output):
            compare.return_value = {"deltas": {}}
            code = main(["compare-tuning", "--profile", "linux-throughput", "--repeat", "3", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs["tuned_profile"], "linux-throughput")
        self.assertEqual(compare.call_args.kwargs["repeat"], 3)
        self.assertIn("Wrote tuning comparison", output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Windows")
    def test_compare_tuning_auto_selects_windows_profile(self, _system) -> None:
        output = io.StringIO()
        with patch("autotune.cli.compare_tuning") as compare, redirect_stdout(output):
            compare.return_value = {"deltas": {}}
            code = main(["compare-tuning", "--workload-profile", "throughput", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs["tuned_profile"], "windows-throughput")

    def test_compare_runs_writes_output(self) -> None:
        output = io.StringIO()
        with patch("autotune.resource.comparison_runner.build_comparison_result") as compare, redirect_stdout(output):
            compare.return_value = {"baseline": {}, "tuned": {}, "deltas": {}}
            code = main(["compare-runs", "--baseline-run-id", "run1", "--tuned-run-id", "run2"])
        self.assertEqual(code, 0)
        self.assertIn("Wrote run comparison", output.getvalue())

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

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Linux")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_system_on_linux(self, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["run", "--auto-tune-system", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-training-safe")
        self.assertIn("Run directory", output.getvalue())

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Linux")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_memory_profile_when_budgeted(self, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with redirect_stdout(io.StringIO()):
            code = main(["run", "--auto-tune-system", "--memory-budget-gb", "-3", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-memory-conservative")

    @patch("autotune.system_tuner.profile_selector.platform.system", return_value="Windows")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_system_on_windows(self, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with redirect_stdout(io.StringIO()):
            code = main(["run", "--auto-tune-system", "--workload-profile", "throughput", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "windows-throughput")

    @patch("autotune.cli.recommend_nvidia_tuning")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_gpu(self, run_with_budget, recommend_gpu) -> None:
        recommend_gpu.return_value = {"supported": True}
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with redirect_stdout(io.StringIO()):
            code = main(["run", "--auto-tune-gpu", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_gpu_profile"], "nvidia-throughput")

    def test_run_command_rejects_conflicting_system_tuning_options(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run", "--auto-tune-system", "--tune-system", "linux-training-safe", "--", "python", "train.py"])

    def test_run_command_requires_workload(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run"])


if __name__ == "__main__":
    unittest.main()
