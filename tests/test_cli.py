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

    @patch("autotune.cli.platform.system", return_value="Linux")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_system_on_linux(self, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["run", "--auto-tune-system", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-training-safe")
        self.assertIn("Run directory", output.getvalue())

    @patch("autotune.cli.platform.system", return_value="Linux")
    @patch("autotune.cli.run_with_budget")
    def test_run_command_auto_tunes_memory_profile_when_budgeted(self, run_with_budget, _system) -> None:
        run_with_budget.return_value = (0, ".autotuneai/runs/run1")
        with redirect_stdout(io.StringIO()):
            code = main(["run", "--auto-tune-system", "--memory-budget-gb", "-3", "--", "python", "train.py"])
        self.assertEqual(code, 0)
        self.assertEqual(run_with_budget.call_args.kwargs["tune_system_profile"], "linux-memory-conservative")

    def test_run_command_rejects_conflicting_system_tuning_options(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run", "--auto-tune-system", "--tune-system", "linux-training-safe", "--", "python", "train.py"])

    def test_run_command_requires_workload(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run"])


if __name__ == "__main__":
    unittest.main()
