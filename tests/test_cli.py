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

    def test_run_command_requires_workload(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run"])


if __name__ == "__main__":
    unittest.main()
