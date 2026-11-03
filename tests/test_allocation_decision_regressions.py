from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.resource.workload_runner import launch_performance, run_with_budget
from autotune.recommendation.fingerprints import compare_environment_fingerprints
from autotune.golden_path.config import default_command_config
from autotune.golden_path.runner import tune_command_config


class CallerAffinityRegressionTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "sched_getaffinity"), "Linux affinity required")
    def test_local_runner_restores_caller_and_restricts_child(self) -> None:
        before = os.sched_getaffinity(0)
        if len(before) < 2:
            self.skipTest("requires at least two permitted CPUs")
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "child_mask.txt"
            command = [
                sys.executable, "-c",
                "import os,sys,pathlib; pathlib.Path(sys.argv[1]).write_text(','.join(map(str,sorted(os.sched_getaffinity(0)))))",
                str(result_path),
            ]
            try:
                return_code, _ = run_with_budget(
                    command,
                    ResourceBudget(reserve_cores=1),
                    sample_interval_seconds=0.05,
                )
                self.assertEqual(return_code, 0)
                child = {int(item) for item in result_path.read_text().split(",")}
                self.assertEqual(child, set(sorted(before)[:-1]))
                self.assertEqual(os.sched_getaffinity(0), before)

                result_path.unlink()
                return_code, _ = launch_performance(command, ResourceBudget(reserve_cores=1))
                self.assertEqual(return_code, 0)
                self.assertEqual(os.sched_getaffinity(0), before)
            finally:
                os.sched_setaffinity(0, before)

    @unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "sched_getaffinity"), "Linux affinity required")
    def test_spawn_error_restores_caller(self) -> None:
        before = os.sched_getaffinity(0)
        if len(before) < 2:
            self.skipTest("requires at least two permitted CPUs")
        try:
            with self.assertRaises(FileNotFoundError):
                run_with_budget(
                    ["/perfrunbench/nonexistent/executable"],
                    ResourceBudget(reserve_cores=1),
                )
        finally:
            self.assertEqual(os.sched_getaffinity(0), before)


class MemoryDriftRegressionTest(unittest.TestCase):
    def test_small_memory_changes_do_not_trigger_canary(self) -> None:
        hard = {"architecture": "aarch64"}
        before = {"hard": hard, "soft": {"available_memory_mb": 20000}}
        minor = {"hard": hard, "soft": {"available_memory_mb": 19800}}
        material = {"hard": hard, "soft": {"available_memory_mb": 17000}}
        self.assertEqual(compare_environment_fingerprints(before, minor)["status"], "match")
        self.assertEqual(compare_environment_fingerprints(before, material)["status"], "soft_mismatch")


class BaselineSelectionRegressionTest(unittest.TestCase):
    def test_baseline_selection_gets_fresh_validation_and_no_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = default_command_config(preset="quick")
            config["workload"]["cwd"] = temporary
            config["workload"]["timeout_seconds"] = 10
            config["cache"]["root"] = str(Path(temporary) / "cache")
            config["tuning"].update({
                "budget_minutes": 1,
                "warmup_runs": 0,
                "search_runs": 1,
                "confirmation_runs": 1,
                "min_valid_trials": 1,
                "max_candidates": 1,
                "candidates": [{"id": "deliberately_slow", "env": {"TEST_SLOW": "1"}}],
            })
            command = [
                sys.executable, "-c",
                "import os,time; time.sleep(0.10 if os.getenv('TEST_SLOW') else 0.005)",
            ]
            result = tune_command_config(config, command)
            self.assertEqual(result["selected_candidate_id"], "baseline")
            self.assertEqual(result["status"], "baseline_retained")
            self.assertGreaterEqual(result["statistical_evidence"]["baseline_confirmation_samples"], 1)
            self.assertNotIn("recommendation_bundle_id", result)


if __name__ == "__main__":
    unittest.main()
