from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from autotune.cli import main


class GoldenPathDoctorTest(unittest.TestCase):
    def test_valid_generic_fixture_reports_ready_without_running_workload(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _write_generic_config(root, script)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--config", str(config), "--", sys.executable, str(script), "--unicode", "測試 value"])

            self.assertEqual(code, 0)
            self.assertIn("PerfRunbench Doctor", output.getvalue())
            self.assertIn("Ready: yes", output.getvalue())
            self.assertFalse((root / "executed.marker").exists())
            self.assertFalse((root / ".autotuneai").exists())

    def test_json_output_is_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _write_generic_config(root, script)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--json", "--config", str(config), "--", sys.executable, str(script)])

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["ready"])
            self.assertEqual(payload["kind"], "perfrunbench_doctor_result")
            self.assertTrue(any(check["name"] == "Cache match" for check in payload["checks"]))

    def test_pytorch_available_reports_cpu_pass(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _write_training_config(root, script)

            with patch(
                "autotune.golden_path.doctor._probe_python_torch",
                return_value={"status": "pass", "torch_version": "2.0.0", "cuda_available": False},
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["doctor", "--json", "--config", str(config), "--", sys.executable, str(script)])

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            torch_check = _check(payload, "PyTorch")
            self.assertEqual(torch_check["status"], "pass")
            self.assertIn("CPU", torch_check["message"])

    def test_pytorch_unavailable_is_warning_not_base_dependency_failure(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _write_training_config(root, script)

            with patch(
                "autotune.golden_path.doctor._probe_python_torch",
                return_value={"status": "warning", "error": "No module named torch"},
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(["doctor", "--json", "--config", str(config), "--", sys.executable, str(script)])

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["ready"])
            self.assertEqual(_check(payload, "PyTorch")["status"], "warning")

    def test_missing_executable_fails_without_stack_trace(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _write_generic_config(root, script)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--config", str(config), "--", "perfrunbench-missing-executable"])

            self.assertEqual(code, 1)
            self.assertIn("Executable could not be resolved", output.getvalue())
            self.assertNotIn("Traceback", output.getvalue())

    def test_invalid_metric_source_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _generic_config(root, script)
            config["metrics"]["source"] = "private_socket"
            path = _write_json(root / "autotune.json", config)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--json", "--config", str(path), "--", sys.executable, str(script)])

            self.assertEqual(code, 1)
            self.assertEqual(_check(json.loads(output.getvalue()), "Metric contract")["status"], "fail")

    def test_missing_bounded_slice_fails_pytorch_readiness(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _training_config(root, script)
            config["training"]["slice"]["mode"] = "none"
            path = _write_json(root / "autotune.json", config)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--json", "--config", str(path), "--", sys.executable, str(script)])

            self.assertEqual(code, 1)
            self.assertEqual(_check(json.loads(output.getvalue()), "Bounded training slice")["status"], "fail")

    def test_invalid_effective_batch_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _training_config(root, script)
            config["training"]["knobs"][0]["baseline"] = 0
            path = _write_json(root / "autotune.json", config)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--json", "--config", str(path), "--", sys.executable, str(script)])

            self.assertEqual(code, 1)
            self.assertEqual(_check(json.loads(output.getvalue()), "Effective batch")["status"], "fail")

    def test_unwritable_cache_directory_fails_when_path_is_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_marker_workload(root)
            config = _generic_config(root, script)
            cache_file = root / "cache-is-file"
            cache_file.write_text("not a directory", encoding="utf-8")
            config["cache"]["root"] = str(cache_file)
            path = _write_json(root / "autotune.json", config)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["doctor", "--json", "--config", str(path), "--", sys.executable, str(script)])

            self.assertEqual(code, 1)
            self.assertEqual(_check(json.loads(output.getvalue()), "Cache directory")["status"], "fail")


def _check(payload: dict, name: str) -> dict:
    return next(check for check in payload["checks"] if check["name"] == name)


def _write_marker_workload(root: Path) -> Path:
    script = root / "train.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "Path('executed.marker').write_text('ran', encoding='utf-8')",
                "print('{\"metrics\":{\"throughput\":1.0,\"samples_per_second\":1.0,\"completed_steps\":1}}')",
            ]
        ),
        encoding="utf-8",
    )
    return script


def _write_generic_config(root: Path, script: Path) -> Path:
    return _write_json(root / "autotune.json", _generic_config(root, script))


def _generic_config(root: Path, script: Path) -> dict:
    return {
        "schema_version": "1.0",
        "workload": {"adapter": "generic-command", "cwd": str(root), "timeout_seconds": 10, "identity_files": [script.name]},
        "objective": {"metric": "throughput", "direction": "maximize"},
        "metrics": {"source": "stdout_json_prefix", "prefix": "PERFRUNBENCH_METRIC "},
        "tuning": {
            "preset": "quick",
            "warmup_runs": 0,
            "search_runs": 1,
            "confirmation_runs": 2,
            "min_valid_trials": 2,
            "noise_band_percent": 1.0,
            "shuffle_seed": 7,
            "max_candidates": 1,
            "candidates": [{"id": "candidate", "env": {"PERFRUNBENCH_FAKE_GAIN": "1"}}],
        },
        "cache": {"root": str(root / "recommendations"), "retune_policy": "on_hard_drift_or_missing", "canary_policy": "on_new_or_drift"},
        "apply": {"canary_runs": 1, "lease_minutes": 1},
    }


def _write_training_config(root: Path, script: Path) -> Path:
    return _write_json(root / "autotune.json", _training_config(root, script))


def _training_config(root: Path, script: Path) -> dict:
    return {
        "schema_version": "1.0",
        "workload": {"adapter": "pytorch-training", "cwd": str(root), "timeout_seconds": 30, "identity_files": [script.name]},
        "objective": {"metric": "samples_per_second", "direction": "maximize"},
        "metrics": {"source": "stdout_json_prefix", "prefix": "AUTOTUNEAI_TRAIN_RESULT "},
        "training": {
            "schema_version": "1.0",
            "world_size": 1,
            "slice": {
                "mode": "cli_argument",
                "flag": "--max-steps",
                "warmup_steps": 1,
                "measured_steps": 4,
                "minimum_completed_steps": 4,
                "max_wall_time_seconds": 30,
                "omit_from_full_run": True,
            },
            "progress": {
                "source": "stdout_json_prefix",
                "step_prefix": "AUTOTUNEAI_TRAIN_STEP ",
                "result_prefix": "AUTOTUNEAI_TRAIN_RESULT ",
            },
            "semantics": {"effective_batch_policy": "preserve", "semantic_change_allowed": False},
            "capabilities": {"amp_modes": ["disabled"], "compile_modes": ["disabled"], "cpu_only": True},
            "search": {"max_candidates": 4, "shuffle_seed": 7, "stages": ["batch_config", "cpu_runtime"]},
            "knobs": [
                {"name": "micro_batch_size", "baseline": 4, "candidates": [4, 8], "binding": {"type": "cli_argument", "flag": "--micro-batch-size"}},
                {"name": "gradient_accumulation", "baseline": 2, "candidates": [2, 1], "binding": {"type": "cli_argument", "flag": "--gradient-accumulation"}},
                {"name": "cpu_threads", "baseline": 1, "candidates": [1, 2], "binding": {"type": "env_var", "name": "OMP_NUM_THREADS"}},
            ],
        },
        "tuning": {
            "preset": "quick",
            "warmup_runs": 0,
            "search_runs": 1,
            "confirmation_runs": 2,
            "min_valid_trials": 2,
            "noise_band_percent": 2.0,
            "shuffle_seed": 7,
            "max_candidates": 4,
        },
        "cache": {"root": str(root / "recommendations"), "retune_policy": "on_hard_drift_or_missing", "canary_policy": "on_new_or_drift"},
        "apply": {"canary_runs": 1, "lease_minutes": 1},
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
