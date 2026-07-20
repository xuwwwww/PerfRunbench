from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from autotune.cli import main
from autotune.golden_path.config import load_golden_path_config
from autotune.golden_path.runner import tune_command
from autotune.training_adapter.bindings import build_candidate_plan
from autotune.training_adapter.candidates import generate_training_candidates
from autotune.training_adapter.config import validate_training_config
from autotune.training_adapter.errors import TrainingConfigError


class PyTorchTrainingAdapterTest(unittest.TestCase):
    def test_init_writes_valid_public_config_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            output = Path(temp_dir) / "autotune.yaml"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["init", "--adapter", "pytorch-training", "--output", str(output)]), 0)
            payload = load_golden_path_config(output)
            validate_training_config(payload)
            text = output.read_text(encoding="utf-8")
            self.assertIn("autotuneai run --auto --config", stdout.getvalue())
            self.assertNotIn(".pem", text.lower())
            self.assertNotRegex(text, r"\b[A-Za-z]:\\")
            with self.assertRaises(SystemExit):
                main(["init", "--adapter", "pytorch-training", "--output", str(output)])

    def test_schema_rejects_unknown_training_fields(self) -> None:
        config = _training_config(Path.cwd())
        config["training"]["private_shell_hook"] = "python mutate.py"
        with self.assertRaises(TrainingConfigError):
            validate_training_config(config)

    def test_public_example_config_is_parseable_placeholder(self) -> None:
        path = Path("configs/examples/pytorch_training.yaml")
        payload = load_golden_path_config(path)
        validate_training_config(payload)
        content = path.read_text(encoding="utf-8")
        self.assertNotIn(".pem", content.lower())
        self.assertNotRegex(content, r"\b[A-Za-z]:\\")

    def test_candidate_generation_preserves_effective_batch_and_stage_status(self) -> None:
        config = _training_config(Path.cwd())
        candidates, statuses = generate_training_candidates(config)
        self.assertEqual(candidates[0]["id"], "baseline")
        batch = next(item for item in candidates if item["id"] == "micro_batch_size_8_gradient_accumulation_1")
        self.assertEqual(batch["effective_batch_size"], batch["baseline_effective_batch_size"])
        self.assertTrue(batch["recommendation_eligible"])
        self.assertIn("batch_config", {item["stage"] for item in statuses})

    def test_config_overlay_binding_copies_candidate_config_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_fixture(root)
            settings = root / "train_settings.json"
            settings.write_text(json.dumps({"train": {"batch": 4}}), encoding="utf-8")
            config = _training_config(root)
            knob = {
                "name": "micro_batch_size",
                "baseline": 4,
                "candidates": [8],
                "binding": {"type": "json_config_path", "file": str(settings), "path": "train.batch", "argv_flag": "--config"},
            }
            candidate = {
                "id": "json_overlay",
                "stage": "batch_config",
                "knobs": [knob],
                "knob_values": {"micro_batch_size": 8},
                "recommendation_eligible": True,
            }
            plan = build_candidate_plan([sys.executable, str(script), "--config", str(settings)], config, candidate, plan_root=root / "plans")
            overlay = Path(plan.overlays[0]["overlay_relative_path"])
            overlay_path = root / "plans" / overlay
            self.assertEqual(json.loads(settings.read_text(encoding="utf-8"))["train"]["batch"], 4)
            self.assertEqual(json.loads(overlay_path.read_text(encoding="utf-8"))["train"]["batch"], 8)
            self.assertIn(str(overlay_path), plan.slice_argv)

    def test_tune_run_auto_cache_hit_and_run_best_use_training_bundle(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_fixture(root)
            config_path = _write_config(root)
            command = [sys.executable, str(script)]

            tuned = io.StringIO()
            with redirect_stdout(tuned):
                self.assertEqual(main(["tune", "--config", str(config_path), "--", *command]), 0)
            tuned_payload = json.loads(tuned.getvalue())
            self.assertEqual(tuned_payload["status"], "recommended")
            self.assertIn("recommendation_bundle_id", tuned_payload)

            auto = io.StringIO()
            with redirect_stdout(auto):
                self.assertEqual(main(["run", "--auto", "--config", str(config_path), "--", *command]), 0)
            auto_payload = json.loads(auto.getvalue())
            self.assertEqual(auto_payload["status"], "completed")
            stdout_path = Path(auto_payload["run_dir"]) / "stdout.txt"
            self.assertIn('"slice_limit_applied":false', stdout_path.read_text(encoding="utf-8"))

            best = io.StringIO()
            with redirect_stdout(best):
                self.assertEqual(main(["run", "--best", "--config", str(config_path), "--", *command]), 0)
            self.assertEqual(json.loads(best.getvalue())["status"], "completed")

    def test_cache_miss_auto_fails_without_bounded_slice(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_fixture(root)
            config = _training_config(root)
            config["training"]["slice"]["mode"] = "none"
            config_path = root / "autotune.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["run", "--auto", "--config", str(config_path), "--", sys.executable, str(script)])

            self.assertEqual(code, 1)
            self.assertIn("bounded slice", output.getvalue())

    def test_oom_and_stall_candidates_are_rejected_without_zero_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = _write_fixture(root)
            for mode, expected in [("oom", "out_of_memory"), ("stall", "stalled")]:
                config_path = _write_config(root, suffix=f"_{mode}")
                result = tune_command(config_path, [sys.executable, str(script), "--mode", mode])
                failures = [trial for trial in result["trials"] if trial.get("failed")]
                self.assertTrue(failures)
                self.assertTrue(any(trial["training_diagnostics"].get("failure_class") == expected for trial in failures))
                self.assertTrue(all(trial.get("objective_metric_value") is None for trial in failures))


def _write_fixture(root: Path) -> Path:
    script = root / "train_fixture.py"
    script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse, json, os, sys, time",
                "def emit(prefix, payload):",
                "    print(prefix + json.dumps(payload, sort_keys=True, separators=(',', ':')), flush=True)",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--micro-batch-size', type=int, default=4)",
                "parser.add_argument('--gradient-accumulation', type=int, default=2)",
                "parser.add_argument('--num-workers', type=int, default=0)",
                "parser.add_argument('--max-steps', type=int, default=None)",
                "parser.add_argument('--mode', default=os.environ.get('AUTOTUNEAI_TRAINING_FIXTURE_MODE', 'ok'))",
                "args = parser.parse_args()",
                "if args.mode == 'oom':",
                "    sys.stderr.write('RuntimeError: CUDA out of memory while allocating fixture tensor\\n')",
                "    raise SystemExit(9)",
                "if args.mode == 'stall':",
                "    time.sleep(0.01)",
                "    raise SystemExit(0)",
                "steps = int(args.max_steps or 3)",
                "cpu_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))",
                "throughput = 100.0 + args.micro_batch_size * 10.0 + args.num_workers * 4.0 + cpu_threads * 3.0",
                "if args.gradient_accumulation == 1:",
                "    throughput += 8.0",
                "measured_steps = max(1, steps - (1 if args.max_steps else 0))",
                "for step in range(1, steps + 1):",
                "    emit('AUTOTUNEAI_TRAIN_STEP ', {'schema_version':'1.0','completed_steps':step,'metrics':{'step_time_ms':1000.0/throughput}})",
                "emit('AUTOTUNEAI_TRAIN_RESULT ', {'schema_version':'1.0','completed_steps':measured_steps,'metrics':{'completed_steps':measured_steps,'samples_per_second':throughput,'step_time_ms':1000.0/throughput,'final_loss':0.1,'examples_processed':measured_steps*args.micro_batch_size,'slice_limit_applied':args.max_steps is not None}})",
            ]
        ),
        encoding="utf-8",
    )
    return script


def _write_config(root: Path, *, suffix: str = "") -> Path:
    config = root / f"autotune{suffix}.json"
    payload = _training_config(root)
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config


def _training_config(root: Path) -> dict:
    return {
        "schema_version": "1.0",
        "workload": {"adapter": "pytorch-training", "cwd": str(root), "timeout_seconds": 30, "identity_files": ["train_fixture.py"]},
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
            "search": {"max_candidates": 8, "shuffle_seed": 7, "stages": ["input_pipeline", "batch_config", "cpu_runtime"]},
            "knobs": [
                {
                    "name": "micro_batch_size",
                    "baseline": 4,
                    "candidates": [4, 8],
                    "binding": {"type": "cli_argument", "flag": "--micro-batch-size"},
                },
                {
                    "name": "gradient_accumulation",
                    "baseline": 2,
                    "candidates": [2, 1],
                    "binding": {"type": "cli_argument", "flag": "--gradient-accumulation"},
                },
                {
                    "name": "dataloader_workers",
                    "baseline": 0,
                    "candidates": [0, 2],
                    "binding": {"type": "cli_argument", "flag": "--num-workers"},
                },
                {
                    "name": "cpu_threads",
                    "baseline": 1,
                    "candidates": [1, 2],
                    "binding": {"type": "env_var", "name": "OMP_NUM_THREADS"},
                },
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
            "max_candidates": 8,
        },
        "cache": {
            "root": str(root / "recommendations"),
            "retune_policy": "on_hard_drift_or_missing",
            "canary_policy": "on_new_or_drift",
            "ttl_days": 7,
        },
        "apply": {"canary_runs": 1, "lease_minutes": 1},
    }


if __name__ == "__main__":
    unittest.main()
