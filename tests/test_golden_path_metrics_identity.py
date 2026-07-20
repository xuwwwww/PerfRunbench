from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from autotune.golden_path.identity import build_workload_identity, redact_argv
from autotune.golden_path.metrics import MetricSourceError, load_metrics_for_run, objective_value
from autotune.golden_path.config import load_golden_path_config


class GoldenPathMetricsIdentityTest(unittest.TestCase):
    def test_file_json_jsonl_and_stdout_prefix_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "metrics.json").write_text(json.dumps({"metrics": {"throughput": 12.5}}), encoding="utf-8")
            config = _config({"source": "json_file", "path": "metrics.json", "base": "run_dir"})
            self.assertEqual(load_metrics_for_run(run_dir, config)["throughput"], 12.5)

            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"metrics": {"throughput": 10.0}}) + "\n" + json.dumps({"throughput": 15.0}) + "\n",
                encoding="utf-8",
            )
            config = _config({"source": "jsonl_file", "path": "metrics.jsonl", "base": "run_dir"})
            self.assertEqual(load_metrics_for_run(run_dir, config)["throughput"], 15.0)

            (run_dir / "stdout.txt").write_text('log\nPERFRUNBENCH_METRIC {"metrics": {"throughput": 20.0}}\n', encoding="utf-8")
            config = _config({"source": "stdout_json_prefix", "prefix": "PERFRUNBENCH_METRIC "})
            self.assertEqual(load_metrics_for_run(run_dir, config)["throughput"], 20.0)

    def test_stdout_prefix_requires_exactly_one_record(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "stdout.txt").write_text("PERF {}\nPERF {}\n", encoding="utf-8")
            with self.assertRaises(MetricSourceError):
                load_metrics_for_run(run_dir, _config({"source": "stdout_json_prefix", "prefix": "PERF "}))

    def test_objective_value_rejects_nonfinite_values(self) -> None:
        self.assertEqual(objective_value({"throughput": 1.25}, "throughput"), 1.25)
        self.assertIsNone(objective_value({"throughput": "fast"}, "throughput"))
        self.assertIsNone(objective_value({"throughput": float("inf")}, "throughput"))

    def test_workload_identity_redacts_sensitive_argv_and_hashes_files(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            script = root / "train.py"
            config_file = root / "train.json"
            script.write_text("print('ok')\n", encoding="utf-8")
            config_file.write_text("{}", encoding="utf-8")
            config = {
                "schema_version": "1.0",
                "workload": {"adapter": "generic-command", "identity_files": [str(config_file)]},
                "objective": {"metric": "throughput", "direction": "maximize"},
                "metrics": {"source": "json_file", "path": "metrics.json"},
            }

            identity = build_workload_identity(
                [sys.executable, str(script), "--token", "secret-value"],
                config,
                config_path=config_file,
            )

            self.assertEqual(redact_argv(["python", "--password", "secret"])[2], "<redacted>")
            self.assertEqual(len(identity["script"]["sha256"]), 64)
            self.assertFalse(Path(identity["identity_files"][0]["display"]).is_absolute())
            self.assertNotIn("secret-value", json.dumps(identity))
            self.assertIn("digest", identity)

    def test_public_golden_path_examples_are_parseable_placeholders(self) -> None:
        for relative in [
            Path("configs/examples/run_command.yaml"),
            Path("configs/examples/tune_command.yaml"),
            Path("configs/examples/auto_command.yaml"),
        ]:
            content = relative.read_text(encoding="utf-8")
            payload = load_golden_path_config(relative)
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertIn("<workload_identity_file>", content)
            self.assertNotIn(".pem", content.lower())
            self.assertNotRegex(content, r"\b[A-Za-z]:\\")


def _config(metrics: dict) -> dict:
    return {
        "schema_version": "1.0",
        "objective": {"metric": "throughput", "direction": "maximize"},
        "metrics": metrics,
    }
