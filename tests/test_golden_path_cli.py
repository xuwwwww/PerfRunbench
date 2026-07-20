from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from autotune.cli import main
from autotune.golden_path.cache import invalidate_cache_entry, lookup_cache
from autotune.golden_path.config import cache_root, load_golden_path_config
from autotune.golden_path.identity import build_workload_identity, environment_fingerprint_for_command


class GoldenPathCliTest(unittest.TestCase):
    def test_tune_creates_eligible_bundle_and_cache_entry(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            script = _write_fake_workload(temp_path)
            config = _write_config(temp_path)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["tune", "--config", str(config), "--", sys.executable, str(script)])

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "recommended")
            self.assertIn("recommendation_bundle_id", payload)
            self.assertTrue(Path(payload["performance_passport_path"]).exists())
            entries = list((temp_path / "recommendations").glob("*.json"))
            self.assertEqual(len(entries), 1)
            entry = json.loads(entries[0].read_text(encoding="utf-8"))
            self.assertEqual(entry["selected_configuration_id"], "fast")

    def test_run_best_applies_cached_recommendation_and_launches_workload(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            script = _write_fake_workload(temp_path)
            config = _write_config(temp_path)
            command = [sys.executable, str(script)]

            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["tune", "--config", str(config), "--", *command]), 0)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["run", "--best", "--config", str(config), "--", *command])

            self.assertEqual(code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["apply_result"]["status"], "applied")
            self.assertEqual(payload["apply_result"]["canary_status"], "passed")

    def test_run_auto_retunes_on_missing_cache_and_reuses_on_exact_match(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            script = _write_fake_workload(temp_path)
            config = _write_config(temp_path)
            command = [sys.executable, str(script)]

            first = io.StringIO()
            with redirect_stdout(first):
                first_code = main(["run", "--auto", "--config", str(config), "--", *command])
            self.assertEqual(first_code, 0)
            first_payload = json.loads(first.getvalue())
            self.assertTrue(first_payload["newly_tuned"])

            second = io.StringIO()
            with redirect_stdout(second):
                second_code = main(["run", "--auto", "--config", str(config), "--", *command])
            self.assertEqual(second_code, 0)
            second_payload = json.loads(second.getvalue())
            self.assertFalse(second_payload["newly_tuned"])
            self.assertEqual(second_payload["cache_state"], "exact_match")

    def test_cache_lookup_reports_revoked_and_hard_drift_states(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            script = _write_fake_workload(temp_path)
            config_path = _write_config(temp_path)
            command = [sys.executable, str(script)]

            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["tune", "--config", str(config_path), "--", *command]), 0)
            config = load_golden_path_config(config_path)
            root = cache_root(config)
            entry_path = next(root.glob("*.json"))
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            identity = build_workload_identity(command, config, config_path=config_path)
            fingerprint = environment_fingerprint_for_command(command, config, config_path=config_path)

            entry["environment_fingerprint"]["hard"]["architecture"] = "different-arch"
            entry_path.write_text(json.dumps(entry), encoding="utf-8")
            drift = lookup_cache(root=root, workload_identity=identity, environment_fingerprint=fingerprint)
            self.assertEqual(drift["state"], "hard_drift")

            # Restore the entry digest by invalidating through the cache API; lookup
            # must report a revoked matching entry instead of silently reusing it.
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["tune", "--config", str(config_path), "--", *command]), 0)
            fresh_entry = json.loads(next(path for path in root.glob("*.json") if path != entry_path).read_text(encoding="utf-8"))
            invalidate_cache_entry(root, fresh_entry["cache_id"])
            revoked = lookup_cache(root=root, workload_identity=identity, environment_fingerprint=fingerprint)
            self.assertEqual(revoked["state"], "revoked")

    def test_rejects_empty_golden_path_command(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            config = _write_config(Path(temp_dir))
            with self.assertRaises(SystemExit):
                main(["tune", "--config", str(config)])

    def test_cache_cli_list_show_invalidate_and_gc(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            script = _write_fake_workload(temp_path)
            config = _write_config(temp_path)
            command = [sys.executable, str(script)]

            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["tune", "--config", str(config), "--", *command]), 0)
            root = temp_path / "recommendations"
            cache_id = json.loads(next(root.glob("*.json")).read_text(encoding="utf-8"))["cache_id"]

            list_output = io.StringIO()
            with redirect_stdout(list_output):
                self.assertEqual(main(["cache", "--root", str(root), "list"]), 0)
            self.assertIn(cache_id, list_output.getvalue())

            show_output = io.StringIO()
            with redirect_stdout(show_output):
                self.assertEqual(main(["cache", "--root", str(root), "show", cache_id]), 0)
            self.assertEqual(json.loads(show_output.getvalue())["cache_id"], cache_id)

            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["cache", "--root", str(root), "invalidate", cache_id]), 0)
            gc_output = io.StringIO()
            with redirect_stdout(gc_output):
                self.assertEqual(main(["cache", "--root", str(root), "gc"]), 0)
            self.assertIn(f"{cache_id}.json", gc_output.getvalue())


def _write_fake_workload(root: Path) -> Path:
    script = root / "fake_workload.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "from pathlib import Path",
                "gain = float(os.environ.get('PERFRUNBENCH_FAKE_GAIN', '0'))",
                "metrics = {'throughput': 100.0 + gain, 'latency_ms': 10.0 - min(gain, 50.0) / 20.0}",
                "run_dir = os.environ.get('AUTOTUNEAI_RUN_DIR')",
                "if run_dir:",
                "    Path(run_dir, 'metrics.json').write_text(json.dumps({'metrics': metrics}), encoding='utf-8')",
                "print(json.dumps({'metrics': metrics}))",
            ]
        ),
        encoding="utf-8",
    )
    return script


def _write_config(root: Path) -> Path:
    config = root / "autotune.json"
    payload = {
        "schema_version": "1.0",
        "workload": {"adapter": "generic-command", "timeout_seconds": 20},
        "objective": {"metric": "throughput", "direction": "maximize"},
        "metrics": {"source": "json_file", "path": "metrics.json", "base": "run_dir"},
        "tuning": {
            "preset": "quick",
            "warmup_runs": 0,
            "search_runs": 1,
            "confirmation_runs": 2,
            "min_valid_trials": 2,
            "noise_band_percent": 1.0,
            "shuffle_seed": 7,
            "candidates": [
                {"id": "slow", "env": {"PERFRUNBENCH_FAKE_GAIN": "5"}},
                {"id": "fast", "env": {"PERFRUNBENCH_FAKE_GAIN": "35"}},
            ],
        },
        "cache": {
            "root": str(root / "recommendations"),
            "retune_policy": "on_hard_drift_or_missing",
            "canary_policy": "on_new_or_drift",
            "ttl_days": 7,
        },
        "apply": {"canary_runs": 1, "lease_minutes": 1},
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config
