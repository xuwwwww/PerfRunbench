from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.training_tuner.multi_knob import parse_knob_specs, tune_training_knobs


class MultiKnobTunerTest(unittest.TestCase):
    def test_parse_knob_specs(self) -> None:
        specs = parse_knob_specs(["batch_size=128,64,32", "dataloader_workers=0,2,4"])
        self.assertEqual(specs["batch_size"], [128, 64, 32])
        self.assertEqual(specs["dataloader_workers"], [0, 2, 4])

    def test_tune_training_knobs_recommends_highest_throughput_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "train.yaml"
            output = root / "summary.json"
            config.write_text("batch_size: 16\ndataloader_workers: 1\n", encoding="utf-8")
            command = [
                "python",
                "-c",
                (
                    "import json, os, pathlib; "
                    f"text=pathlib.Path(r'{config}').read_text(encoding='utf-8').splitlines(); "
                    "values={line.split(':',1)[0].strip(): int(line.split(':',1)[1].strip()) for line in text if ':' in line}; "
                    "score=values['batch_size'] + values['dataloader_workers']; "
                    "payload={'samples_per_second': float(score), 'duration_seconds': float(100-score), 'final_accuracy': 0.9}; "
                    "pathlib.Path(os.environ['AUTOTUNEAI_RUN_DIR'], 'training_metrics.json').write_text(json.dumps(payload), encoding='utf-8')"
                ),
            ]
            result = tune_training_knobs(
                config,
                {"batch_size": [8, 32], "dataloader_workers": [0, 4]},
                command,
                ResourceBudget(memory_budget_gb=1),
                output,
                sample_interval_seconds=0.05,
            )

            self.assertEqual(result["final_recommendation"]["batch_size"], 32)
            self.assertEqual(result["final_recommendation"]["dataloader_workers"], 4)
            self.assertEqual(config.read_text(encoding="utf-8"), "batch_size: 16\ndataloader_workers: 1\n")
            self.assertTrue(output.exists())

    def test_tune_training_knobs_can_enforce_accuracy_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "train.yaml"
            output = root / "summary.json"
            config.write_text("batch_size: 16\n", encoding="utf-8")
            command = [
                "python",
                "-c",
                (
                    "import json, os, pathlib; "
                    f"text=pathlib.Path(r'{config}').read_text(encoding='utf-8'); "
                    "value=int(text.split(':',1)[1].strip()); "
                    "accuracy=0.95 if value == 16 else 0.5; "
                    "payload={'samples_per_second': float(value), 'duration_seconds': 1.0, 'final_accuracy': accuracy}; "
                    "pathlib.Path(os.environ['AUTOTUNEAI_RUN_DIR'], 'training_metrics.json').write_text(json.dumps(payload), encoding='utf-8')"
                ),
            ]
            result = tune_training_knobs(
                config,
                {"batch_size": [16, 32]},
                command,
                ResourceBudget(memory_budget_gb=1),
                output,
                min_final_accuracy=0.9,
                sample_interval_seconds=0.05,
            )

            self.assertEqual(result["final_recommendation"]["batch_size"], 16)


if __name__ == "__main__":
    unittest.main()
