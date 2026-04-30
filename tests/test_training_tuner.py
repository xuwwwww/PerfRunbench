from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.training_tuner.batch_size import (
    find_batch_size_assignment,
    replace_assignment_value,
    tune_batch_size,
)


class TrainingTunerTest(unittest.TestCase):
    def test_find_and_replace_batch_size_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text("batch_size: 64\n", encoding="utf-8")
            value, line = find_batch_size_assignment(config, "batch_size")
            self.assertEqual(value, 64)
            self.assertEqual(replace_assignment_value(line, 16), "batch_size: 16")

    def test_tune_batch_size_recommends_largest_safe_value_and_restores_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "train.yaml"
            output = root / "summary.json"
            config.write_text("batch_size: 64\n", encoding="utf-8")
            result = tune_batch_size(
                config,
                "batch_size",
                [32, 16],
                ["python", "-c", "print('train')"],
                ResourceBudget(memory_budget_gb=1),
                output,
                sample_interval_seconds=0.05,
            )
            self.assertEqual(result["recommended_batch_size"], 32)
            self.assertEqual(config.read_text(encoding="utf-8"), "batch_size: 64\n")
            self.assertTrue(output.exists())
            self.assertEqual(len(result["trials"]), 2)


if __name__ == "__main__":
    unittest.main()

