from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.training_tuner.batch_size import (
    find_batch_size_assignment,
    find_numeric_assignment,
    find_scalar_assignment,
    format_scalar_value,
    parse_scalar_value,
    replace_assignment_value,
    tune_batch_size,
    tune_numeric_config_key,
)
from autotune.training_tuner.multi_knob import parse_knob_specs


class TrainingTunerTest(unittest.TestCase):
    def test_find_and_replace_batch_size_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text("batch_size: 64\n", encoding="utf-8")
            value, line = find_batch_size_assignment(config, "batch_size")
            self.assertEqual(value, 64)
            self.assertEqual(replace_assignment_value(line, 16), "batch_size: 16")

    def test_find_numeric_assignment_supports_non_batch_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text("num_workers: 4\n", encoding="utf-8")
            value, line = find_numeric_assignment(config, "num_workers")
            self.assertEqual(value, 4)
            self.assertEqual(replace_assignment_value(line, 2), "num_workers: 2")

    def test_find_scalar_assignment_supports_bool_and_string_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "train.yaml"
            config.write_text("pin_memory: true\namp_dtype: bfloat16\n", encoding="utf-8")
            pin_memory, pin_line = find_scalar_assignment(config, "pin_memory")
            amp_dtype, amp_line = find_scalar_assignment(config, "amp_dtype")
            self.assertTrue(pin_memory)
            self.assertEqual(amp_dtype, "bfloat16")
            self.assertEqual(replace_assignment_value(pin_line, False), "pin_memory: false")
            self.assertEqual(replace_assignment_value(amp_line, "float16"), "amp_dtype: float16")

    def test_parse_scalar_value_and_knob_specs_support_mixed_values(self) -> None:
        self.assertEqual(parse_scalar_value("12"), 12)
        self.assertEqual(parse_scalar_value("0.5"), 0.5)
        self.assertTrue(parse_scalar_value("true"))
        self.assertEqual(parse_scalar_value("bf16"), "bf16")
        self.assertEqual(format_scalar_value(False), "false")
        knobs = parse_knob_specs(["pin_memory=true,false", "amp_dtype=float16,bfloat16", "prefetch_factor=2,4"])
        self.assertEqual(knobs["pin_memory"], [True, False])
        self.assertEqual(knobs["amp_dtype"], ["float16", "bfloat16"])
        self.assertEqual(knobs["prefetch_factor"], [2, 4])

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

    def test_tune_numeric_config_key_uses_generic_result_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "train.yaml"
            output = root / "summary.json"
            config.write_text("num_workers: 4\n", encoding="utf-8")
            result = tune_numeric_config_key(
                config,
                "num_workers",
                [0, 2],
                ["python", "-c", "print('train')"],
                ResourceBudget(memory_budget_gb=1),
                output,
                sample_interval_seconds=0.05,
            )
            self.assertEqual(result["recommended_value"], 2)
            self.assertEqual(result["trials"][0]["key"], "num_workers")
            self.assertEqual(config.read_text(encoding="utf-8"), "num_workers: 4\n")


if __name__ == "__main__":
    unittest.main()
