from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import load_manifest
from autotune.source_tuner.tuned_runner import SourceEdit, run_tuned_with_budget


class TunedRunnerTest(unittest.TestCase):
    def test_run_tuned_with_budget_auto_restores_source_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "train.py"
            target.write_text("batch_size = 64\n", encoding="utf-8")
            return_code, run_dir = run_tuned_with_budget(
                ["python", "-c", "print('training')"],
                [SourceEdit(str(target), "batch_size = 64", "batch_size = 16")],
                ResourceBudget(memory_budget_gb=1),
                sample_interval_seconds=0.05,
            )
            self.assertEqual(return_code, 0)
            self.assertEqual(target.read_text(encoding="utf-8"), "batch_size = 64\n")
            manifest = load_manifest(run_dir)
            self.assertEqual(len(manifest["changed_files"]), 1)
            self.assertTrue(any("auto_restored_files" in note for note in manifest["notes"]))


if __name__ == "__main__":
    unittest.main()
