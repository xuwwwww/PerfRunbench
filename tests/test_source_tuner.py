from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autotune.resource.run_state import load_manifest
from autotune.source_tuner.transaction import apply_find_replace


class SourceTunerTest(unittest.TestCase):
    def test_find_replace_apply_and_restore_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "train.py"
            target.write_text("batch_size = 64\n", encoding="utf-8")
            runs_dir = root / ".autotuneai" / "runs"
            result = apply_find_replace(
                target,
                "batch_size = 64",
                "batch_size = 16",
                apply=True,
                runs_dir=runs_dir,
            )
            self.assertTrue(result["applied"])
            self.assertEqual(target.read_text(encoding="utf-8"), "batch_size = 16\n")
            manifest = load_manifest(runs_dir / result["run_id"])
            self.assertEqual(len(manifest["changed_files"]), 1)
            backup = Path(manifest["changed_files"][0]["backup"])
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_text(encoding="utf-8"), "batch_size = 64\n")

    def test_find_replace_dry_run_does_not_change_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "train.py"
            target.write_text("batch_size = 64\n", encoding="utf-8")
            result = apply_find_replace(target, "batch_size = 64", "batch_size = 16")
            self.assertFalse(result["applied"])
            self.assertEqual(target.read_text(encoding="utf-8"), "batch_size = 64\n")


if __name__ == "__main__":
    unittest.main()

