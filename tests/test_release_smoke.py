from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_release_smoke_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "release_smoke.py"
    spec = importlib.util.spec_from_file_location("perfrunbench_release_smoke_script", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load release_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseSmokeTest(unittest.TestCase):
    def test_cache_hit_accepts_exact_match_and_soft_drift_only(self) -> None:
        release_smoke = _load_release_smoke_module()

        self.assertTrue(release_smoke._is_reused_cache_state("exact_match"))
        self.assertTrue(release_smoke._is_reused_cache_state("soft_drift"))
        self.assertFalse(release_smoke._is_reused_cache_state("missing"))
        self.assertFalse(release_smoke._is_reused_cache_state("hard_drift"))
        self.assertFalse(release_smoke._is_reused_cache_state(None))


if __name__ == "__main__":
    unittest.main()
