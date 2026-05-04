from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SLOW_MODULES = {
    "test_multi_knob_tuner",
    "test_training_tuner",
    "test_tuned_runner",
    "test_workload_runner",
    "tests.test_multi_knob_tuner",
    "tests.test_training_tuner",
    "tests.test_tuned_runner",
    "tests.test_workload_runner",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AutoTuneAI tests.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="Skip subprocess-heavy integration tests.")
    mode.add_argument("--all", action="store_true", help="Run the full unittest suite.")
    args = parser.parse_args(argv)

    loader = unittest.defaultTestLoader
    suite = loader.discover("tests")
    if args.fast:
        suite = _filter_suite(suite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def _filter_suite(suite: unittest.TestSuite) -> unittest.TestSuite:
    filtered = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            child = _filter_suite(item)
            if child.countTestCases():
                filtered.addTest(child)
            continue
        module = item.__class__.__module__
        if module not in SLOW_MODULES:
            filtered.addTest(item)
    return filtered


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
