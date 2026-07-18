from __future__ import annotations

from pathlib import Path

from autotune.arm_tune.quality.suite import load_quality_suite


def load_boottrace_suite(path: str | Path):
    return load_quality_suite(path)
