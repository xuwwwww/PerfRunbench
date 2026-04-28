from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimerResult:
    elapsed_seconds: float = 0.0


@contextmanager
def timer() -> Iterator[TimerResult]:
    result = TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start

