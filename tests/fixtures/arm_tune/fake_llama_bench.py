from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


HELP = """fake llama-bench fixture
  -m, --model FILE
  -t, --threads N
  -p, --n-prompt N
  -n, --n-gen N
  -r, --repetitions N
"""

HELP_MISSING_GENERATION = """fake llama-bench fixture
  -m, --model FILE
  -t, --threads N
  -p, --n-prompt N
  -r, --repetitions N
"""

VALID_OUTPUT = """warning: fixture output, not a measured Arm run
| model | threads | test | t/s |
| --- | --- | --- | --- |
| fake-q4 | 4 | pp128 | 321.50 +/- 1.0 |
| fake-q4 | 4 | tg64 | 87.25 +/- 0.5 |
"""

MALFORMED_OUTPUT = """| model | threads | test | t/s |
| --- | --- | --- | --- |
| fake-q4 | 4 | pp128 | not-a-number |
"""

NO_METRICS_OUTPUT = "fixture completed without a metrics table\n"

PARTIAL_OUTPUT = """test t/s
pp128 321.50
"""


def main() -> int:
    args = sys.argv[1:]
    record_path = os.environ.get("FAKE_LLAMA_BENCH_ARGS_PATH")
    if record_path:
        Path(record_path).write_text(json.dumps(args), encoding="utf-8")
    if "--version" in args:
        print("fake llama-bench fixture version 1.0")
        return 0
    if "--help" in args:
        print(HELP_MISSING_GENERATION if os.environ.get("FAKE_LLAMA_BENCH_HELP_MODE") == "missing_generation" else HELP)
        return 0
    mode = os.environ.get("FAKE_LLAMA_BENCH_MODE", "valid")
    if mode == "timeout":
        time.sleep(30)
        return 0
    if mode == "nonzero":
        print("fixture benchmark failed", file=sys.stderr)
        return 42
    if mode == "malformed":
        print(MALFORMED_OUTPUT)
        return 0
    if mode == "no_metrics":
        print(NO_METRICS_OUTPUT)
        return 0
    if mode == "partial":
        print(PARTIAL_OUTPUT)
        return 0
    print(VALID_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
