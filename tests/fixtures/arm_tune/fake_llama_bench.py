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

HELP_FULL = """fake llama-bench fixture
  -m, --model FILE
  -t, --threads N
  --threads-batch N
  -p, --n-prompt N
  -n, --n-gen N
  -r, --repetitions N
  --batch-size N
  --ubatch-size N
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
        help_mode = os.environ.get("FAKE_LLAMA_BENCH_HELP_MODE")
        if help_mode == "missing_generation":
            print(HELP_MISSING_GENERATION)
        elif help_mode == "full":
            print(HELP_FULL)
        else:
            print(HELP)
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
    print(_valid_output(args))
    return 0


def _valid_output(args: list[str]) -> str:
    if os.environ.get("FAKE_LLAMA_BENCH_SCALE_BY_THREADS") != "1":
        return VALID_OUTPUT
    threads = _arg_int(args, "-t", "--threads", default=1)
    prefill = 300.0 + threads * 10.0
    decode = 80.0 + threads * 2.0
    return f"""warning: fixture output, not a measured Arm run
| model | threads | test | t/s |
| --- | --- | --- | --- |
| fake-q4 | {threads} | pp128 | {prefill:.2f} +/- 1.0 |
| fake-q4 | {threads} | tg64 | {decode:.2f} +/- 0.5 |
"""


def _arg_int(args: list[str], short: str, long: str, *, default: int) -> int:
    for index, token in enumerate(args):
        if token in {short, long} and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except ValueError:
                return default
    return default


if __name__ == "__main__":
    raise SystemExit(main())
