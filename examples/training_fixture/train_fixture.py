from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "autotune" / "training_metrics.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))
from autotune.training_metrics import emit_result, emit_step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--mode", default=os.environ.get("AUTOTUNEAI_TRAINING_FIXTURE_MODE", "ok"))
    args = parser.parse_args()
    if args.mode == "oom":
        sys.stderr.write("RuntimeError: CUDA out of memory while allocating fixture tensor\n")
        sys.stderr.flush()
        return 9
    if args.mode == "stall":
        time.sleep(0.01)
        return 0
    steps = int(args.max_steps or 3)
    if steps <= 0:
        return 2
    cpu_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    base = 100.0
    throughput = base + (args.micro_batch_size * 10.0) + (args.num_workers * 4.0) + (cpu_threads * 3.0)
    if args.gradient_accumulation == 1:
        throughput += 8.0
    step_time_ms = 1000.0 / throughput
    warmup_steps = 1 if args.max_steps else 0
    measured_steps = max(1, steps - warmup_steps)
    for step in range(1, steps + 1):
        emit_step(completed_steps=step, metrics={"step_time_ms": step_time_ms})
    emit_result(
        completed_steps=measured_steps,
        metrics={
            "completed_steps": measured_steps,
            "samples_per_second": throughput,
            "step_time_ms": step_time_ms,
            "final_loss": max(0.01, 1.0 / args.micro_batch_size),
            "examples_processed": measured_steps * args.micro_batch_size,
            "peak_memory_mb": 128.0 + args.micro_batch_size,
            "slice_limit_applied": args.max_steps is not None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
