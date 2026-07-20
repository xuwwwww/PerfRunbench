from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "autotune" / "training_metrics.py").exists():
    sys.path.insert(0, str(_REPO_ROOT))
from autotune.training_metrics import emit_result, emit_step


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("SKIP: torch is not installed")
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    start = time.perf_counter()
    final_loss = 0.0
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation):
            x = torch.randn(args.micro_batch_size, 32)
            y = torch.randint(0, 4, (args.micro_batch_size,))
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            final_loss = float(loss.detach().item())
        optimizer.step()
        emit_step(completed_steps=step, metrics={"final_loss": final_loss})
    elapsed = max(time.perf_counter() - start, sys.float_info.epsilon)
    examples = args.max_steps * args.micro_batch_size * args.gradient_accumulation
    emit_result(
        completed_steps=args.max_steps,
        metrics={
            "completed_steps": args.max_steps,
            "samples_per_second": examples / elapsed,
            "step_time_ms": (elapsed / args.max_steps) * 1000.0,
            "final_loss": final_loss,
            "examples_processed": examples,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
