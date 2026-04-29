from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.tuner.recommend import RecommendationRequest, recommend_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend the best safe configuration from benchmark records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--objective", choices=["throughput", "latency", "memory"], default="throughput")
    parser.add_argument("--latency-budget-ms", type=float)
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--allow-unsafe", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    request = RecommendationRequest(
        objective=args.objective,
        latency_budget_ms=args.latency_budget_ms,
        memory_budget_mb=args.memory_budget_gb * 1024.0 if args.memory_budget_gb is not None else None,
        require_safe=not args.allow_unsafe,
    )
    best, reasons = recommend_config(records, request)
    if args.json:
        print(json.dumps({"recommendation": best, "reasoning": reasons}, indent=2))
        return
    print("Recommended configuration")
    for key in ["backend", "batch_size", "thread_count", "precision", "graph_optimization"]:
        print(f"{key}: {best.get(key)}")
    print("")
    print("Measured performance")
    for key in ["latency_p95_ms", "throughput", "peak_rss_mb", "memory_mb", "average_process_cpu_percent"]:
        if key in best:
            print(f"{key}: {best.get(key)}")
    print("")
    print("Reasoning")
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()

