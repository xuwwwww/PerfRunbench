from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.profiler.benchmark_runner import real_profile, write_records
from autotune.resource.affinity import apply_cpu_affinity, filter_thread_budget
from autotune.resource.budget import ResourceBudget
from autotune.tuner.search_space import InferenceConfig
from autotune.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact real CPU benchmark sweep.")
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--output", default="results/raw/resnet18_real_sweep.json")
    parser.add_argument("--csv-output", default="results/raw/resnet18_real_sweep.csv")
    parser.add_argument("--backends", nargs="+", default=["pytorch", "onnxruntime"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--graph-optimizations", nargs="+", default=["disable", "all"])
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--reserve-cores", type=int)
    parser.add_argument("--cpu-quota-percent", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    budget = _resource_budget_from_args(config, args)
    configs = _build_mini_space(args.backends, args.batch_sizes, args.threads, args.graph_optimizations)
    if budget.enabled and budget.enforce:
        configs = filter_thread_budget(configs, budget)
    if not configs:
        raise SystemExit("No configurations remain after resource-budget filtering.")

    model_cache: dict = {}
    resource_context = apply_cpu_affinity(budget)
    records = [
        real_profile(item, config.get("model", {}), config.get("profiler", {}), model_cache, budget, resource_context)
        for item in configs
    ]
    write_records(records, args.output, args.csv_output)
    print(f"Wrote {len(records)} real sweep records to {args.output}")
    print(f"Wrote CSV sweep records to {args.csv_output}")


def _build_mini_space(
    backends: list[str],
    batch_sizes: list[int],
    threads: list[int],
    graph_optimizations: list[str],
) -> list[InferenceConfig]:
    configs: list[InferenceConfig] = []
    for backend in backends:
        backend_graph_opts = graph_optimizations if backend == "onnxruntime" else ["disable"]
        for batch_size in batch_sizes:
            for thread_count in threads:
                for graph_optimization in backend_graph_opts:
                    configs.append(InferenceConfig(backend, batch_size, thread_count, "fp32", graph_optimization))
    return configs


def _resource_budget_from_args(config: dict, args: argparse.Namespace) -> ResourceBudget:
    budget = ResourceBudget.from_config(config)
    overrides = {}
    if args.memory_budget_gb is not None:
        overrides["memory_budget_gb"] = args.memory_budget_gb
    if args.reserve_cores is not None:
        overrides["reserve_cores"] = args.reserve_cores
    if args.cpu_quota_percent is not None:
        overrides["cpu_quota_percent"] = args.cpu_quota_percent
    if overrides:
        return replace(budget, **overrides)
    return budget


if __name__ == "__main__":
    main()

