from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.profiler.benchmark_runner import filter_real_configs, real_profile, synthetic_profile, write_records
from autotune.resource.affinity import apply_cpu_affinity, filter_thread_budget
from autotune.resource.budget import ResourceBudget
from autotune.tuner.search_space import build_search_space
from autotune.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--backends", nargs="+")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--output")
    parser.add_argument("--csv-output")
    parser.add_argument("--memory-budget-gb", type=float)
    parser.add_argument("--reserve-cores", type=int)
    parser.add_argument("--cpu-quota-percent", type=float)
    parser.add_argument("--ignore-resource-budget", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    budget = _resource_budget_from_args(config, args)
    configs = build_search_space(config)
    if args.backends:
        requested = set(args.backends)
        configs = [item for item in configs if item.backend in requested]
    profiler_config = config.get("profiler", {})
    if args.mode == "real":
        configs = filter_real_configs(configs)
        if budget.enabled and budget.enforce:
            configs = filter_thread_budget(configs, budget)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    if not configs:
        raise SystemExit("No configurations remain after backend, precision, and resource-budget filtering.")

    if args.mode == "real":
        model_cache: dict = {}
        resource_context = apply_cpu_affinity(budget)
        records = [
            real_profile(
                item,
                config.get("model", {}),
                profiler_config,
                model_cache,
                budget,
                resource_context,
            )
            for item in configs
        ]
    else:
        records = [synthetic_profile(item, config.get("model")) for item in configs]

    output = args.output or profiler_config.get("output", "results/raw/profile.json")
    csv_output = args.csv_output or profiler_config.get("csv_output")
    write_records(records, output, csv_output)
    print(f"Wrote {len(records)} {args.mode} profiling records to {output}")
    if csv_output:
        print(f"Wrote CSV profiling records to {csv_output}")


def _resource_budget_from_args(config: dict, args: argparse.Namespace) -> ResourceBudget:
    if args.ignore_resource_budget:
        return ResourceBudget(enforce=False)
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
