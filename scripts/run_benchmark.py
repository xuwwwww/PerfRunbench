from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.profiler.benchmark_runner import filter_real_configs, real_profile, synthetic_profile, write_records
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
    args = parser.parse_args()

    config = load_config(args.config)
    configs = build_search_space(config)
    if args.backends:
        requested = set(args.backends)
        configs = [item for item in configs if item.backend in requested]
    profiler_config = config.get("profiler", {})
    if args.mode == "real":
        configs = filter_real_configs(configs)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    if args.mode == "real":
        model_cache: dict = {}
        records = [real_profile(item, config.get("model", {}), profiler_config, model_cache) for item in configs]
    else:
        records = [synthetic_profile(item, config.get("model")) for item in configs]

    output = args.output or profiler_config.get("output", "results/raw/profile.json")
    csv_output = args.csv_output or profiler_config.get("csv_output")
    write_records(records, output, csv_output)
    print(f"Wrote {len(records)} {args.mode} profiling records to {output}")
    if csv_output:
        print(f"Wrote CSV profiling records to {csv_output}")


if __name__ == "__main__":
    main()
