from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.profiler.benchmark_runner import synthetic_profile, write_records
from autotune.tuner.search_space import build_search_space
from autotune.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    records = [synthetic_profile(item, config.get("model")) for item in build_search_space(config)]
    output = config.get("profiler", {}).get("output", "results/raw/profile.json")
    write_records(records, output)
    print(f"Wrote {len(records)} profiling records to {output}")


if __name__ == "__main__":
    main()
