from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotune.cost_model.predict import synthetic_predict
from autotune.profiler.benchmark_runner import synthetic_profile
from autotune.tuner.cost_model_search import run_cost_model_search
from autotune.tuner.exhaustive_search import run_exhaustive_search
from autotune.tuner.objective import Objective
from autotune.tuner.random_search import run_random_search
from autotune.tuner.search_space import build_search_space
from autotune.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--search", choices=["exhaustive", "random", "cost_model"], default="cost_model")
    parser.add_argument("--trials", type=int, default=18)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.get("model")
    configs = build_search_space(config)
    objective = Objective.from_config(config)
    profile = lambda item: synthetic_profile(item, model_config)
    predict = lambda item: synthetic_predict(item, model_config)

    if args.search == "exhaustive":
        best, records = run_exhaustive_search(configs, profile, objective)
    elif args.search == "random":
        best, records = run_random_search(configs, profile, objective, args.trials)
    else:
        best, records = run_cost_model_search(configs, profile, predict, objective, args.trials)

    print(json.dumps({"best": best, "trials_used": len(records), "total_configs": len(configs)}, indent=2))


if __name__ == "__main__":
    main()
