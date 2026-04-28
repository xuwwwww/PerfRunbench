from __future__ import annotations

from collections.abc import Callable

from autotune.tuner.objective import Objective, satisfies_constraints, score
from autotune.tuner.search_space import InferenceConfig


ProfileFn = Callable[[InferenceConfig], dict]
PredictFn = Callable[[InferenceConfig], dict]


def run_cost_model_search(
    configs: list[InferenceConfig],
    profile: ProfileFn,
    predict: PredictFn,
    objective: Objective,
    trials: int,
) -> tuple[dict, list[dict]]:
    predicted = [predict(config) for config in configs]
    predicted_feasible = [record for record in predicted if satisfies_constraints(record, objective)]
    ranked = sorted(predicted_feasible or predicted, key=lambda record: score(record, objective))
    selected_keys = {
        (
            record["backend"],
            record["batch_size"],
            record["thread_count"],
            record["precision"],
            record["graph_optimization"],
        )
        for record in ranked[:trials]
    }
    selected = [
        config
        for config in configs
        if (
            config.backend,
            config.batch_size,
            config.thread_count,
            config.precision,
            config.graph_optimization,
        )
        in selected_keys
    ]
    records = [profile(config) for config in selected]
    feasible = [record for record in records if satisfies_constraints(record, objective)]
    candidates = feasible or records
    best = min(candidates, key=lambda record: score(record, objective))
    return best, records

