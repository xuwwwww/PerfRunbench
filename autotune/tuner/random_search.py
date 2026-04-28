from __future__ import annotations

import random
from collections.abc import Callable

from autotune.tuner.objective import Objective, satisfies_constraints, score
from autotune.tuner.search_space import InferenceConfig


ProfileFn = Callable[[InferenceConfig], dict]


def run_random_search(
    configs: list[InferenceConfig],
    profile: ProfileFn,
    objective: Objective,
    trials: int,
    seed: int = 7,
) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    sampled = rng.sample(configs, k=min(trials, len(configs)))
    records = [profile(config) for config in sampled]
    feasible = [record for record in records if satisfies_constraints(record, objective)]
    candidates = feasible or records
    best = min(candidates, key=lambda record: score(record, objective))
    return best, records

