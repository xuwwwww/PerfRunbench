from __future__ import annotations

from collections.abc import Callable

from autotune.tuner.objective import Objective, satisfies_constraints, score
from autotune.tuner.search_space import InferenceConfig


ProfileFn = Callable[[InferenceConfig], dict]


def run_exhaustive_search(
    configs: list[InferenceConfig],
    profile: ProfileFn,
    objective: Objective,
) -> tuple[dict, list[dict]]:
    records = [profile(config) for config in configs]
    feasible = [record for record in records if satisfies_constraints(record, objective)]
    candidates = feasible or records
    best = min(candidates, key=lambda record: score(record, objective))
    return best, records

