from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True)
class InferenceConfig:
    backend: str
    batch_size: int
    thread_count: int
    precision: str
    graph_optimization: str


def build_search_space(config: dict) -> list[InferenceConfig]:
    space = config["search_space"]
    return [
        InferenceConfig(
            backend=backend,
            batch_size=int(batch_size),
            thread_count=int(thread_count),
            precision=precision,
            graph_optimization=graph_optimization,
        )
        for backend, batch_size, thread_count, precision, graph_optimization in product(
            space["backends"],
            space["batch_sizes"],
            space["thread_counts"],
            space["precisions"],
            space["graph_optimizations"],
        )
    ]


def as_records(configs: Iterable[InferenceConfig]) -> list[dict]:
    return [config.__dict__ for config in configs]

