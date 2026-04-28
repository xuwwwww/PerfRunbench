from __future__ import annotations

import random

from autotune.scheduler.request import Request


def generate_workload(pattern: str, count: int = 100, seed: int = 7) -> list[Request]:
    rng = random.Random(seed)
    spacing = {"low": 14.0, "medium": 7.0, "burst": 3.0}.get(pattern, 7.0)
    requests: list[Request] = []
    current = 0.0
    for request_id in range(count):
        if pattern == "burst" and request_id % 25 == 0:
            current += 1.0
        else:
            current += rng.expovariate(1.0 / spacing)
        service_time = rng.uniform(4.0, 10.0)
        deadline = current + rng.uniform(25.0, 60.0)
        requests.append(Request(request_id, current, deadline, service_time))
    return requests

