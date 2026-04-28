from __future__ import annotations

from autotune.scheduler.request import CompletedRequest, Request


def schedule_static_batching(requests: list[Request], batch_size: int = 4) -> list[CompletedRequest]:
    ordered = sorted(requests, key=lambda item: item.arrival_time)
    current_time = 0.0
    completed: list[CompletedRequest] = []
    for index in range(0, len(ordered), batch_size):
        batch = ordered[index : index + batch_size]
        start = max(current_time, max(request.arrival_time for request in batch))
        service_time = max(request.service_time for request in batch) * (0.72 + 0.28 / len(batch))
        finish = start + service_time
        completed.extend(CompletedRequest(request, start, finish) for request in batch)
        current_time = finish
    return completed

