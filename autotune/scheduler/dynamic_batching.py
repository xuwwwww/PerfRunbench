from __future__ import annotations

from autotune.scheduler.request import CompletedRequest, Request


def schedule_dynamic_batching(
    requests: list[Request],
    max_batch_size: int = 8,
    max_wait_time: float = 8.0,
) -> list[CompletedRequest]:
    ordered = sorted(requests, key=lambda item: item.arrival_time)
    current_time = 0.0
    index = 0
    completed: list[CompletedRequest] = []
    while index < len(ordered):
        first = ordered[index]
        window_end = first.arrival_time + max_wait_time
        batch = []
        while index < len(ordered) and len(batch) < max_batch_size and ordered[index].arrival_time <= window_end:
            batch.append(ordered[index])
            index += 1
        start = max(current_time, max(request.arrival_time for request in batch))
        service_time = max(request.service_time for request in batch) * (0.68 + 0.32 / len(batch))
        finish = start + service_time
        completed.extend(CompletedRequest(request, start, finish) for request in batch)
        current_time = finish
    return completed

