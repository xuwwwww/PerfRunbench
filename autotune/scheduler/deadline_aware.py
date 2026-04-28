from __future__ import annotations

from autotune.scheduler.request import CompletedRequest, Request


def schedule_deadline_aware(
    requests: list[Request],
    max_batch_size: int = 8,
    max_wait_time: float = 8.0,
) -> list[CompletedRequest]:
    queue = sorted(requests, key=lambda item: item.arrival_time)
    current_time = 0.0
    completed: list[CompletedRequest] = []
    while queue:
        available = [request for request in queue if request.arrival_time <= current_time]
        if not available:
            current_time = queue[0].arrival_time
            available = [queue[0]]
        anchor = min(available, key=lambda item: item.deadline)
        window_end = min(anchor.arrival_time + max_wait_time, anchor.deadline)
        candidates = [request for request in queue if request.arrival_time <= window_end]
        batch = sorted(candidates, key=lambda item: item.deadline)[:max_batch_size]
        start = max(current_time, max(request.arrival_time for request in batch))
        service_time = max(request.service_time for request in batch) * (0.68 + 0.32 / len(batch))
        finish = start + service_time
        completed.extend(CompletedRequest(request, start, finish) for request in batch)
        batch_ids = {request.request_id for request in batch}
        queue = [request for request in queue if request.request_id not in batch_ids]
        current_time = finish
    return completed

