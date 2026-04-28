from __future__ import annotations

from autotune.scheduler.request import CompletedRequest, Request


def schedule_fcfs(requests: list[Request]) -> list[CompletedRequest]:
    current_time = 0.0
    completed: list[CompletedRequest] = []
    for request in sorted(requests, key=lambda item: item.arrival_time):
        start = max(current_time, request.arrival_time)
        finish = start + request.service_time
        completed.append(CompletedRequest(request, start, finish))
        current_time = finish
    return completed

