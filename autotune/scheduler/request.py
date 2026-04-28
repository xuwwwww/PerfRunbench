from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_time: float
    deadline: float
    service_time: float
    input_size: int = 1


@dataclass(frozen=True)
class CompletedRequest:
    request: Request
    start_time: float
    finish_time: float

    @property
    def latency(self) -> float:
        return self.finish_time - self.request.arrival_time

    @property
    def missed_deadline(self) -> bool:
        return self.finish_time > self.request.deadline

