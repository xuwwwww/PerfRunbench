from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SearchBudget:
    limit_seconds: float
    started_at: float = field(default_factory=time.monotonic)
    exhausted: bool = False
    skipped_work: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_minutes(cls, minutes: float) -> SearchBudget:
        return cls(limit_seconds=max(0.001, float(minutes) * 60.0))

    @property
    def deadline(self) -> float:
        return self.started_at + self.limit_seconds

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def can_start(self, required_seconds: float, *, phase: str, candidate_id: str) -> bool:
        required = max(0.001, float(required_seconds))
        remaining = self.remaining_seconds()
        if remaining + 0.001 >= required:
            return True
        self.exhausted = True
        self.skipped_work.append(
            {
                "phase": phase,
                "candidate_id": candidate_id,
                "reason": "insufficient_remaining_budget",
                "required_seconds": round(required, 6),
                "remaining_seconds": round(remaining, 6),
            }
        )
        return False

    def bounded_timeout(self, requested_seconds: float) -> float:
        return max(0.001, min(float(requested_seconds), self.remaining_seconds()))

    def cooldown(self, requested_seconds: float) -> None:
        seconds = min(max(0.0, float(requested_seconds)), self.remaining_seconds())
        if seconds > 0:
            time.sleep(seconds)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "limit_seconds": round(self.limit_seconds, 6),
            "elapsed_seconds": round(self.elapsed_seconds(), 6),
            "remaining_seconds": round(self.remaining_seconds(), 6),
            "status": "exhausted" if self.exhausted else "completed",
            "skipped_work": list(self.skipped_work),
        }
