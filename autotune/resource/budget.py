from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class ResourceBudget:
    memory_budget_gb: float | None = None
    reserve_memory_gb: float = 0.0
    reserve_cores: int = 0
    cpu_quota_percent: float | None = None
    enforce: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "ResourceBudget":
        raw = config.get("resource_budget", {})
        return cls(
            memory_budget_gb=_optional_float(raw.get("memory_budget_gb")),
            reserve_memory_gb=float(raw.get("reserve_memory_gb", 0.0)),
            reserve_cores=int(raw.get("reserve_cores", 0)),
            cpu_quota_percent=_optional_float(raw.get("cpu_quota_percent")),
            enforce=bool(raw.get("enforce", True)),
        )

    @property
    def enabled(self) -> bool:
        return any(
            [
                self.memory_budget_gb is not None,
                self.reserve_memory_gb > 0,
                self.reserve_cores > 0,
                self.cpu_quota_percent is not None,
            ]
        )

    @property
    def memory_budget_mb(self) -> float | None:
        if self.memory_budget_gb is None:
            return None
        return self.memory_budget_gb * 1024.0

    @property
    def memory_budget_mode(self) -> str:
        if self.memory_budget_gb is None:
            return "none"
        if self.memory_budget_gb < 0:
            return "reserve_to_full"
        return "absolute"

    def effective_memory_budget_mb(self, total_memory_mb: float | None) -> float | None:
        explicit_budget = self._explicit_memory_budget_mb(total_memory_mb)
        system_budget = None
        if total_memory_mb is not None:
            system_budget = max(0.0, total_memory_mb - self.reserve_memory_gb * 1024.0)
        if explicit_budget is None:
            return system_budget
        if system_budget is None:
            return explicit_budget
        return min(explicit_budget, system_budget)

    def allowed_threads(self, total_cores: int | None) -> int | None:
        if not total_cores:
            return None
        allowed = total_cores
        if self.reserve_cores > 0:
            allowed = min(allowed, max(1, total_cores - self.reserve_cores))
        if self.cpu_quota_percent is not None:
            quota_limited = max(1, floor(total_cores * (self.cpu_quota_percent / 100.0)))
            allowed = min(allowed, quota_limited)
        return max(1, allowed)

    def to_record(self, total_cores: int | None = None, total_memory_mb: float | None = None) -> dict:
        effective_memory_budget = self.effective_memory_budget_mb(total_memory_mb)
        return {
            "memory_budget_gb": self.memory_budget_gb,
            "memory_budget_mode": self.memory_budget_mode,
            "memory_budget_mb": round(self.memory_budget_mb, 3) if self.memory_budget_mb is not None else None,
            "effective_memory_budget_mb": round(effective_memory_budget, 3)
            if effective_memory_budget is not None
            else None,
            "reserve_memory_mb": round(self.reserve_memory_gb * 1024.0, 3),
            "reserve_cores": self.reserve_cores,
            "cpu_quota_percent": self.cpu_quota_percent,
            "allowed_threads": self.allowed_threads(total_cores),
            "resource_budget_enforced": self.enforce,
        }

    def _explicit_memory_budget_mb(self, total_memory_mb: float | None) -> float | None:
        if self.memory_budget_gb is None:
            return None
        requested_mb = self.memory_budget_gb * 1024.0
        if requested_mb >= 0:
            return requested_mb
        if total_memory_mb is None:
            return None
        return max(0.0, total_memory_mb - abs(requested_mb))


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
