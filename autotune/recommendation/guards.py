from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


GUARD_SCHEMA_VERSION = "1.0"
GUARD_STAGES = {"preflight", "candidate", "trial", "recommendation", "apply", "post_apply"}
GUARD_STATUSES = {"pass", "warn", "fail", "not_available"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class GuardResult:
    stage: str
    status: str
    code: str
    message: str
    evidence_refs: list[str] = field(default_factory=list)
    recoverable: bool = False
    timestamp: str = field(default_factory=utc_now)
    schema_version: str = GUARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.stage not in GUARD_STAGES:
            raise ValueError(f"unsupported guard stage: {self.stage}")
        if self.status not in GUARD_STATUSES:
            raise ValueError(f"unsupported guard status: {self.status}")
        if not self.code:
            raise ValueError("guard code cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GuardResult":
        return cls(
            stage=str(raw["stage"]),
            status=str(raw["status"]),
            code=str(raw["code"]),
            message=str(raw.get("message") or ""),
            evidence_refs=[str(item) for item in raw.get("evidence_refs", [])],
            recoverable=bool(raw.get("recoverable", False)),
            timestamp=str(raw.get("timestamp") or utc_now()),
            schema_version=str(raw.get("schema_version") or GUARD_SCHEMA_VERSION),
        )


def guard_pass(stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None) -> GuardResult:
    return GuardResult(stage, "pass", code, message, evidence_refs=evidence_refs or [])


def guard_warn(stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None, recoverable: bool = True) -> GuardResult:
    return GuardResult(stage, "warn", code, message, evidence_refs=evidence_refs or [], recoverable=recoverable)


def guard_fail(stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None, recoverable: bool = False) -> GuardResult:
    return GuardResult(stage, "fail", code, message, evidence_refs=evidence_refs or [], recoverable=recoverable)


def guard_not_available(stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None) -> GuardResult:
    return GuardResult(stage, "not_available", code, message, evidence_refs=evidence_refs or [], recoverable=False)


def guard_dicts(results: list[GuardResult]) -> list[dict[str, Any]]:
    return [result.to_dict() for result in results]


def has_guard_failure(results: list[GuardResult] | list[dict[str, Any]]) -> bool:
    for item in results:
        status = item.status if isinstance(item, GuardResult) else item.get("status")
        if status == "fail":
            return True
    return False


class GuardEngine:
    def __init__(self) -> None:
        self._results: list[GuardResult] = []

    @property
    def results(self) -> list[GuardResult]:
        return list(self._results)

    def add(self, result: GuardResult) -> GuardResult:
        self._results.append(result)
        return result

    def pass_(self, stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None) -> GuardResult:
        return self.add(guard_pass(stage, code, message, evidence_refs=evidence_refs))

    def warn(self, stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None, recoverable: bool = True) -> GuardResult:
        return self.add(guard_warn(stage, code, message, evidence_refs=evidence_refs, recoverable=recoverable))

    def fail(self, stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None, recoverable: bool = False) -> GuardResult:
        return self.add(guard_fail(stage, code, message, evidence_refs=evidence_refs, recoverable=recoverable))

    def not_available(self, stage: str, code: str, message: str, *, evidence_refs: list[str] | None = None) -> GuardResult:
        return self.add(guard_not_available(stage, code, message, evidence_refs=evidence_refs))

    def status(self) -> str:
        if any(result.status == "fail" for result in self._results):
            return "fail"
        if any(result.status == "warn" for result in self._results):
            return "warn"
        if any(result.status == "not_available" for result in self._results):
            return "not_available"
        return "pass"

    def to_dicts(self) -> list[dict[str, Any]]:
        return guard_dicts(self._results)
