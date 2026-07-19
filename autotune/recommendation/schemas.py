from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


RECOMMENDATION_SCHEMA_VERSION = "1.0"
RECOMMENDATION_STATUS_ELIGIBLE = {"recommended", "eligible_preliminary"}
RECOMMENDATION_STATUS_REJECTED = {
    "fixture",
    "incomplete",
    "inconclusive",
    "invalid",
    "stale_environment",
    "rejected_correctness",
}
APPLY_SCOPES = {"process", "system"}
OBJECTIVE_DIRECTIONS = {"maximize", "minimize"}
SENSITIVE_ENV_MARKERS = ("TOKEN", "PASSWORD", "SECRET", "PRIVATE_KEY", "CREDENTIAL", "ACCESS_KEY")


class RecommendationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchSpec:
    argv: list[str]
    adapter: str = "generic-command"
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_artifacts: list[str] = field(default_factory=list)
    correctness_contract: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RECOMMENDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_launch_spec(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_launch_spec(raw: dict[str, Any]) -> None:
    if raw.get("schema_version", RECOMMENDATION_SCHEMA_VERSION) != RECOMMENDATION_SCHEMA_VERSION:
        raise RecommendationError("unsupported launch spec schema_version")
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        raise RecommendationError("launch spec requires a non-empty argv list of strings")
    if raw.get("shell") is True:
        raise RecommendationError("launch spec must not request shell execution")
    adapter = raw.get("adapter")
    if adapter is not None and not isinstance(adapter, str):
        raise RecommendationError("launch spec adapter must be a string")
    cwd = raw.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise RecommendationError("launch spec cwd must be a string when provided")
    env = raw.get("env", {})
    if not isinstance(env, dict):
        raise RecommendationError("launch spec env must be an object")
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise RecommendationError("launch spec env keys and values must be strings")
        upper_key = key.upper()
        if any(marker in upper_key for marker in SENSITIVE_ENV_MARKERS):
            raise RecommendationError(f"launch spec env must not include credential-like key: {key}")
    timeout = raw.get("timeout_seconds", 300.0)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise RecommendationError("launch spec timeout_seconds must be > 0")
    contract = raw.get("correctness_contract", {})
    if contract is not None and not isinstance(contract, dict):
        raise RecommendationError("launch spec correctness_contract must be an object")


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_versioned_top_level(raw: dict[str, Any], *, expected: str = RECOMMENDATION_SCHEMA_VERSION) -> None:
    if not isinstance(raw, dict):
        raise RecommendationError("top-level JSON structure must be an object")
    if raw.get("schema_version") != expected:
        raise RecommendationError(f"top-level schema_version must be {expected}")
