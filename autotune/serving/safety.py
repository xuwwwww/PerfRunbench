from __future__ import annotations

from typing import Any

from autotune.serving import SERVING_SCHEMA_VERSION


HARD_DRIFT_FIELDS = {
    "server_binary_digest",
    "model_digest",
    "serving_workload_id",
    "traffic_mode",
    "concurrency",
    "network_topology",
    "server_capability_digest",
    "architecture",
    "kernel_identity",
}


def validate_serving_reuse(bundle: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    guards = []
    for field in sorted(HARD_DRIFT_FIELDS):
        expected = _read(bundle, field)
        observed = _read(current, field)
        if expected is None or observed is None:
            guards.append(_guard(field, "failed", "missing"))
        elif expected != observed:
            guards.append(_guard(field, "failed", "hard_drift"))
        else:
            guards.append(_guard(field, "passed", "matched"))
    status = "failed" if any(item["status"] == "failed" for item in guards) else "passed"
    return {"schema_version": SERVING_SCHEMA_VERSION, "status": status, "guard_results": guards}


def _read(payload: dict[str, Any], field: str) -> Any:
    if field in payload:
        return payload[field]
    if field == "network_topology":
        network = payload.get("network_profile")
        if isinstance(network, dict):
            return network.get("topology")
    if field in {"architecture", "kernel_identity"}:
        runtime = payload.get("runtime_identity")
        if isinstance(runtime, dict):
            return runtime.get(field)
    return None


def _guard(field: str, status: str, reason: str) -> dict[str, Any]:
    return {"schema_version": SERVING_SCHEMA_VERSION, "field": field, "status": status, "reason": reason}
