from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from autotune.golden_path import SCHEMA_VERSION
from autotune.recommendation.bundle import load_recommendation_bundle, validate_recommendation_bundle
from autotune.recommendation.fingerprints import compare_environment_fingerprints
from autotune.recommendation.guards import utc_now
from autotune.recommendation.schemas import atomic_write_json, file_sha256, stable_digest


CACHE_STATES = {"exact_match", "soft_drift", "hard_drift", "expired", "revoked", "missing", "corrupt", "ambiguous"}


def insert_cache_entry(
    *,
    root: str | Path,
    bundle_path: str | Path,
    bundle: dict[str, Any],
    workload_identity: dict[str, Any],
    environment_fingerprint: dict[str, Any],
    tuning_result: dict[str, Any],
    ttl_days: int = 14,
) -> dict[str, Any]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    _revoke_matching(root_path, workload_identity.get("digest"))
    created = utc_now()
    cache_id = stable_digest({"workload": workload_identity.get("digest"), "bundle": bundle.get("bundle_id"), "created_at": created})[:24]
    expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    entry = {
        "schema_version": SCHEMA_VERSION,
        "cache_id": cache_id,
        "created_at": created,
        "expires_at": expires_at,
        "revoked": False,
        "bundle_id": bundle.get("bundle_id"),
        "bundle_path": str(Path(bundle_path)),
        "bundle_sha256": file_sha256(bundle_path),
        "workload_identity": workload_identity,
        "environment_fingerprint": environment_fingerprint,
        "objective": bundle.get("objective"),
        "selected_configuration_id": bundle.get("selected_configuration", {}).get("configuration_id"),
        "source_run_id": bundle.get("source_run_id"),
        "summary": {
            "status": tuning_result.get("status"),
            "metric": tuning_result.get("objective", {}).get("metric"),
            "baseline_median": tuning_result.get("statistical_evidence", {}).get("baseline_median"),
            "selected_median": tuning_result.get("statistical_evidence", {}).get("selected_median"),
            "improvement_percent": tuning_result.get("statistical_evidence", {}).get("improvement_percent"),
        },
    }
    atomic_write_json(_entry_path(root_path, cache_id), entry)
    return entry


def lookup_cache(
    *,
    root: str | Path,
    workload_identity: dict[str, Any],
    environment_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    root_path = Path(root)
    entries = list_cache_entries(root_path, include_revoked=True)
    if not entries:
        return {"schema_version": SCHEMA_VERSION, "state": "missing", "reason": "cache is empty"}
    matching = [entry for entry in entries if entry.get("workload_identity", {}).get("digest") == workload_identity.get("digest")]
    if not matching:
        return {"schema_version": SCHEMA_VERSION, "state": "missing", "reason": "no matching workload identity"}
    usable = []
    expired = []
    revoked = []
    corrupt = []
    for entry in matching:
        if entry.get("cache_state") == "corrupt":
            corrupt.append(entry)
            continue
        if entry.get("revoked") is True:
            revoked.append(entry)
            continue
        if _expired(entry):
            expired.append(entry)
            continue
        bundle_result = _validate_bundle_ref(entry)
        if bundle_result["state"] == "corrupt":
            corrupt.append({**entry, **bundle_result})
            continue
        drift = compare_environment_fingerprints(entry.get("environment_fingerprint", {}), environment_fingerprint)
        state = {"match": "exact_match", "soft_mismatch": "soft_drift", "hard_mismatch": "hard_drift"}.get(drift["status"], "hard_drift")
        usable.append({**entry, "state": state, "environment_drift": drift, "bundle": bundle_result["bundle"]})
    exact = [entry for entry in usable if entry["state"] == "exact_match"]
    if len(exact) > 1:
        return {"schema_version": SCHEMA_VERSION, "state": "ambiguous", "entries": _public_entries(exact), "reason": "multiple exact cache entries"}
    if exact:
        return {"schema_version": SCHEMA_VERSION, "state": "exact_match", "entry": _without_bundle(exact[0]), "bundle": exact[0]["bundle"]}
    soft = [entry for entry in usable if entry["state"] == "soft_drift"]
    if len(soft) > 1:
        return {"schema_version": SCHEMA_VERSION, "state": "ambiguous", "entries": _public_entries(soft), "reason": "multiple soft-drift entries"}
    if soft:
        return {"schema_version": SCHEMA_VERSION, "state": "soft_drift", "entry": _without_bundle(soft[0]), "bundle": soft[0]["bundle"]}
    hard = [entry for entry in usable if entry["state"] == "hard_drift"]
    if hard:
        return {"schema_version": SCHEMA_VERSION, "state": "hard_drift", "entry": _without_bundle(hard[0]), "bundle": hard[0]["bundle"]}
    if expired:
        return {"schema_version": SCHEMA_VERSION, "state": "expired", "entries": _public_entries(expired)}
    if revoked:
        return {"schema_version": SCHEMA_VERSION, "state": "revoked", "entries": _public_entries(revoked)}
    if corrupt:
        return {"schema_version": SCHEMA_VERSION, "state": "corrupt", "entries": _public_entries(corrupt)}
    return {"schema_version": SCHEMA_VERSION, "state": "missing", "reason": "no usable cache entry"}


def list_cache_entries(root: str | Path, *, include_revoked: bool = False) -> list[dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    entries = []
    for path in sorted(root_path.glob("*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries.append({"schema_version": SCHEMA_VERSION, "cache_id": path.stem, "cache_state": "corrupt", "path": str(path)})
            continue
        if not include_revoked and entry.get("revoked") is True:
            continue
        entries.append(entry)
    return entries


def show_cache_entry(root: str | Path, cache_id: str) -> dict[str, Any]:
    path = _entry_path(Path(root), cache_id)
    return json.loads(path.read_text(encoding="utf-8"))


def invalidate_cache_entry(root: str | Path, cache_id: str) -> dict[str, Any]:
    entry = show_cache_entry(root, cache_id)
    entry["revoked"] = True
    entry["revoked_at"] = utc_now()
    atomic_write_json(_entry_path(Path(root), cache_id), entry)
    return entry


def gc_cache(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    removed: list[str] = []
    if not root_path.exists():
        return {"schema_version": SCHEMA_VERSION, "removed": removed}
    for entry in list_cache_entries(root_path, include_revoked=True):
        if entry.get("cache_state") == "corrupt" or entry.get("revoked") is True or _expired(entry):
            path = _entry_path(root_path, str(entry.get("cache_id")))
            try:
                path.unlink()
            except OSError:
                continue
            removed.append(path.name)
    return {"schema_version": SCHEMA_VERSION, "removed": removed}


def _entry_path(root: Path, cache_id: str) -> Path:
    return root / f"{cache_id}.json"


def _revoke_matching(root: Path, workload_digest: object) -> None:
    if not workload_digest:
        return
    for entry in list_cache_entries(root, include_revoked=False):
        if entry.get("workload_identity", {}).get("digest") == workload_digest:
            entry["revoked"] = True
            entry["revoked_at"] = utc_now()
            atomic_write_json(_entry_path(root, str(entry.get("cache_id"))), entry)


def _expired(entry: dict[str, Any]) -> bool:
    raw = entry.get("expires_at")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    return parsed < datetime.now(timezone.utc)


def _validate_bundle_ref(entry: dict[str, Any]) -> dict[str, Any]:
    bundle_path = entry.get("bundle_path")
    expected = entry.get("bundle_sha256")
    if not isinstance(bundle_path, str) or not bundle_path:
        return {"state": "corrupt", "reason": "bundle_path missing"}
    path = Path(bundle_path)
    if not path.exists():
        return {"state": "corrupt", "reason": "bundle missing"}
    if isinstance(expected, str) and expected and file_sha256(path) != expected:
        return {"state": "corrupt", "reason": "bundle digest mismatch"}
    try:
        bundle = load_recommendation_bundle(path)
        validation = validate_recommendation_bundle(bundle)
    except Exception as exc:
        return {"state": "corrupt", "reason": str(exc)}
    if validation.get("status") != "pass":
        return {"state": "corrupt", "reason": "bundle validation failed"}
    return {"state": "ok", "bundle": bundle}


def _public_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_without_bundle(entry) for entry in entries]


def _without_bundle(entry: dict[str, Any]) -> dict[str, Any]:
    public = dict(entry)
    public.pop("bundle", None)
    return public
