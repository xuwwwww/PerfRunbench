from __future__ import annotations

from pathlib import Path
from typing import Any

from autotune.recommendation.schemas import stable_digest
from autotune.system_profiles.catalog import create_plan
from autotune.system_profiles.schemas import (
    HostStateSnapshot,
    LeaseChangeRecord,
    NOT_AVAILABLE,
    SystemProfileError,
    SystemRollbackReport,
    SystemTuningLease,
    SystemTuningPlan,
    SystemTuningResult,
    utc_after_minutes,
    utc_now,
)
from autotune.system_profiles.snapshot import RootReader


def apply_system_profile_plan(
    plan: SystemTuningPlan,
    snapshot: HostStateSnapshot,
    *,
    root: str | Path | None = None,
    lease_minutes: int = 30,
    allow_real_system: bool = False,
) -> SystemTuningResult:
    reader = RootReader(root)
    _require_fake_or_allowed(reader, allow_real_system)
    if snapshot.snapshot_digest != plan.snapshot_digest:
        raise SystemProfileError("snapshot digest does not match plan")
    if lease_minutes <= 0:
        raise SystemProfileError("lease_minutes must be positive")
    records: list[LeaseChangeRecord] = []
    issues: list[str] = []
    host_state_dirty = False
    applied_any = False
    for change in plan.changes:
        if change.interface in {"process", "process_env", "workload_protocol"}:
            records.append(
                LeaseChangeRecord(
                    label=change.label,
                    interface=change.interface,
                    path=change.path,
                    before_value=change.before_value,
                    requested_value=change.requested_value,
                    effective_value=change.requested_value,
                    applied=True,
                )
            )
            applied_any = True
            continue
        target = reader.path(change.path)
        if "*" in change.path:
            records.append(_failed(change, "wildcard changes are planning-only in Phase 21A"))
            issues.append(f"{change.label}: wildcard changes are planning-only in Phase 21A")
            host_state_dirty = host_state_dirty or applied_any
            break
        if not target.exists():
            records.append(_failed(change, "target path is unavailable"))
            issues.append(f"{change.label}: target path is unavailable")
            host_state_dirty = host_state_dirty or applied_any
            break
        before = _read(target)
        if change.before_value != NOT_AVAILABLE and before != change.before_value:
            records.append(_failed(change, f"dirty host: expected {change.before_value!r}, observed {before!r}"))
            issues.append(f"{change.label}: dirty host before apply")
            host_state_dirty = True
            break
        try:
            target.write_text(change.requested_value + "\n", encoding="utf-8", newline="\n")
        except OSError as exc:
            records.append(_failed(change, f"write failed: {exc}"))
            issues.append(f"{change.label}: write failed")
            host_state_dirty = True
            break
        effective = _read(target)
        applied = effective == change.requested_value
        if not applied:
            issues.append(f"{change.label}: effective state mismatch")
            host_state_dirty = True
            break
        records.append(
            LeaseChangeRecord(
                label=change.label,
                interface=change.interface,
                path=change.path,
                before_value=before,
                requested_value=change.requested_value,
                effective_value=effective,
                applied=True,
            )
        )
        applied_any = True
    lease_id = stable_digest({"plan_id": plan.plan_id, "snapshot_digest": snapshot.snapshot_digest, "created_at": utc_now()})[:24]
    lease = SystemTuningLease(
        lease_id=lease_id,
        profile_id=plan.profile.profile_id,
        snapshot_digest=snapshot.snapshot_digest,
        expires_at_utc=utc_after_minutes(lease_minutes),
        changes=records,
        active=applied_any and not host_state_dirty,
    )
    verification = verify_system_profile(plan, lease, root=root, allow_real_system=allow_real_system)
    status = "applied" if lease.active and verification["verification_status"] == "passed" else "failed"
    return SystemTuningResult(
        profile_id=plan.profile.profile_id,
        apply_status=status,
        lease=lease,
        verification_status=verification["verification_status"],
        host_state_dirty=host_state_dirty or verification["host_state_dirty"],
        issues=issues + verification["issues"],
    )


def verify_system_profile(
    plan: SystemTuningPlan,
    lease: SystemTuningLease,
    *,
    root: str | Path | None = None,
    allow_real_system: bool = False,
) -> dict[str, Any]:
    reader = RootReader(root)
    _require_fake_or_allowed(reader, allow_real_system)
    issues = []
    dirty = False
    for record in lease.changes:
        if not record.applied:
            issues.append(f"{record.label}: not applied")
            continue
        if record.interface in {"process", "process_env", "workload_protocol"}:
            continue
        target = reader.path(record.path)
        observed = _read(target) if target.exists() else NOT_AVAILABLE
        if observed != record.requested_value:
            issues.append(f"{record.label}: expected effective {record.requested_value!r}, observed {observed!r}")
            dirty = True
    return {"schema_version": "1.0", "profile_id": plan.profile.profile_id, "verification_status": "failed" if issues else "passed", "host_state_dirty": dirty, "issues": issues}


def rollback_system_profile(
    snapshot: HostStateSnapshot,
    lease: SystemTuningLease,
    *,
    root: str | Path | None = None,
    allow_real_system: bool = False,
) -> SystemRollbackReport:
    reader = RootReader(root)
    _require_fake_or_allowed(reader, allow_real_system)
    if lease.snapshot_digest != snapshot.snapshot_digest:
        raise SystemProfileError("lease snapshot digest does not match snapshot")
    restored: list[LeaseChangeRecord] = []
    issues = []
    dirty = False
    for record in reversed(lease.changes):
        if not record.applied:
            restored.append(record)
            continue
        if record.interface in {"process", "process_env", "workload_protocol"}:
            restored.append(LeaseChangeRecord(**{**record.to_dict(), "effective_value": record.before_value, "applied": True}))
            continue
        target = reader.path(record.path)
        if not target.exists():
            issues.append(f"{record.label}: rollback target missing")
            dirty = True
            restored.append(LeaseChangeRecord(**{**record.to_dict(), "error": "rollback target missing"}))
            continue
        try:
            target.write_text(record.before_value + "\n", encoding="utf-8", newline="\n")
        except OSError as exc:
            issues.append(f"{record.label}: rollback write failed")
            dirty = True
            restored.append(LeaseChangeRecord(**{**record.to_dict(), "error": f"rollback write failed: {exc}"}))
            continue
        observed = _read(target)
        if observed != record.before_value:
            issues.append(f"{record.label}: rollback verification mismatch")
            dirty = True
        restored.append(LeaseChangeRecord(**{**record.to_dict(), "effective_value": observed, "applied": observed == record.before_value}))
    return SystemRollbackReport(
        profile_id=lease.profile_id,
        rollback_status="failed" if issues else "passed",
        restored_changes=restored,
        host_state_dirty=dirty,
        issues=issues,
    )


def plan_and_apply(
    profile_id: str,
    snapshot: HostStateSnapshot,
    *,
    root: str | Path | None,
    lease_minutes: int,
    allow_real_system: bool = False,
) -> SystemTuningResult:
    plan = create_plan(profile_id, snapshot, root)
    return apply_system_profile_plan(plan, snapshot, root=root, lease_minutes=lease_minutes, allow_real_system=allow_real_system)


def _require_fake_or_allowed(reader: RootReader, allow_real_system: bool) -> None:
    if reader.root == Path("/") and not allow_real_system:
        raise SystemProfileError("Phase 21A refuses real system writes; pass a fake --root for apply/rollback tests")


def _failed(change: Any, error: str) -> LeaseChangeRecord:
    return LeaseChangeRecord(
        label=change.label,
        interface=change.interface,
        path=change.path,
        before_value=change.before_value,
        requested_value=change.requested_value,
        effective_value=NOT_AVAILABLE,
        applied=False,
        error=error,
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return NOT_AVAILABLE
