from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autotune.arm_tune.build_profiles.schemas import NativeBinaryReuseDecision, NativeBinaryScope


def native_scope_from_payload(payload: Mapping[str, Any] | None) -> NativeBinaryScope | None:
    if not isinstance(payload, Mapping):
        return None
    required = {"architecture", "portability_state"}
    if not required.issubset(payload):
        return None
    return NativeBinaryScope(
        architecture=str(payload.get("architecture", "unknown")),
        portability_state=str(payload.get("portability_state", "exact_target_only")),
        source_commit=str(payload.get("source_commit", "unknown")),
        compiler_identity=str(payload.get("compiler_identity", "unknown")),
        effective_cpu_flags=[str(item) for item in payload.get("effective_cpu_flags", []) if isinstance(item, str)],
        required_cpu_features=sorted({str(item).lower() for item in payload.get("required_cpu_features", []) if isinstance(item, str)}),
        cpu_model=str(payload.get("cpu_model", "unknown")),
        cpu_family=str(payload.get("cpu_family", "unknown")),
        binary_sha256=str(payload.get("binary_sha256", "unknown")),
        build_target_fingerprint=str(payload.get("build_target_fingerprint", "unknown")),
        verified_target_fingerprints=[
            str(item) for item in payload.get("verified_target_fingerprints", []) if isinstance(item, str)
        ],
        startup_smoke_required=bool(payload.get("startup_smoke_required", False)),
    )


def evaluate_native_binary_reuse(
    scope: NativeBinaryScope,
    *,
    current_architecture: str,
    current_features: set[str],
    current_cpu_model: str,
    current_cpu_family: str = "unknown",
    current_target_fingerprint: str = "unknown",
    observed_binary_sha256: str | None = None,
    startup_smoke_status: str = "not_required",
) -> NativeBinaryReuseDecision:
    reasons: list[str] = []
    state = scope.portability_state
    if state not in {"portable", "architecture_bound", "feature_bound", "cpu_model_bound", "exact_target_only"}:
        return NativeBinaryReuseDecision(
            compatibility_status="invalid_evidence",
            launch_allowed=False,
            reasons=[f"unsupported portability_state={state!r}"],
        )
    if not scope.required_cpu_features and state in {"feature_bound", "cpu_model_bound", "exact_target_only"}:
        return NativeBinaryReuseDecision(
            compatibility_status="invalid_evidence",
            launch_allowed=False,
            reasons=["required_cpu_features must be present for CPU-specific binaries"],
        )
    if _norm(scope.architecture) != _norm(current_architecture):
        reasons.append(f"architecture mismatch: required={scope.architecture} observed={current_architecture}")
    missing = sorted(set(scope.required_cpu_features) - {item.lower() for item in current_features})
    if missing:
        reasons.append(f"required CPU feature mismatch: {', '.join(missing)}")
    if state in {"cpu_model_bound", "exact_target_only"} and _known(scope.cpu_model) and _norm(scope.cpu_model) != _norm(current_cpu_model):
        reasons.append(f"CPU model mismatch: required={scope.cpu_model} observed={current_cpu_model}")
    if state == "exact_target_only":
        verified = set(scope.verified_target_fingerprints)
        if scope.build_target_fingerprint != "unknown":
            verified.add(scope.build_target_fingerprint)
        if current_target_fingerprint not in verified:
            reasons.append("target fingerprint is not in the verified native-binary set")
    if observed_binary_sha256 and scope.binary_sha256 != "unknown" and observed_binary_sha256.lower() != scope.binary_sha256.lower():
        reasons.append("binary SHA-256 mismatch")
    smoke_required = scope.startup_smoke_required or state in {"cpu_model_bound", "exact_target_only"}
    if smoke_required and startup_smoke_status not in {"passed", "not_run_allowed_by_policy"}:
        reasons.append(f"startup smoke did not pass: {startup_smoke_status}")
    if reasons:
        return NativeBinaryReuseDecision(
            compatibility_status="hard_drift",
            launch_allowed=False,
            reasons=reasons,
            required_startup_smoke=smoke_required,
        )
    return NativeBinaryReuseDecision(
        compatibility_status="compatible",
        launch_allowed=True,
        reasons=[],
        required_startup_smoke=smoke_required,
    )


def _known(value: str) -> bool:
    return bool(value and value.lower() not in {"unknown", "not available", "n/a"})


def _norm(value: str) -> str:
    return value.strip().lower().replace("_", "-")
