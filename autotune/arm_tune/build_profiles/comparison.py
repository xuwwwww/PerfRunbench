from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.artifact_validation import validate_artifact_manifest
from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest
from autotune.arm_tune.build_profiles.schemas import (
    ProfileComparisonEntry,
    ProfileComparisonRecommendation,
    ProfileComparisonResult,
)
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.schemas import BenchmarkArtifactManifest
from autotune.arm_tune.search.statistics import NoiseBandDecision, summarize_metric
from autotune.arm_tune.targets.protocol import atomic_write_json, read_json
from autotune.utils.config import load_config


PROFILE_COMPARISON_OPERATION = "profile_comparison.quality_aware"
PRIMARY_METRIC = "ttft_ms"
SECONDARY_METRIC = "end_to_end_latency_ms"
PROFILE_COMPARISON_KEYS = {
    "schema_version",
    "result_origin",
    "baseline_profile_id",
    "noise_band_percent",
    "profiles",
}


def run_profile_comparison(*, config_path: str | Path, run_id: str, output_dir: str | Path) -> ProfileComparisonResult:
    config = _load_comparison_config(config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "profile_comparison_job.json", _job_payload(config, run_id))
    entries = [_entry_from_payload(item, config) for item in config["profiles"]]
    baseline_id = str(config.get("baseline_profile_id") or entries[0].profile_id)
    recommendation = _recommendation(entries, baseline_id, float(config.get("noise_band_percent", 2.0)))
    result = ProfileComparisonResult(
        run_id=run_id,
        operation=PROFILE_COMPARISON_OPERATION,
        result_origin=_combined_origin([entry.result_origin for entry in entries]),
        baseline_profile_id=baseline_id,
        primary_metric=PRIMARY_METRIC,
        secondary_metric=SECONDARY_METRIC,
        recommendation=recommendation,
        profiles=entries,
        rejected_profiles=_rejected_profiles(entries),
        artifact_directory=str(root),
    )
    atomic_write_json(root / "profile_comparison_result.json", result.to_dict())
    artifacts = _finalize_artifacts(root)
    validation = validate_artifact_manifest(root)
    result = replace(
        result,
        artifacts=BenchmarkArtifactManifest(artifacts=artifacts),
        artifact_validation=validation,
    )
    atomic_write_json(root / "artifact_validation.json", validation.to_dict())
    artifacts = _finalize_artifacts(root)
    result = replace(result, artifacts=BenchmarkArtifactManifest(artifacts=artifacts))
    atomic_write_json(root / "profile_comparison_result.json", result.to_dict())
    return result


def _load_comparison_config(path: str | Path) -> dict[str, Any]:
    payload = load_config(path)
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("profile comparison config must be an object")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ArmTuneConfigError(f"unsupported schema_version: {payload.get('schema_version')!r}")
    unknown = set(payload) - PROFILE_COMPARISON_KEYS
    if unknown:
        raise ArmTuneConfigError(f"unknown profile comparison config field(s): {sorted(unknown)}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ArmTuneConfigError("profile comparison config must contain profiles")
    for item in profiles:
        if not isinstance(item, dict):
            raise ArmTuneConfigError("profile comparison profiles must be objects")
    return dict(payload)


def _entry_from_payload(payload: dict[str, Any], config: dict[str, Any]) -> ProfileComparisonEntry:
    profile_id = _required_text(payload, "profile_id")
    result_origin = str(payload.get("result_origin", config.get("result_origin", "unverified")))
    build = _load_optional_json(payload.get("build_result_path"))
    quality = _load_optional_json(payload.get("quality_calibration_result_path"))
    profile_build = _profile_build_payload(build, profile_id)
    build_status = str(payload.get("build_status") or profile_build.get("build_status") or "unknown")
    kleidiai_status = str(payload.get("kleidiai_status") or profile_build.get("kleidiai_status") or "unavailable")
    quality_status = str(payload.get("quality_gate_status") or quality.get("quality_gate_status") or "not_evaluated")
    measurements = _measurements(payload)
    failed = sum(1 for item in measurements if item.get("failed"))
    ttft_values = [float(item[PRIMARY_METRIC]) for item in measurements if PRIMARY_METRIC in item and not item.get("failed")]
    e2e_values = [float(item[SECONDARY_METRIC]) for item in measurements if SECONDARY_METRIC in item and not item.get("failed")]
    reasons: list[str] = []
    if build_status != "completed":
        reasons.append(f"build_status={build_status}")
    if profile_id == "arm-optimized" and kleidiai_status == "requested_but_unverified":
        reasons.append("KleidiAI was requested but not evidenced")
    if quality_status != "passed":
        reasons.append(f"quality_gate_status={quality_status}")
    if not ttft_values:
        reasons.append("median TTFT evidence is missing")
    eligible = not reasons
    if not eligible:
        if build_status != "completed" or "KleidiAI" in " ".join(reasons):
            eligibility = "rejected_build_evidence"
        elif quality_status != "passed":
            eligibility = "rejected_quality"
        else:
            eligibility = "insufficient_evidence"
    else:
        eligibility = "eligible"
    return ProfileComparisonEntry(
        profile_id=profile_id,
        result_origin=result_origin,
        build_status=build_status,
        kleidiai_status=kleidiai_status,
        quality_gate_status=quality_status,
        eligibility_status=eligibility,
        eligible=eligible,
        ttft_ms=summarize_metric(PRIMARY_METRIC, ttft_values, failed_count=failed),
        end_to_end_latency_ms=summarize_metric(SECONDARY_METRIC, e2e_values, failed_count=failed),
        failure_rate=round(failed / len(measurements), 6) if measurements else 1.0,
        failure_reasons=sorted(set(reasons)),
    )


def _recommendation(
    entries: list[ProfileComparisonEntry],
    baseline_id: str,
    noise_band_percent: float,
) -> ProfileComparisonRecommendation:
    baseline = next((item for item in entries if item.profile_id == baseline_id), None)
    eligible = [item for item in entries if item.eligible and item.ttft_ms.median is not None]
    if baseline is None:
        return ProfileComparisonRecommendation(
            status="insufficient_evidence",
            profile_id=None,
            message=f"Baseline profile {baseline_id!r} is missing.",
        )
    if not eligible:
        return ProfileComparisonRecommendation(
            status="no_eligible_profile",
            profile_id=None,
            message="No profile is eligible because build, execution, or quality evidence is incomplete.",
        )
    best = min(eligible, key=lambda item: float(item.ttft_ms.median))
    decision = _classify_latency_noise_band(baseline=baseline.ttft_ms, candidate=best.ttft_ms, noise_band_percent=noise_band_percent)
    if best.profile_id == baseline.profile_id:
        return ProfileComparisonRecommendation(
            status="inconclusive",
            profile_id=best.profile_id,
            message=f"Best observed quality-passing profile among {len(entries)} tested profiles is the baseline.",
            noise_band_decision=decision,
        )
    if decision.classification == "improved":
        return ProfileComparisonRecommendation(
            status="recommended",
            profile_id=best.profile_id,
            message=f"Best observed quality-passing profile among {len(entries)} tested profiles.",
            noise_band_decision=decision,
        )
    return ProfileComparisonRecommendation(
        status="inconclusive",
        profile_id=best.profile_id,
        message="Best observed quality-passing profile did not exceed the configured TTFT noise band.",
        noise_band_decision=decision,
    )


def _classify_latency_noise_band(
    *,
    baseline,
    candidate,
    noise_band_percent: float,
) -> NoiseBandDecision:
    if baseline.median is None or candidate.median is None or baseline.median <= 0:
        return NoiseBandDecision(
            classification="not_comparable",
            baseline_median=baseline.median,
            candidate_median=candidate.median,
            delta_percent=None,
            noise_band_percent=noise_band_percent,
            message="missing comparable median TTFT metrics",
        )
    improvement = ((baseline.median - candidate.median) / baseline.median) * 100.0
    if improvement > noise_band_percent:
        classification = "improved"
        message = "candidate median TTFT is lower than baseline beyond the configured noise band"
    elif improvement < -noise_band_percent:
        classification = "regressed"
        message = "candidate median TTFT is higher than baseline beyond the configured noise band"
    else:
        classification = "inconclusive"
        message = "candidate median TTFT is within the configured noise band"
    return NoiseBandDecision(
        classification=classification,
        baseline_median=baseline.median,
        candidate_median=candidate.median,
        delta_percent=round(improvement, 6),
        noise_band_percent=noise_band_percent,
        message=message,
    )


def _measurements(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("measurements", [])
    if not isinstance(raw, list):
        raise ArmTuneConfigError("profile measurements must be a list")
    return [dict(item) for item in raw if isinstance(item, dict)]


def _profile_build_payload(build: dict[str, Any], profile_id: str) -> dict[str, Any]:
    profiles = build.get("profiles", [])
    if not isinstance(profiles, list):
        return {}
    for item in profiles:
        if isinstance(item, dict) and item.get("profile_id") == profile_id:
            return item
    return {}


def _load_optional_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, str):
        raise ArmTuneConfigError("artifact path fields must be strings")
    path = Path(value)
    if not path.exists():
        raise ArmTuneConfigError(f"referenced artifact does not exist: {path}")
    return read_json(path)


def _rejected_profiles(entries: list[ProfileComparisonEntry]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "profile_id": entry.profile_id,
            "eligibility_status": entry.eligibility_status,
            "failure_reasons": list(entry.failure_reasons),
        }
        for entry in entries
        if not entry.eligible
    ]


def _job_payload(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": PROFILE_COMPARISON_OPERATION,
        "run_id": run_id,
        "baseline_profile_id": config.get("baseline_profile_id"),
        "noise_band_percent": config.get("noise_band_percent", 2.0),
        "profile_count": len(config.get("profiles", [])),
    }


def _combined_origin(origins: list[str]) -> str:
    unique = set(origins)
    if not unique or "unverified" in unique:
        return "unverified"
    if unique == {"measured"}:
        return "measured"
    if unique == {"test_fixture"}:
        return "test_fixture"
    return "unverified"


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArmTuneConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _finalize_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    skipped = {"artifact_manifest.json", "artifact_validation.json", "profile_comparison_result.json"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in skipped:
            continue
        label = path.relative_to(root).as_posix().replace("/", "__")
        artifacts[label] = artifact_record(path, label=label, root=root)
    write_artifact_manifest(root / "artifact_manifest.json", artifacts)
    return artifacts
