from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.schemas import ArmBenchmarkResult, ArtifactValidationIssue, ArtifactValidationResult
from autotune.arm_tune.targets.protocol import read_json


def validate_artifact_manifest(run_dir: str | Path, manifest_path: str | Path | None = None) -> ArtifactValidationResult:
    root = Path(run_dir).resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else root / "artifact_manifest.json"
    issues: list[ArtifactValidationIssue] = []
    checked = 0
    if not _is_within(root, manifest):
        issues.append(ArtifactValidationIssue(label="artifact_manifest", path=str(manifest), issue="manifest path is outside run directory"))
        return ArtifactValidationResult(validation_status="failed", checked_artifacts=0, issues=issues)
    try:
        payload = read_json(manifest)
    except Exception as exc:
        issues.append(ArtifactValidationIssue(label="artifact_manifest", path=str(manifest), issue=f"manifest is not readable JSON: {exc}"))
        return ArtifactValidationResult(validation_status="failed", checked_artifacts=0, issues=issues)
    if payload.get("schema_version") != "1.0":
        issues.append(ArtifactValidationIssue(label="artifact_manifest", path=str(manifest), issue="unsupported schema_version"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        issues.append(ArtifactValidationIssue(label="artifact_manifest", path=str(manifest), issue="artifacts must be an object"))
        return ArtifactValidationResult(validation_status="failed", checked_artifacts=0, issues=issues)
    seen_relative_paths: set[str] = set()
    for label, record in artifacts.items():
        checked += 1
        if not isinstance(record, dict):
            issues.append(ArtifactValidationIssue(label=str(label), path="", issue="artifact record must be an object"))
            continue
        target = _artifact_target(root, record)
        record_path = str(record.get("relative_path") or record.get("path") or "")
        if target is None:
            issues.append(ArtifactValidationIssue(label=str(label), path=record_path, issue="artifact path is outside run directory or invalid"))
            continue
        relative = target.relative_to(root).as_posix()
        if relative in seen_relative_paths:
            issues.append(ArtifactValidationIssue(label=str(label), path=relative, issue="duplicate artifact relative_path"))
        seen_relative_paths.add(relative)
        if not target.exists():
            issues.append(ArtifactValidationIssue(label=str(label), path=relative, issue="artifact is missing"))
            continue
        if target.is_symlink() and not _is_within(root, target.resolve()):
            issues.append(ArtifactValidationIssue(label=str(label), path=relative, issue="artifact symlink escapes run directory"))
            continue
        expected = record.get("sha256")
        if expected:
            digest = sha256_file(target)
            if digest.lower() != str(expected).lower():
                issues.append(ArtifactValidationIssue(label=str(label), path=relative, issue="sha256 mismatch"))
    status = "passed" if not issues else "failed"
    return ArtifactValidationResult(validation_status=status, checked_artifacts=checked, issues=issues)


def apply_artifact_validation(result: ArmBenchmarkResult, validation: ArtifactValidationResult, artifact_directory: str | Path) -> ArmBenchmarkResult:
    benchmark_status = result.benchmark_status
    message = result.message
    if validation.validation_status != "passed" and benchmark_status == "passed":
        benchmark_status = "failed"
        message = f"{message}; artifact validation failed" if message else "artifact validation failed"
    return replace(
        result,
        benchmark_status=benchmark_status,
        artifact_validation=validation,
        artifact_directory=str(artifact_directory),
    )


def _artifact_target(root: Path, record: dict[str, Any]) -> Path | None:
    relative = record.get("relative_path")
    if isinstance(relative, str) and relative:
        candidate = root / relative
    else:
        path_text = record.get("path")
        if not isinstance(path_text, str) or not path_text:
            return None
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        return None
    if not _is_within(root, resolved):
        return None
    return resolved


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False
