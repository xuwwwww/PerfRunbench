from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.arm_tune.build_profiles.schemas import BinaryEvidence, BuildSourceEvidence, ToolchainEvidence
from autotune.arm_tune.schemas import BenchmarkFileManifest


def inspect_source_repo(path_text: str | None) -> BuildSourceEvidence:
    if not path_text:
        return BuildSourceEvidence(source_repo="", source_status="not_configured")
    path = Path(path_text)
    if not path.exists() or not path.is_dir():
        return BuildSourceEvidence(source_repo=str(path), source_status="missing")
    commit = _git_output(path, ["rev-parse", "HEAD"]) or "unknown"
    status = _git_output(path, ["status", "--porcelain"])
    remote = _git_output(path, ["config", "--get", "remote.origin.url"]) or "unknown"
    dirty_state = "dirty" if status else "clean"
    source_status = "git" if commit != "unknown" else "directory"
    return BuildSourceEvidence(
        source_repo=str(path.resolve()),
        source_status=source_status,
        git_commit=commit,
        git_dirty_state=dirty_state,
        git_remote=remote,
    )


def inspect_binary(profile_id: str, path_text: str | None) -> BinaryEvidence:
    if not path_text:
        return BinaryEvidence(
            profile_id=profile_id,
            binary_path="",
            manifest=None,
            validation_status="failed",
            validation_message="binary path is not configured",
        )
    path = Path(path_text)
    if not path.exists():
        return BinaryEvidence(
            profile_id=profile_id,
            binary_path=str(path),
            manifest=None,
            validation_status="failed",
            validation_message="binary file does not exist",
        )
    if not path.is_file():
        return BinaryEvidence(
            profile_id=profile_id,
            binary_path=str(path),
            manifest=None,
            validation_status="failed",
            validation_message="binary path is not a regular file",
        )
    try:
        digest = sha256_file(path)
        manifest = BenchmarkFileManifest(
            path=str(path.resolve()),
            basename=path.name,
            size_bytes=path.stat().st_size,
            sha256=digest,
            validation_status="passed",
        )
    except OSError as exc:
        return BinaryEvidence(
            profile_id=profile_id,
            binary_path=str(path),
            manifest=None,
            validation_status="failed",
            validation_message=f"binary is not readable: {exc}",
        )
    return BinaryEvidence(
        profile_id=profile_id,
        binary_path=str(path.resolve()),
        manifest=manifest,
        validation_status="passed",
    )


def toolchain_from_payload(payload: dict[str, Any] | None) -> ToolchainEvidence:
    if payload is None:
        payload = {}
    return ToolchainEvidence(
        compiler_id=str(payload.get("compiler_id", "unknown")),
        compiler_version=str(payload.get("compiler_version", "unknown")),
        cmake_version=str(payload.get("cmake_version", "unknown")),
        generator=str(payload.get("generator", "unknown")),
        build_type=str(payload.get("build_type", "Release")),
    )


def _git_output(repo: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()
