from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.artifact_validation import validate_artifact_manifest
from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest
from autotune.arm_tune.build_profiles.definitions import available_profile_ids, get_profile_definition
from autotune.arm_tune.build_profiles.evidence import kleidiai_status
from autotune.arm_tune.build_profiles.inspection import inspect_binary, inspect_source_repo, toolchain_from_payload
from autotune.arm_tune.build_profiles.schemas import BuildProfileEvidence, BuildProfilePlan, BuildProfileRunResult
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.schemas import BenchmarkArtifactManifest
from autotune.arm_tune.targets.protocol import atomic_write_json
from autotune.utils.config import load_config


BUILD_PLAN_OPERATION = "build_profile.plan"
BUILD_PROFILE_OPERATION = "build_profile.inspect_existing"
BUILD_TOP_LEVEL_KEYS = {
    "schema_version",
    "result_origin",
    "source_repo",
    "build_root",
    "target_architecture",
    "profiles",
    "binaries",
    "toolchain",
    "build",
    "linked_libraries",
}
FORBIDDEN_COMMAND_KEYS = {"command", "shell", "shell_command", "prebuild_command", "postbuild_command"}


def write_build_plan(*, config_path: str | Path, run_id: str, output_dir: str | Path) -> BuildProfilePlan:
    config = _load_build_config(config_path)
    source = inspect_source_repo(_optional_text(config.get("source_repo")))
    profile_ids = _profile_ids(config)
    definitions = [get_profile_definition(profile_id) for profile_id in profile_ids]
    toolchain = toolchain_from_payload(_object_or_empty(config.get("toolchain"), "toolchain"))
    plan = BuildProfilePlan(
        run_id=run_id,
        operation=BUILD_PLAN_OPERATION,
        result_origin=_result_origin(config),
        source=source,
        target_architecture=str(config.get("target_architecture", "unknown")),
        toolchain=toolchain,
        profiles=definitions,
        build_root=str(config.get("build_root", "")),
        execute_build=False,
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "build_plan.json", plan.to_dict())
    _finalize_artifacts(root)
    return plan


def run_build_profile(*, config_path: str | Path, run_id: str, output_dir: str | Path) -> BuildProfileRunResult:
    config = _load_build_config(config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = write_build_plan(config_path=config_path, run_id=run_id, output_dir=root)
    binaries = _object_or_empty(config.get("binaries"), "binaries")
    build_payload = _object_or_empty(config.get("build"), "build")
    linked_libraries = _linked_libraries(config)
    profiles: list[BuildProfileEvidence] = []
    for definition in plan.profiles:
        profile_payload = _profile_payload(config, definition.profile_id)
        binary = inspect_binary(definition.profile_id, _binary_path(binaries, definition.profile_id))
        requested = dict(definition.cmake_definitions)
        requested.update(_string_dict(profile_payload.get("requested_definitions", {}), f"profiles.{definition.profile_id}.requested_definitions"))
        effective_cache = dict(requested)
        effective_cache.update(_string_dict(build_payload.get("effective_cmake_cache", {}), "build.effective_cmake_cache"))
        effective_cache.update(_string_dict(profile_payload.get("effective_cmake_cache", {}), f"profiles.{definition.profile_id}.effective_cmake_cache"))
        compile_evidence = _compile_evidence(profile_payload.get("compile_command_evidence"))
        build_status = "completed" if binary.validation_status == "passed" else "failed"
        completeness = "complete" if binary.validation_status == "passed" else "incomplete"
        profile_linked = [*linked_libraries, *[str(item) for item in profile_payload.get("linked_libraries", []) if isinstance(item, str)]]
        kleidi = kleidiai_status(
            definition=definition,
            build_status=build_status,
            effective_cmake_cache=effective_cache,
            compile_command_evidence=compile_evidence,
            linked_library_evidence=profile_linked,
        )
        profiles.append(
            BuildProfileEvidence(
                run_id=run_id,
                profile_id=definition.profile_id,
                definition=definition,
                source=plan.source,
                toolchain=plan.toolchain,
                requested_definitions=requested,
                effective_cmake_cache=effective_cache,
                compile_command_evidence=compile_evidence,
                target_architecture=plan.target_architecture,
                binary=binary,
                linked_library_evidence=profile_linked,
                build_output={
                    "schema_version": SCHEMA_VERSION,
                    "mode": BUILD_PROFILE_OPERATION,
                    "message": "No arbitrary build command was executed; existing build artifacts were inspected.",
                },
                build_status=build_status,
                completeness=completeness,
                kleidiai_status=kleidi,
                result_origin=plan.result_origin,
            )
        )
    result = BuildProfileRunResult(
        run_id=run_id,
        operation=BUILD_PROFILE_OPERATION,
        result_origin=_combined_origin([profile.result_origin for profile in profiles]),
        profiles=profiles,
        artifact_directory=str(root),
    )
    atomic_write_json(root / "build_profile_result.json", result.to_dict())
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
    atomic_write_json(root / "build_profile_result.json", result.to_dict())
    return result


def inspect_build(*, config_path: str | Path, run_id: str, output_dir: str | Path) -> BuildProfileRunResult:
    return run_build_profile(config_path=config_path, run_id=run_id, output_dir=output_dir)


def _load_build_config(path: str | Path) -> dict[str, Any]:
    payload = load_config(path)
    if not isinstance(payload, dict):
        raise ArmTuneConfigError("build profile config must be an object")
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ArmTuneConfigError(f"unsupported schema_version: {payload.get('schema_version')!r}")
    forbidden = _find_forbidden_command_keys(payload)
    if forbidden:
        raise ArmTuneConfigError(f"build profile config must not contain arbitrary command field(s): {sorted(forbidden)}")
    unknown = set(payload) - BUILD_TOP_LEVEL_KEYS
    if unknown:
        raise ArmTuneConfigError(f"unknown build profile config field(s): {sorted(unknown)}")
    return dict(payload)


def _find_forbidden_command_keys(value: Any, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text == "compile_command_evidence":
                continue
            if key_text in FORBIDDEN_COMMAND_KEYS:
                found.add(f"{prefix}{key_text}" if prefix else key_text)
            found.update(_find_forbidden_command_keys(item, f"{prefix}{key_text}."))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_find_forbidden_command_keys(item, f"{prefix}{index}."))
    return found


def _profile_ids(config: dict[str, Any]) -> list[str]:
    raw = config.get("profiles") or available_profile_ids()
    if not isinstance(raw, list) or not raw:
        raise ArmTuneConfigError("profiles must be a non-empty list")
    ids: list[str] = []
    for item in raw:
        if isinstance(item, str):
            profile_id = item
        elif isinstance(item, dict) and isinstance(item.get("profile_id"), str):
            profile_id = str(item["profile_id"])
        else:
            raise ArmTuneConfigError("profiles must contain profile IDs or profile objects")
        if profile_id not in available_profile_ids():
            raise ArmTuneConfigError(f"unsupported profile_id: {profile_id}")
        if profile_id not in ids:
            ids.append(profile_id)
    return ids


def _profile_payload(config: dict[str, Any], profile_id: str) -> dict[str, Any]:
    raw = config.get("profiles") or []
    if not isinstance(raw, list):
        return {}
    for item in raw:
        if isinstance(item, dict) and item.get("profile_id") == profile_id:
            return item
    return {}


def _binary_path(binaries: dict[str, Any], profile_id: str) -> str | None:
    value = binaries.get(profile_id)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return str(value["path"])
    return None


def _compile_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _linked_libraries(config: dict[str, Any]) -> list[str]:
    raw = config.get("linked_libraries", [])
    if not isinstance(raw, list):
        raise ArmTuneConfigError("linked_libraries must be a list")
    return [str(item) for item in raw if isinstance(item, str)]


def _string_dict(value: Any, label: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ArmTuneConfigError(f"{label} must be an object")
    return {str(key): str(item) for key, item in value.items()}


def _object_or_empty(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ArmTuneConfigError(f"{label} must be an object")
    return dict(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArmTuneConfigError("source_repo must be a string")
    return value


def _result_origin(config: dict[str, Any]) -> str:
    origin = str(config.get("result_origin", "unverified"))
    if origin not in {"measured", "test_fixture", "unverified"}:
        raise ArmTuneConfigError("result_origin must be 'measured', 'test_fixture', or 'unverified'")
    return origin


def _combined_origin(origins: list[str]) -> str:
    unique = set(origins)
    if not unique or "unverified" in unique:
        return "unverified"
    if unique == {"measured"}:
        return "measured"
    if unique == {"test_fixture"}:
        return "test_fixture"
    return "unverified"


def _finalize_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    skipped = {"artifact_manifest.json", "artifact_validation.json", "build_profile_result.json"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in skipped:
            continue
        label = path.relative_to(root).as_posix().replace("/", "__")
        artifacts[label] = artifact_record(path, label=label, root=root)
    write_artifact_manifest(root / "artifact_manifest.json", artifacts)
    return artifacts
