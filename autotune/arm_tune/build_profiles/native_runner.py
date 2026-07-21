from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.benchmark.artifact_validation import validate_artifact_manifest
from autotune.arm_tune.benchmark.artifacts import artifact_record, write_artifact_manifest
from autotune.arm_tune.build_profiles.builder import _combined_origin, _eligibility_state, _finalize_artifacts, _load_build_config, _profile_ids, _profile_payload, _result_origin, _string_dict
from autotune.arm_tune.build_profiles.definitions import get_profile_definition
from autotune.arm_tune.build_profiles.evidence import kleidiai_status
from autotune.arm_tune.build_profiles.inspection import inspect_binary, inspect_source_repo, toolchain_from_payload
from autotune.arm_tune.build_profiles.portability import native_scope_from_payload
from autotune.arm_tune.build_profiles.schemas import BuildProfileEvidence, BuildProfileRunResult
from autotune.arm_tune.build_profiles.validation import validate_compiler_argument_list
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.schemas import BenchmarkArtifactManifest
from autotune.arm_tune.targets.protocol import atomic_write_json


NATIVE_BUILD_OPERATION = "build_profile.native_cmake"


def run_native_build_profiles(*, config_path: str | Path, run_id: str, output_dir: str | Path) -> BuildProfileRunResult:
    config = _load_build_config(config_path)
    build = config.get("build") if isinstance(config.get("build"), dict) else {}
    if not bool(build.get("execute_build", False)):
        from autotune.arm_tune.build_profiles.builder import run_build_profile

        return run_build_profile(config_path=config_path, run_id=run_id, output_dir=output_dir)
    source_repo = _required_path(config.get("source_repo"), "source_repo")
    build_root = _required_path(config.get("build_root"), "build_root")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source = inspect_source_repo(str(source_repo))
    toolchain = toolchain_from_payload(config.get("toolchain") if isinstance(config.get("toolchain"), dict) else {})
    profile_ids = _profile_ids(config)
    profiles: list[BuildProfileEvidence] = []
    model_path = _optional_model_path(config)
    cmake_executable = str(build.get("cmake_executable", "cmake"))
    build_target = str(build.get("build_target", "llama-bench"))
    parallel = max(1, int(build.get("parallel", 1)))
    configure_timeout = int(build.get("configure_timeout_seconds", 600))
    build_timeout = int(build.get("build_timeout_seconds", 1800))
    smoke_timeout = int(build.get("smoke_timeout_seconds", 120))
    for profile_id in profile_ids:
        definition = get_profile_definition(profile_id)
        profile_payload = _profile_payload(config, profile_id)
        profile_dir = root / "profiles" / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        cmake_build_dir = build_root / profile_id
        requested = dict(definition.cmake_definitions)
        requested.update(_string_dict(profile_payload.get("requested_definitions", {}), f"profiles.{profile_id}.requested_definitions"))
        configure = _run(
            [
                cmake_executable,
                "-S",
                str(source_repo),
                "-B",
                str(cmake_build_dir),
                *[f"-D{key}={value}" for key, value in sorted(requested.items())],
            ],
            cwd=source_repo,
            timeout_seconds=configure_timeout,
        )
        _write_process_artifacts(profile_dir, "configure", configure)
        build_result = {"returncode": None, "stdout": "", "stderr": ""}
        if configure["returncode"] == 0:
            build_result = _run(
                [cmake_executable, "--build", str(cmake_build_dir), "--target", build_target, "--parallel", str(parallel)],
                cwd=source_repo,
                timeout_seconds=build_timeout,
            )
            _write_process_artifacts(profile_dir, "build", build_result)
        binary_path = _find_binary(cmake_build_dir, build_target)
        smoke = {"returncode": None, "stdout": "", "stderr": "", "skipped": True}
        if binary_path is not None and model_path is not None:
            smoke = _run(
                [str(binary_path), "-m", str(model_path), "-p", "8", "-n", "4", "-t", "1"],
                cwd=source_repo,
                timeout_seconds=smoke_timeout,
            )
            _write_process_artifacts(profile_dir, "runtime_smoke", smoke)
        cache = _parse_cmake_cache(cmake_build_dir / "CMakeCache.txt")
        compile_evidence = _compile_command_evidence(cmake_build_dir / "compile_commands.json", profile_dir)
        linked = _linked_library_evidence(binary_path, profile_dir)
        runtime_backend_evidence = _runtime_backend_evidence(smoke)
        build_status = _build_status(configure, build_result, binary_path, smoke)
        binary = inspect_binary(profile_id, str(binary_path) if binary_path is not None else None)
        kleidi = kleidiai_status(
            definition=definition,
            build_status=build_status,
            effective_cmake_cache=cache,
            compile_command_evidence=compile_evidence,
            linked_library_evidence=linked,
            runtime_backend_evidence=runtime_backend_evidence,
        )
        native_scope = native_scope_from_payload(profile_payload.get("native_binary_scope"))
        profiles.append(
            BuildProfileEvidence(
                run_id=run_id,
                profile_id=profile_id,
                definition=definition,
                source=source,
                toolchain=toolchain,
                requested_definitions=requested,
                effective_cmake_cache=cache,
                compile_command_evidence=compile_evidence,
                effective_compiler_flags=_effective_flags(compile_evidence),
                effective_linker_flags=validate_compiler_argument_list(
                    profile_payload.get("effective_linker_flags"),
                    label=f"profiles.{profile_id}.effective_linker_flags",
                ),
                cpu_target_policy=str(profile_payload.get("cpu_target_policy", definition.cpu_target_policy)),
                backend_policy=str(profile_payload.get("backend_policy", definition.backend_policy)),
                optimization_policy=str(profile_payload.get("optimization_policy", definition.optimization_policy)),
                target_architecture=str(config.get("target_architecture", "unknown")),
                binary=binary,
                native_binary_scope=native_scope,
                linked_library_evidence=linked,
                runtime_backend_evidence=runtime_backend_evidence,
                build_output={
                    "schema_version": SCHEMA_VERSION,
                    "mode": NATIVE_BUILD_OPERATION,
                    "configure_returncode": configure.get("returncode"),
                    "build_returncode": build_result.get("returncode"),
                    "runtime_smoke_returncode": smoke.get("returncode"),
                    "artifact_subdirectory": str(profile_dir),
                },
                build_status=build_status,
                completeness="complete" if build_status == "completed" else "incomplete",
                kleidiai_status=kleidi,
                eligibility_state=_eligibility_state(
                    build_status=build_status,
                    completeness="complete" if build_status == "completed" else "incomplete",
                    kleidiai_status=kleidi,
                ),
                result_origin=_result_origin(config),
            )
        )
    result = BuildProfileRunResult(
        run_id=run_id,
        operation=NATIVE_BUILD_OPERATION,
        result_origin=_combined_origin([profile.result_origin for profile in profiles]),
        profiles=profiles,
        artifact_directory=str(root),
    )
    atomic_write_json(root / "build_profile_result.json", result.to_dict())
    artifacts = _finalize_artifacts(root)
    validation = validate_artifact_manifest(root)
    result = replace(result, artifacts=BenchmarkArtifactManifest(artifacts=artifacts), artifact_validation=validation)
    atomic_write_json(root / "artifact_validation.json", validation.to_dict())
    artifacts = _finalize_artifacts(root)
    result = replace(result, artifacts=BenchmarkArtifactManifest(artifacts=artifacts))
    atomic_write_json(root / "build_profile_result.json", result.to_dict())
    return result


def _run(command: list[str], *, cwd: Path, timeout_seconds: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def _write_process_artifacts(root: Path, label: str, payload: dict[str, Any]) -> None:
    atomic_write_json(root / f"{label}_process.json", {key: value for key, value in payload.items() if key not in {"stdout", "stderr"}})
    (root / f"{label}_stdout.txt").write_text(str(payload.get("stdout", "")), encoding="utf-8", newline="\n")
    (root / f"{label}_stderr.txt").write_text(str(payload.get("stderr", "")), encoding="utf-8", newline="\n")


def _required_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ArmTuneConfigError(f"{label} must be configured for native build execution")
    return Path(value).expanduser().resolve()


def _optional_model_path(config: dict[str, Any]) -> Path | None:
    model = config.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("path"), str):
        return None
    return Path(model["path"]).expanduser().resolve()


def _find_binary(build_dir: Path, target: str) -> Path | None:
    candidates = [
        build_dir / "bin" / target,
        build_dir / target,
        build_dir / "tools" / target / target,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(path for path in build_dir.rglob(target) if path.is_file())
    return matches[0] if matches else None


def _parse_cmake_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    shutil.copy2(path, path.parent / "CMakeCache.captured.txt")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        left, value = line.split("=", 1)
        key = left.split(":", 1)[0]
        cache[key] = value
    return cache


def _compile_command_evidence(path: Path, artifact_dir: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    shutil.copy2(path, artifact_dir / "compile_commands.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return [{"schema_version": SCHEMA_VERSION, "status": "malformed_compile_commands"}]
    if not isinstance(payload, list):
        return []
    evidence: list[dict[str, Any]] = []
    for item in payload[:200]:
        if isinstance(item, dict):
            command = str(item.get("command") or " ".join(str(arg) for arg in item.get("arguments", [])))
            if any(token in command.lower() for token in ("ggml", "llama", "kleidi", "-mcpu", "-march", "-mtune")):
                evidence.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "file": str(item.get("file", "")),
                        "command": command,
                    }
                )
    return evidence


def _linked_library_evidence(binary_path: Path | None, artifact_dir: Path) -> list[str]:
    if binary_path is None:
        return []
    ldd = shutil.which("ldd")
    file_cmd = shutil.which("file")
    evidence: list[str] = []
    for name, command in {
        "file": [file_cmd, str(binary_path)] if file_cmd else None,
        "ldd": [ldd, str(binary_path)] if ldd else None,
    }.items():
        if command is None:
            continue
        result = _run(command, cwd=binary_path.parent, timeout_seconds=30)
        _write_process_artifacts(artifact_dir, f"linked_{name}", result)
        text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip()
        if text:
            evidence.extend(text.splitlines())
    return evidence


def _runtime_backend_evidence(smoke: dict[str, Any]) -> dict[str, Any]:
    text = f"{smoke.get('stdout', '')}\n{smoke.get('stderr', '')}"
    status = "not_collected" if smoke.get("skipped") else "runtime_unverified"
    if "CPU_KLEIDIAI" in text or "KleidiAI" in text or "kleidiai" in text.lower():
        status = "enabled_and_evidenced"
    if smoke.get("returncode") not in {0, None}:
        status = "runtime_failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime_backend_status": status,
        "runtime_smoke_returncode": smoke.get("returncode"),
    }


def _build_status(configure: dict[str, Any], build: dict[str, Any], binary_path: Path | None, smoke: dict[str, Any]) -> str:
    if configure.get("returncode") != 0:
        return "build_failed"
    if build.get("returncode") != 0:
        return "build_failed"
    if binary_path is None:
        return "build_failed"
    if smoke.get("skipped"):
        return "built_runtime_unverified"
    if smoke.get("returncode") == 132:
        return "illegal_instruction"
    if smoke.get("returncode") not in {0, None}:
        return "runtime_failed"
    return "completed"


def _effective_flags(compile_evidence: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for item in compile_evidence:
        command = str(item.get("command", ""))
        for token in command.split():
            if token.startswith(("-mcpu=", "-march=", "-mtune=")) and token not in flags:
                flags.append(token)
    return flags
