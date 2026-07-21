from __future__ import annotations

from typing import Any

from autotune.arm_tune.build_profiles.definitions import available_profile_ids
from autotune.arm_tune.errors import ArmTuneConfigError


ALLOWED_CMAKE_DEFINITIONS = {
    "CMAKE_BUILD_TYPE",
    "CMAKE_EXPORT_COMPILE_COMMANDS",
    "CMAKE_INTERPROCEDURAL_OPTIMIZATION",
    "GGML_NATIVE",
    "GGML_CPU_ALL_VARIANTS",
    "GGML_CPU_KLEIDIAI",
    "GGML_BLAS",
    "GGML_BLAS_VENDOR",
    "LLAMA_NATIVE",
    "LLAMA_KLEIDIAI",
}
ALLOWED_CPU_TARGET_POLICIES = {
    "portable",
    "upstream_default",
    "native_current_cpu",
    "explicit_cpu_target",
}
ALLOWED_BACKEND_POLICIES = {
    "default_cpu",
    "kleidiai",
    "openblas_prefill_only",
}
ALLOWED_OPTIMIZATION_POLICIES = {
    "standard_release",
    "lto",
}
ALLOWED_PORTABILITY_STATES = {
    "portable",
    "architecture_bound",
    "feature_bound",
    "cpu_model_bound",
    "exact_target_only",
}


def validate_profile_payload(profile: dict[str, Any]) -> None:
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or profile_id not in available_profile_ids():
        raise ArmTuneConfigError(f"unsupported profile_id: {profile_id!r}")
    for key in ("cpu_target_policy", "backend_policy", "optimization_policy", "portability_scope"):
        if key in profile and not isinstance(profile[key], str):
            raise ArmTuneConfigError(f"{key} must be a string")
    _require_member(profile, "cpu_target_policy", ALLOWED_CPU_TARGET_POLICIES)
    _require_member(profile, "backend_policy", ALLOWED_BACKEND_POLICIES)
    _require_member(profile, "optimization_policy", ALLOWED_OPTIMIZATION_POLICIES)
    _require_member(profile, "portability_scope", ALLOWED_PORTABILITY_STATES)
    if "requested_definitions" in profile:
        validate_cmake_definitions(profile["requested_definitions"], label=f"profiles.{profile_id}.requested_definitions")
    if "effective_cmake_cache" in profile:
        validate_cmake_definitions(profile["effective_cmake_cache"], label=f"profiles.{profile_id}.effective_cmake_cache", allow_cache_extra=True)


def validate_cmake_definitions(value: Any, *, label: str, allow_cache_extra: bool = False) -> None:
    if not isinstance(value, dict):
        raise ArmTuneConfigError(f"{label} must be an object")
    for key, item in value.items():
        key_text = str(key)
        if not allow_cache_extra and key_text not in ALLOWED_CMAKE_DEFINITIONS:
            raise ArmTuneConfigError(f"{label} contains unsupported CMake definition: {key_text}")
        if _looks_like_shell_fragment(str(item)):
            raise ArmTuneConfigError(f"{label}.{key_text} must not contain shell fragments")


def validate_compiler_argument_list(value: Any, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ArmTuneConfigError(f"{label} must be a list")
    args: list[str] = []
    for item in value:
        text = str(item)
        if _looks_like_shell_fragment(text):
            raise ArmTuneConfigError(f"{label} must not contain shell fragments")
        args.append(text)
    return args


def _require_member(profile: dict[str, Any], key: str, allowed: set[str]) -> None:
    if key in profile and str(profile[key]) not in allowed:
        raise ArmTuneConfigError(f"{key} must be one of {sorted(allowed)}")


def _looks_like_shell_fragment(value: str) -> bool:
    dangerous = ["&&", "||", ";", "`", "$(", ">", "<", "|", "\n", "\r"]
    return any(token in value for token in dangerous)
