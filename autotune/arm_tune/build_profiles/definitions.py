from __future__ import annotations

from autotune.arm_tune.build_profiles.schemas import BuildProfileDefinition


PROFILE_DEFINITIONS = {
    "portable-baseline": BuildProfileDefinition(
        profile_id="portable-baseline",
        description="Portable llama.cpp baseline without architecture-native build assumptions.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "LLAMA_NATIVE": "OFF",
        },
        cpu_target_policy="portable",
        portability_scope="architecture_bound",
    ),
    "arm-native": BuildProfileDefinition(
        profile_id="arm-native",
        description="Native Arm build profile using target architecture compiler defaults.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "LLAMA_NATIVE": "ON",
        },
        cpu_target_policy="native_current_cpu",
        portability_scope="cpu_model_bound",
    ),
    "arm-optimized": BuildProfileDefinition(
        profile_id="arm-optimized",
        description="Arm optimized profile that may request KleidiAI but requires real evidence before claims.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "LLAMA_NATIVE": "ON",
            "LLAMA_KLEIDIAI": "ON",
        },
        kleidiai_requested=True,
        cpu_target_policy="native_current_cpu",
        backend_policy="kleidiai",
        portability_scope="cpu_model_bound",
    ),
    "upstream-default": BuildProfileDefinition(
        profile_id="upstream-default",
        description="Pinned upstream llama.cpp Release build without PerfRunbench architecture overrides.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        },
        cpu_target_policy="upstream_default",
        backend_policy="default_cpu",
        optimization_policy="standard_release",
        portability_scope="architecture_bound",
    ),
    "portable-reference": BuildProfileDefinition(
        profile_id="portable-reference",
        description="Portable reference build when GGML multi-variant CPU support is available.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "GGML_NATIVE": "OFF",
            "GGML_CPU_ALL_VARIANTS": "ON",
        },
        cpu_target_policy="portable",
        backend_policy="default_cpu",
        optimization_policy="standard_release",
        portability_scope="portable",
    ),
    "native-current-cpu": BuildProfileDefinition(
        profile_id="native-current-cpu",
        description="Native build for the current CPU using GGML_NATIVE with compile-command evidence.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "GGML_NATIVE": "ON",
        },
        cpu_target_policy="native_current_cpu",
        backend_policy="default_cpu",
        optimization_policy="standard_release",
        portability_scope="cpu_model_bound",
    ),
    "explicit-current-cpu": BuildProfileDefinition(
        profile_id="explicit-current-cpu",
        description="Explicit compiler CPU target build after allowlisted compiler probe succeeds.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "GGML_NATIVE": "OFF",
        },
        cpu_target_policy="explicit_cpu_target",
        backend_policy="default_cpu",
        optimization_policy="standard_release",
        portability_scope="cpu_model_bound",
    ),
    "explicit-neoverse-v1": BuildProfileDefinition(
        profile_id="explicit-neoverse-v1",
        description="Explicit Neoverse-V1 compiler target for C7g validation after compiler probe succeeds.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "GGML_NATIVE": "OFF",
            "CMAKE_C_FLAGS": "-mcpu=neoverse-v1",
            "CMAKE_CXX_FLAGS": "-mcpu=neoverse-v1",
        },
        cpu_target_policy="explicit_cpu_target",
        backend_policy="default_cpu",
        optimization_policy="standard_release",
        portability_scope="cpu_model_bound",
    ),
    "native-kleidiai": BuildProfileDefinition(
        profile_id="native-kleidiai",
        description="Native CPU build requesting GGML_CPU_KLEIDIAI, eligible only with build and runtime backend evidence.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "GGML_NATIVE": "ON",
            "GGML_CPU_KLEIDIAI": "ON",
        },
        kleidiai_requested=True,
        cpu_target_policy="native_current_cpu",
        backend_policy="kleidiai",
        optimization_policy="standard_release",
        portability_scope="feature_bound",
    ),
    "native-kleidiai-lto": BuildProfileDefinition(
        profile_id="native-kleidiai-lto",
        description="Native KleidiAI build with CMake interprocedural optimization after LTO probe succeeds.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "GGML_NATIVE": "ON",
            "GGML_CPU_KLEIDIAI": "ON",
            "CMAKE_INTERPROCEDURAL_OPTIMIZATION": "ON",
        },
        kleidiai_requested=True,
        cpu_target_policy="native_current_cpu",
        backend_policy="kleidiai",
        optimization_policy="lto",
        portability_scope="feature_bound",
    ),
}


def available_profile_ids() -> list[str]:
    return sorted(PROFILE_DEFINITIONS)


def get_profile_definition(profile_id: str) -> BuildProfileDefinition:
    if profile_id not in PROFILE_DEFINITIONS:
        raise KeyError(profile_id)
    return PROFILE_DEFINITIONS[profile_id]
