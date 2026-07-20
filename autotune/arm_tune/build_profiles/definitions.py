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
    ),
    "arm-native": BuildProfileDefinition(
        profile_id="arm-native",
        description="Native Arm build profile using target architecture compiler defaults.",
        cmake_definitions={
            "CMAKE_BUILD_TYPE": "Release",
            "LLAMA_NATIVE": "ON",
        },
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
    ),
}


def available_profile_ids() -> list[str]:
    return sorted(PROFILE_DEFINITIONS)


def get_profile_definition(profile_id: str) -> BuildProfileDefinition:
    if profile_id not in PROFILE_DEFINITIONS:
        raise KeyError(profile_id)
    return PROFILE_DEFINITIONS[profile_id]
