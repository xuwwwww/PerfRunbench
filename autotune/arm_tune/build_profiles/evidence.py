from __future__ import annotations

from typing import Any

from autotune.arm_tune.build_profiles.schemas import BuildProfileDefinition


def kleidiai_status(
    *,
    definition: BuildProfileDefinition,
    build_status: str,
    effective_cmake_cache: dict[str, str],
    compile_command_evidence: list[dict[str, Any]],
    linked_library_evidence: list[str],
) -> str:
    if not definition.kleidiai_requested:
        return "unavailable"
    if build_status == "failed":
        return "build_failed"
    requested = _enabled(definition.cmake_definitions.get("LLAMA_KLEIDIAI")) or _enabled(
        effective_cmake_cache.get("LLAMA_KLEIDIAI")
    )
    if not requested:
        return "unavailable"
    evidence_text = "\n".join(
        [
            *linked_library_evidence,
            *(
                " ".join(str(value) for value in item.values())
                for item in compile_command_evidence
                if isinstance(item, dict)
            ),
        ]
    ).lower()
    if "kleidiai" in evidence_text or "kleidi" in evidence_text:
        return "enabled_and_evidenced"
    return "requested_but_unverified"


def _enabled(value: str | None) -> bool:
    return str(value or "").upper() in {"ON", "TRUE", "1", "YES"}
