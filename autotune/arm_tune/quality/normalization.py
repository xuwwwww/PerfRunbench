from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.schemas import to_plain


NORMALIZATION_POLICY_VERSION = "generic-single-fence-v1"
FENCED_JSON_PATTERN = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class NormalizationPolicy:
    enabled: bool = False
    strict: bool = True
    allow_markdown_fence: bool = False
    allow_prose_outside_fence: bool = False
    policy_version: str = NORMALIZATION_POLICY_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NormalizationResult:
    raw_output: str
    normalized_output: str
    raw_output_sha256: str
    normalized_output_sha256: str
    normalization_policy_version: str
    normalization_applied: bool
    normalization_steps: list[str] = field(default_factory=list)
    parser_status: str = "raw"
    validation_input: str = "raw"
    rejection_reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def normalize_output(output: str, policy: NormalizationPolicy | None = None) -> NormalizationResult:
    active_policy = policy or NormalizationPolicy()
    raw = output
    raw_hash = _sha256_text(raw)
    if not active_policy.enabled or active_policy.strict:
        return NormalizationResult(
            raw_output=raw,
            normalized_output=raw,
            raw_output_sha256=raw_hash,
            normalized_output_sha256=raw_hash,
            normalization_policy_version=active_policy.policy_version,
            normalization_applied=False,
            parser_status="raw",
            validation_input="raw",
        )

    text = raw
    steps: list[str] = []
    rejection_reasons: list[str] = []
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
        steps.append("remove_utf8_bom")
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized_newlines != text:
        text = normalized_newlines
        steps.append("normalize_line_endings")
    stripped = text.strip()
    if stripped != text:
        text = stripped
        steps.append("trim_outer_whitespace")

    matches = list(FENCED_JSON_PATTERN.finditer(text))
    if len(matches) > 1:
        rejection_reasons.append("multiple_markdown_fenced_blocks")
    elif len(matches) == 1:
        if not active_policy.allow_markdown_fence:
            rejection_reasons.append("markdown_fence_not_allowed")
        else:
            before = text[: matches[0].start()].strip()
            after = text[matches[0].end() :].strip()
            if (before or after) and not active_policy.allow_prose_outside_fence:
                rejection_reasons.append("prose_outside_markdown_fence")
            if not rejection_reasons:
                text = matches[0].group(1).strip()
                steps.append("extract_single_markdown_fenced_json_block")

    normalized_hash = _sha256_text(text)
    if rejection_reasons:
        return NormalizationResult(
            raw_output=raw,
            normalized_output=text,
            raw_output_sha256=raw_hash,
            normalized_output_sha256=normalized_hash,
            normalization_policy_version=active_policy.policy_version,
            normalization_applied=bool(steps),
            normalization_steps=steps,
            parser_status="normalization_rejected",
            validation_input="raw",
            rejection_reasons=rejection_reasons,
        )
    return NormalizationResult(
        raw_output=raw,
        normalized_output=text,
        raw_output_sha256=raw_hash,
        normalized_output_sha256=normalized_hash,
        normalization_policy_version=active_policy.policy_version,
        normalization_applied=text != raw,
        normalization_steps=steps,
        parser_status="normalized" if text != raw else "raw",
        validation_input="normalized" if text != raw else "raw",
    )


def normalization_policy_from_payload(payload: dict[str, Any] | None) -> NormalizationPolicy:
    if payload is None:
        return NormalizationPolicy()
    if not isinstance(payload, dict):
        raise ValueError("normalization must be an object")
    unknown = set(payload) - {
        "schema_version",
        "enabled",
        "strict",
        "allow_markdown_fence",
        "allow_prose_outside_fence",
        "policy_version",
    }
    if unknown:
        raise ValueError(f"unknown normalization field(s): {sorted(unknown)}")
    if "schema_version" in payload and payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported normalization schema_version: {payload.get('schema_version')!r}")
    return NormalizationPolicy(
        enabled=bool(payload.get("enabled", False)),
        strict=bool(payload.get("strict", True)),
        allow_markdown_fence=bool(payload.get("allow_markdown_fence", False)),
        allow_prose_outside_fence=bool(payload.get("allow_prose_outside_fence", False)),
        policy_version=str(payload.get("policy_version", NORMALIZATION_POLICY_VERSION)),
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
