from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autotune.arm_tune import SCHEMA_VERSION
from autotune.arm_tune.quality.schemas import QualityCase, QualitySuiteManifest
from autotune.arm_tune.quality.suite import load_quality_suite
from autotune.arm_tune.schemas import to_plain


def load_boottrace_suite(path: str | Path):
    return load_quality_suite(path)


BOOTTRACE_PROMPT_CONTRACT_VERSION = "boottrace-prompt-contract-v1"
BOOTTRACE_EXPECTED_RESPONSE_SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "type": "object",
    "required": ["case_id", "status", "event_sequence_valid", "latency_ms", "summary", "evidence_line_ids"],
    "properties": {
        "case_id": {"type": "string"},
        "status": {"type": "string", "enum": ["success", "warning", "failure", "unknown"]},
        "event_sequence_valid": {"type": "boolean"},
        "latency_ms": {"type": "number"},
        "summary": {"type": "string"},
        "evidence_line_ids": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def boottrace_prompt_contract(
    *,
    suite_manifest: QualitySuiteManifest,
    cases: list[QualityCase],
    generation_settings: dict[str, Any],
    normalization_policy: dict[str, Any],
) -> dict[str, Any]:
    schema_digest = stable_digest(BOOTTRACE_EXPECTED_RESPONSE_SCHEMA)
    rendered_prompts = [
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case.case_id,
            "prompt": case.prompt,
            "prompt_sha256": stable_text_digest(case.prompt),
            "response_format": case.response_format,
            "maximum_predicted_tokens": case.maximum_predicted_tokens,
            "generation_settings": dict(case.generation_settings),
        }
        for case in cases
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": BOOTTRACE_PROMPT_CONTRACT_VERSION,
        "suite_manifest": suite_manifest.to_dict(),
        "expected_response_schema": BOOTTRACE_EXPECTED_RESPONSE_SCHEMA,
        "expected_response_schema_sha256": schema_digest,
        "generation_settings": to_plain(generation_settings),
        "normalization_policy": to_plain(normalization_policy),
        "requirements": [
            "Return exactly one JSON object.",
            "Do not wrap output in Markdown fences unless normalization policy explicitly allows extraction.",
            "Do not include prose before or after the JSON object.",
            "Use status enum values only: success, warning, failure, unknown.",
            "Copy evidence_line_ids exactly from the prompt.",
            "Use unknown when the supplied evidence is insufficient.",
            "Do not invent evidence IDs.",
        ],
        "rendered_prompts": rendered_prompts,
    }
    payload["prompt_contract_sha256"] = stable_digest(payload)
    return payload


def stable_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
