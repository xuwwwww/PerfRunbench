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


BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT = "current-zero-shot"
BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES = "taxonomy-and-decision-rules"
BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES_FEW_SHOT = "taxonomy-rules-and-disjoint-few-shot"
BOOTTRACE_PROMPT_CONTRACT_LEGACY_V1 = "boottrace-prompt-contract-v1"
BOOTTRACE_PROMPT_CONTRACT_VERSION = BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT
SUPPORTED_BOOTTRACE_PROMPT_CONTRACTS = {
    BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT,
    BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES,
    BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES_FEW_SHOT,
    BOOTTRACE_PROMPT_CONTRACT_LEGACY_V1,
}
BOOTTRACE_TAXONOMY_VERSION = "boottrace-taxonomy-v1"
BOOTTRACE_ALLOWED_ROOT_CAUSE_CATEGORIES = [
    "none",
    "device_initialization_timeout",
    "boot_device_missing",
    "firmware_component_failure",
    "recoverable_device_warning",
    "insufficient_evidence",
]
BOOTTRACE_DECISION_ORDER = [
    "If the trace explicitly reaches OS handoff, status is success.",
    "If a recoverable issue occurs but boot continues, status is warning.",
    "If boot stops or cannot select a boot device, status is failure.",
    "If evidence is incomplete or contradictory, status is unknown.",
]
DEFAULT_BOOTTRACE_CALIBRATION_SUITE_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "armtune" / "quality" / "boottrace_calibration_suite.json"
)
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
    prompt_contract_version: str = BOOTTRACE_PROMPT_CONTRACT_VERSION,
    calibration_cases: list[QualityCase] | None = None,
) -> dict[str, Any]:
    version = normalize_boottrace_prompt_contract_version(prompt_contract_version)
    schema_digest = stable_digest(BOOTTRACE_EXPECTED_RESPONSE_SCHEMA)
    rendered_prompts = [
        {
            "schema_version": SCHEMA_VERSION,
            "case_id": case.case_id,
            "prompt": render_boottrace_prompt(
                case,
                prompt_contract_version=version,
                calibration_cases=calibration_cases,
            ),
            "prompt_sha256": stable_text_digest(
                render_boottrace_prompt(
                    case,
                    prompt_contract_version=version,
                    calibration_cases=calibration_cases,
                )
            ),
            "response_format": case.response_format,
            "maximum_predicted_tokens": case.maximum_predicted_tokens,
            "generation_settings": dict(case.generation_settings),
        }
        for case in cases
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": version,
        "taxonomy_version": BOOTTRACE_TAXONOMY_VERSION,
        "suite_manifest": suite_manifest.to_dict(),
        "expected_response_schema": BOOTTRACE_EXPECTED_RESPONSE_SCHEMA,
        "expected_response_schema_sha256": schema_digest,
        "generation_settings": to_plain(generation_settings),
        "normalization_policy": to_plain(normalization_policy),
        "decision_order": list(BOOTTRACE_DECISION_ORDER),
        "allowed_root_cause_category_values": list(BOOTTRACE_ALLOWED_ROOT_CAUSE_CATEGORIES),
        "calibration_examples": _calibration_examples_payload(calibration_cases or [])
        if version == BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES_FEW_SHOT
        else [],
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


def normalize_boottrace_prompt_contract_version(version: str | None) -> str:
    if version is None:
        candidate = BOOTTRACE_PROMPT_CONTRACT_VERSION
    elif isinstance(version, str):
        candidate = version.strip()
    else:
        raise ValueError("BootTrace prompt contract version must be a string")
    if candidate == BOOTTRACE_PROMPT_CONTRACT_LEGACY_V1:
        return BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT
    if candidate not in SUPPORTED_BOOTTRACE_PROMPT_CONTRACTS:
        allowed = sorted(SUPPORTED_BOOTTRACE_PROMPT_CONTRACTS - {BOOTTRACE_PROMPT_CONTRACT_LEGACY_V1})
        raise ValueError(f"unsupported BootTrace prompt contract version: {candidate!r}; allowed values: {allowed}")
    return candidate


def load_boottrace_calibration_cases(path: str | Path | None = None) -> tuple[QualitySuiteManifest, list[QualityCase], dict[str, Any]]:
    suite_path = Path(path) if path is not None else DEFAULT_BOOTTRACE_CALIBRATION_SUITE_PATH
    _, cases, manifest, payload = load_quality_suite(suite_path)
    return manifest, cases, payload


def render_boottrace_prompt(
    case: QualityCase,
    *,
    prompt_contract_version: str = BOOTTRACE_PROMPT_CONTRACT_VERSION,
    calibration_cases: list[QualityCase] | None = None,
) -> str:
    version = normalize_boottrace_prompt_contract_version(prompt_contract_version)
    if version == BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT:
        return case.prompt
    sections = [
        _taxonomy_and_decision_rules(),
    ]
    if version == BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES_FEW_SHOT:
        examples = _few_shot_examples(calibration_cases or [])
        if examples:
            sections.append(examples)
    sections.append("Evaluation case:\n" + case.prompt)
    sections.append(
        "Output contract:\n"
        "- Return exactly one raw compact JSON object.\n"
        "- Do not include Markdown fences or prose outside the JSON object.\n"
        "- Use only these fields: case_id, status, event_sequence_valid, latency_ms, summary, evidence_line_ids.\n"
        "- Copy case_id and evidence_line_ids exactly from the trace.\n"
        "- Use a numeric latency_ms value; do not use null.\n"
        "- Do not invent evidence line IDs.\n"
    )
    return "\n\n".join(sections)


def prompt_suite_from_boottrace_cases(
    cases: list[QualityCase],
    *,
    prompt_contract_version: str = BOOTTRACE_PROMPT_CONTRACT_VERSION,
    calibration_cases: list[QualityCase] | None = None,
) -> dict[str, Any]:
    version = normalize_boottrace_prompt_contract_version(prompt_contract_version)
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Derived ArmTune BootTrace quality prompt suite snapshot.",
        "prompt_contract_version": version,
        "taxonomy_version": BOOTTRACE_TAXONOMY_VERSION,
        "prompts": [
            {
                "prompt_id": case.case_id,
                "prompt": render_boottrace_prompt(
                    case,
                    prompt_contract_version=version,
                    calibration_cases=calibration_cases,
                ),
                "expected_response_mode": case.response_format,
                "max_predict_tokens": case.maximum_predicted_tokens,
                "deterministic_settings": dict(case.generation_settings),
            }
            for case in cases
        ],
    }


def _taxonomy_and_decision_rules() -> str:
    return (
        "BootTrace classification contract:\n"
        "Classify only from the supplied trace lines. Use unknown when the trace is incomplete or contradictory.\n"
        "Decision order:\n"
        + "\n".join(f"{index}. {rule}" for index, rule in enumerate(BOOTTRACE_DECISION_ORDER, start=1))
        + "\nAllowed root_cause_category values for reasoning, not output fields:\n"
        + "\n".join(f"- {value}" for value in BOOTTRACE_ALLOWED_ROOT_CAUSE_CATEGORIES)
        + "\nMap common evidence as follows:\n"
        "- OS handoff or userspace reached -> status success, root cause none.\n"
        "- A failed probe followed by retry success and continued boot -> status warning, root cause recoverable_device_warning.\n"
        "- No bootable device, absent required boot storage, or halted boot selection -> status failure, root cause boot_device_missing.\n"
        "- Device did not become ready before timeout -> status failure, root cause device_initialization_timeout.\n"
        "- Firmware checksum or component verification failure blocks boot -> status failure, root cause firmware_component_failure.\n"
    )


def _few_shot_examples(cases: list[QualityCase]) -> str:
    examples = []
    for case in cases[:3]:
        response = case.expected.get("example_response")
        if not isinstance(response, dict):
            continue
        examples.append(
            "Example:\n"
            + case.prompt
            + "\nExpected JSON:\n"
            + json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
    if not examples:
        return ""
    return "Disjoint calibration examples. These are not evaluation cases:\n\n" + "\n\n".join(examples)


def _calibration_examples_payload(cases: list[QualityCase]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for case in cases[:3]:
        response = case.expected.get("example_response")
        if not isinstance(response, dict):
            continue
        payload.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case.case_id,
                "prompt_sha256": stable_text_digest(case.prompt),
                "example_response_sha256": stable_digest(response),
            }
        )
    return payload


def stable_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
