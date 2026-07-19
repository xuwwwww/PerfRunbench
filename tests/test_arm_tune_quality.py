from __future__ import annotations

import json
import re
import unittest
import unittest.mock
from pathlib import Path

from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.quality.boottrace import (
    BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT,
    BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES,
    BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES_FEW_SHOT,
    boottrace_prompt_contract,
    load_boottrace_calibration_cases,
    normalize_boottrace_prompt_contract_version,
    render_boottrace_prompt,
)
from autotune.arm_tune.quality.aggregation import aggregate_quality_results
from autotune.arm_tune.quality.normalization import NormalizationPolicy
from autotune.arm_tune.quality.suite import load_quality_suite
from autotune.arm_tune.quality.validators import validate_quality_case


ROOT = Path(__file__).resolve().parents[1]
QUALITY_SUITE = ROOT / "configs" / "armtune" / "quality" / "boottrace_suite.json"
CALIBRATION_SUITE = ROOT / "configs" / "armtune" / "quality" / "boottrace_calibration_suite.json"
TUNING_SUITE = ROOT / "configs" / "armtune" / "quality" / "boottrace_tuning_suite.json"
CASE_AUDIT = ROOT / "configs" / "armtune" / "quality" / "boottrace_case_audit.json"
FIXTURES = Path(__file__).parent / "fixtures" / "arm_tune" / "boottrace"


class ArmTuneQualityTest(unittest.TestCase):
    def test_load_boottrace_suite_and_digest(self) -> None:
        suite_id, cases, manifest, payload = load_quality_suite(QUALITY_SUITE)

        self.assertEqual(suite_id, "synthetic_boottrace_quality_v1")
        self.assertEqual(manifest.case_count, 5)
        self.assertEqual(manifest.required_case_count, 5)
        self.assertEqual(len(manifest.suite_digest), 64)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "boot_success",
                "boot_retry_recovered",
                "boot_config_missing",
                "boot_device_timeout",
                "boot_firmware_component_failure",
            ],
        )

    def test_tuning_suite_is_separate_deterministic_quality_gate(self) -> None:
        _, capability_cases, capability_manifest, _ = load_quality_suite(QUALITY_SUITE)
        tuning_id, tuning_cases, tuning_manifest, tuning_payload = load_quality_suite(TUNING_SUITE)

        self.assertEqual(tuning_id, "synthetic_boottrace_tuning_v1")
        self.assertEqual(tuning_manifest.case_count, 7)
        self.assertEqual(tuning_manifest.required_case_count, 7)
        self.assertEqual(capability_manifest.suite_digest, "42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169")
        self.assertNotEqual(tuning_manifest.suite_digest, capability_manifest.suite_digest)
        self.assertFalse({case.case_id for case in capability_cases} & {case.case_id for case in tuning_cases})
        self.assertEqual(tuning_payload["schema_version"], "1.0")

        expected_tuples = {
            (
                case.expected["status"],
                case.expected["failure_stage"],
                case.expected["root_cause_category"],
                tuple(case.expected["required_evidence_line_ids"]),
            )
            for case in tuning_cases
        }
        self.assertEqual(len(expected_tuples), len(tuning_cases))

    def test_tuning_suite_referenced_evidence_ids_exist(self) -> None:
        _, cases, _, _ = load_quality_suite(TUNING_SUITE)

        for case in cases:
            with self.subTest(case_id=case.case_id):
                prompt_line_ids = set(re.findall(r"^(L\d+)\s", case.prompt, flags=re.MULTILINE))
                required = set(case.expected["required_evidence_line_ids"])
                allowed = set(case.expected["allowed_evidence_line_ids"])
                self.assertTrue(required)
                self.assertLessEqual(required, prompt_line_ids)
                self.assertLessEqual(allowed, prompt_line_ids)

    def test_tuning_gate_valid_canonical_responses_pass(self) -> None:
        _, cases, _, _ = load_quality_suite(TUNING_SUITE)

        for case in cases:
            with self.subTest(case_id=case.case_id):
                result = validate_quality_case(case, _canonical_tuning_response(case))

                self.assertEqual(result.status, "passed")
                self.assertTrue(all(item.status == "passed" for item in result.validator_results))
                self.assertEqual(result.parser_status, "raw_json_valid")

    def test_tuning_gate_rejects_wrong_status_root_and_evidence(self) -> None:
        _, cases, _, _ = load_quality_suite(TUNING_SUITE)
        case = next(item for item in cases if item.case_id == "tuning_device_initialization_timeout")

        wrong_status = validate_quality_case(case, _canonical_tuning_response(case, status="success"))
        wrong_root = validate_quality_case(case, _canonical_tuning_response(case, root_cause_category="none"))
        invented = validate_quality_case(case, _canonical_tuning_response(case, evidence_line_ids=["L121", "L122", "L123", "L124", "L999"]))
        missing = validate_quality_case(case, _canonical_tuning_response(case, evidence_line_ids=["L121", "L122", "L123"]))

        self.assertEqual(wrong_status.status, "failed")
        self.assertIn("wrong_status", wrong_status.failure_categories)
        self.assertEqual(wrong_root.status, "failed")
        self.assertIn("wrong_root_cause", wrong_root.failure_categories)
        self.assertEqual(invented.status, "failed")
        self.assertIn("invented_evidence", invented.failure_categories)
        self.assertEqual(missing.status, "failed")
        self.assertIn("missing_required_evidence", missing.failure_categories)

    def test_tuning_gate_single_fence_normalization_records_evidence_without_semantic_repair(self) -> None:
        _, cases, _, _ = load_quality_suite(TUNING_SUITE)
        case = next(item for item in cases if item.case_id == "tuning_firmware_component_failure")
        policy = NormalizationPolicy(
            enabled=True,
            strict=False,
            allow_markdown_fence=True,
            allow_prose_outside_fence=False,
        )
        canonical = _canonical_tuning_response(case)

        passed = validate_quality_case(case, "```json\n" + canonical + "\n```", normalization_policy=policy)
        wrong = validate_quality_case(
            case,
            "```json\n" + _canonical_tuning_response(case, root_cause_category="none") + "\n```",
            normalization_policy=policy,
        )

        self.assertEqual(passed.status, "passed")
        self.assertEqual(passed.validation_input, "normalized")
        self.assertEqual(passed.parser_status, "normalized_json_valid")
        self.assertTrue(passed.normalization_applied)
        self.assertIn("extract_single_markdown_fenced_json_block", passed.normalization_steps)
        self.assertEqual(len(passed.raw_output_sha256), 64)
        self.assertEqual(len(passed.normalized_output_sha256), 64)
        self.assertEqual(passed.normalized_output, canonical)
        self.assertEqual(wrong.status, "failed")
        self.assertIn('"root_cause_category":"none"', wrong.normalized_output)
        self.assertIn("wrong_root_cause", wrong.failure_categories)

    def test_tuning_prompt_contract_schema_matches_tuning_fields(self) -> None:
        _, cases, manifest, _ = load_quality_suite(TUNING_SUITE)

        contract = boottrace_prompt_contract(
            suite_manifest=manifest,
            cases=cases,
            generation_settings={"temperature": 0.0, "seed": 42},
            normalization_policy=NormalizationPolicy(
                enabled=True,
                strict=False,
                allow_markdown_fence=True,
                allow_prose_outside_fence=False,
            ).to_dict(),
        )

        self.assertIn("failure_stage", contract["expected_response_schema"]["required"])
        self.assertIn("root_cause_category", contract["expected_response_schema"]["required"])
        self.assertEqual(len(contract["expected_response_schema_sha256"]), 64)

    def test_boottrace_case_audit_covers_held_out_suite_without_answer_changes(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        audit = json.loads(CASE_AUDIT.read_text(encoding="utf-8"))
        audited = {item["case_id"]: item for item in audit["cases"]}

        self.assertEqual(set(audited), {case.case_id for case in cases})
        self.assertEqual(audit["schema_version"], "1.0")
        self.assertIn("insufficient_evidence", audit["allowed_root_cause_category_values"])
        for case in cases:
            with self.subTest(case_id=case.case_id):
                item = audited[case.case_id]
                self.assertFalse(item["canonical_answer_change_required"])
                self.assertEqual(item["expected_status"], case.expected["status"])
                self.assertEqual(item["required_evidence_line_ids"], case.expected["required_evidence_line_ids"])

    def test_boottrace_calibration_suite_is_disjoint_from_held_out_evaluation(self) -> None:
        _, held_out_cases, _, _ = load_quality_suite(QUALITY_SUITE)
        manifest, calibration_cases, payload = load_boottrace_calibration_cases(CALIBRATION_SUITE)

        self.assertEqual(manifest.suite_id, "synthetic_boottrace_calibration_v1")
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertFalse({case.case_id for case in held_out_cases} & {case.case_id for case in calibration_cases})
        held_out_evidence = {
            line_id
            for case in held_out_cases
            for line_id in case.expected["allowed_evidence_line_ids"]
        }
        calibration_evidence = {
            line_id
            for case in calibration_cases
            for line_id in case.expected["allowed_evidence_line_ids"]
        }
        self.assertFalse(held_out_evidence & calibration_evidence)
        self.assertTrue(all(isinstance(case.expected.get("example_response"), dict) for case in calibration_cases))

    def test_valid_response_passes_all_validators(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        cases_by_id = {case.case_id: case for case in cases}

        for case_id in ["boot_success", "boot_config_missing", "boot_device_timeout", "boot_firmware_component_failure"]:
            with self.subTest(case_id=case_id):
                response = (FIXTURES / f"{case_id}_response.json").read_text(encoding="utf-8")
                result = validate_quality_case(cases_by_id[case_id], response)

                self.assertEqual(result.status, "passed")
                self.assertTrue(all(item.status == "passed" for item in result.validator_results))

    def test_invalid_json_and_invented_evidence_fail(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        invalid = validate_quality_case(cases[0], "{not-json")
        invented = validate_quality_case(
            cases[0],
            (FIXTURES / "boot_invented_evidence_response.json").read_text(encoding="utf-8"),
        )

        self.assertEqual(invalid.status, "failed")
        self.assertTrue(any(item.validator_type == "valid_json" and item.status == "failed" for item in invalid.validator_results))
        self.assertEqual(invented.status, "failed")
        self.assertTrue(any("invented evidence" in item.message for item in invented.validator_results))
        self.assertIn("invalid_json", invalid.failure_categories)
        self.assertIn("invented_evidence", invented.failure_categories)

    def test_strict_mode_rejects_markdown_fenced_json(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        response = (FIXTURES / "boot_success_response.json").read_text(encoding="utf-8")

        result = validate_quality_case(cases[0], "```json\n" + response + "\n```")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.validation_input, "raw")
        self.assertFalse(result.normalization_applied)
        self.assertIn("markdown_fence_present", result.failure_categories)
        self.assertIn("response_format_violation", result.failure_categories)

    def test_optional_normalization_extracts_single_fenced_json_without_repair(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        response = (FIXTURES / "boot_success_response.json").read_text(encoding="utf-8")

        result = validate_quality_case(
            cases[0],
            "\ufeff```json\r\n" + response + "\r\n```",
            normalization_policy=NormalizationPolicy(
                enabled=True,
                strict=False,
                allow_markdown_fence=True,
                allow_prose_outside_fence=False,
            ),
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.validation_input, "normalized")
        self.assertEqual(result.parser_status, "normalized_json_valid")
        self.assertTrue(result.normalization_applied)
        self.assertIn("extract_single_markdown_fenced_json_block", result.normalization_steps)
        self.assertEqual(len(result.raw_output_sha256), 64)
        self.assertEqual(len(result.normalized_output_sha256), 64)

    def test_normalization_rejects_prose_outside_fence(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        response = (FIXTURES / "boot_success_response.json").read_text(encoding="utf-8")

        result = validate_quality_case(
            cases[0],
            "Here is the result:\n```json\n" + response + "\n```",
            normalization_policy=NormalizationPolicy(
                enabled=True,
                strict=False,
                allow_markdown_fence=True,
                allow_prose_outside_fence=False,
            ),
        )

        self.assertEqual(result.parser_status, "normalization_rejected")
        self.assertEqual(result.status, "failed")
        self.assertIn("response_format_violation", result.failure_categories)

    def test_allowed_values_and_required_substrings_fail_cleanly(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        wrong_status = validate_quality_case(
            cases[1],
            (FIXTURES / "boot_wrong_status_response.json").read_text(encoding="utf-8"),
        )
        missing_summary = json.loads((FIXTURES / "boot_success_response.json").read_text(encoding="utf-8"))
        missing_summary["summary"] = "Started."

        missing = validate_quality_case(cases[0], json.dumps(missing_summary))

        self.assertEqual(wrong_status.status, "failed")
        self.assertTrue(any(item.validator_type == "allowed_values" and item.status == "failed" for item in wrong_status.validator_results))
        self.assertEqual(missing.status, "failed")
        self.assertTrue(any(item.validator_type == "required_substrings" and item.status == "failed" for item in missing.validator_results))

    def test_execution_failure_is_separate_from_validator_failure(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)

        result = validate_quality_case(cases[0], "", request_status="timeout", stream_parse_status="timeout")

        self.assertEqual(result.status, "execution_failed")
        self.assertEqual(result.validator_results, [])
        self.assertIn("request_status=timeout", result.failure_reasons[0])

    def test_aggregate_quality_status(self) -> None:
        _, cases, _, _ = load_quality_suite(QUALITY_SUITE)
        pass_result = validate_quality_case(cases[0], (FIXTURES / "boot_success_response.json").read_text(encoding="utf-8"))
        fail_result = validate_quality_case(cases[0], "{not-json")

        passed = aggregate_quality_results([pass_result])
        failed = aggregate_quality_results([pass_result, fail_result])

        self.assertEqual(passed.quality_gate_status, "passed")
        self.assertEqual(failed.quality_gate_status, "failed")
        self.assertGreater(failed.critical_validator_failures, 0)
        self.assertGreater(failed.failure_category_histogram["invalid_json"], 0)

    def test_boottrace_prompt_contract_digest_is_stable(self) -> None:
        _, cases, manifest, _ = load_quality_suite(QUALITY_SUITE)
        policy = NormalizationPolicy().to_dict()

        first = boottrace_prompt_contract(
            suite_manifest=manifest,
            cases=cases,
            generation_settings={"temperature": 0.0, "seed": 42},
            normalization_policy=policy,
        )
        second = boottrace_prompt_contract(
            suite_manifest=manifest,
            cases=cases,
            generation_settings={"temperature": 0.0, "seed": 42},
            normalization_policy=policy,
        )

        self.assertEqual(first["prompt_contract_sha256"], second["prompt_contract_sha256"])
        self.assertEqual(len(first["expected_response_schema_sha256"]), 64)

    def test_boottrace_prompt_contract_versions_render_distinct_prompts(self) -> None:
        _, cases, manifest, _ = load_quality_suite(QUALITY_SUITE)
        _, calibration_cases, _ = load_boottrace_calibration_cases(CALIBRATION_SUITE)
        policy = NormalizationPolicy().to_dict()

        current_prompt = render_boottrace_prompt(cases[0], prompt_contract_version=BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT)
        rules_prompt = render_boottrace_prompt(cases[0], prompt_contract_version=BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES)
        few_shot_prompt = render_boottrace_prompt(
            cases[0],
            prompt_contract_version=BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES_FEW_SHOT,
            calibration_cases=calibration_cases,
        )
        current_contract = boottrace_prompt_contract(
            suite_manifest=manifest,
            cases=cases,
            generation_settings={"temperature": 0.0, "seed": 42},
            normalization_policy=policy,
            prompt_contract_version=BOOTTRACE_PROMPT_CONTRACT_CURRENT_ZERO_SHOT,
        )
        rules_contract = boottrace_prompt_contract(
            suite_manifest=manifest,
            cases=cases,
            generation_settings={"temperature": 0.0, "seed": 42},
            normalization_policy=policy,
            prompt_contract_version=BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES,
        )
        few_shot_contract = boottrace_prompt_contract(
            suite_manifest=manifest,
            cases=cases,
            generation_settings={"temperature": 0.0, "seed": 42},
            normalization_policy=policy,
            prompt_contract_version=BOOTTRACE_PROMPT_CONTRACT_TAXONOMY_RULES_FEW_SHOT,
            calibration_cases=calibration_cases,
        )

        self.assertEqual(current_prompt, cases[0].prompt)
        self.assertIn("Decision order", rules_prompt)
        self.assertNotIn("Disjoint calibration examples", rules_prompt)
        self.assertIn("Disjoint calibration examples", few_shot_prompt)
        self.assertIn("dev_no_boot_device_failure", few_shot_prompt)
        self.assertNotEqual(current_contract["prompt_contract_sha256"], rules_contract["prompt_contract_sha256"])
        self.assertNotEqual(rules_contract["prompt_contract_sha256"], few_shot_contract["prompt_contract_sha256"])
        self.assertEqual(len(few_shot_contract["calibration_examples"]), 3)

    def test_unknown_boottrace_prompt_contract_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported BootTrace prompt contract version"):
            normalize_boottrace_prompt_contract_version("made-up-contract")

    def test_suite_rejects_duplicate_case_id(self) -> None:
        payload = json.loads(QUALITY_SUITE.read_text(encoding="utf-8"))
        payload["cases"].append(dict(payload["cases"][0]))
        with self.assertRaisesRegex(ArmTuneConfigError, "duplicate quality case_id"):
            with unittest.mock.patch.object(Path, "read_text", return_value=json.dumps(payload)):
                load_quality_suite(Path("synthetic.json"))

def _canonical_tuning_response(
    case,
    *,
    status: str | None = None,
    failure_stage: str | None = None,
    root_cause_category: str | None = None,
    evidence_line_ids: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "case_id": case.case_id,
            "status": status or case.expected["status"],
            "failure_stage": failure_stage or case.expected["failure_stage"],
            "root_cause_category": root_cause_category or case.expected["root_cause_category"],
            "evidence_line_ids": evidence_line_ids or case.expected["required_evidence_line_ids"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    unittest.main()
