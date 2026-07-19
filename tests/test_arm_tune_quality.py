from __future__ import annotations

import json
import unittest
import unittest.mock
from pathlib import Path

from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.quality.boottrace import boottrace_prompt_contract
from autotune.arm_tune.quality.aggregation import aggregate_quality_results
from autotune.arm_tune.quality.normalization import NormalizationPolicy
from autotune.arm_tune.quality.suite import load_quality_suite
from autotune.arm_tune.quality.validators import validate_quality_case


ROOT = Path(__file__).resolve().parents[1]
QUALITY_SUITE = ROOT / "configs" / "armtune" / "quality" / "boottrace_suite.json"
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

    def test_suite_rejects_duplicate_case_id(self) -> None:
        payload = json.loads(QUALITY_SUITE.read_text(encoding="utf-8"))
        payload["cases"].append(dict(payload["cases"][0]))
        with self.assertRaisesRegex(ArmTuneConfigError, "duplicate quality case_id"):
            with unittest.mock.patch.object(Path, "read_text", return_value=json.dumps(payload)):
                load_quality_suite(Path("synthetic.json"))


if __name__ == "__main__":
    unittest.main()
