from __future__ import annotations

import json
import unittest
import unittest.mock
from pathlib import Path

from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.quality.aggregation import aggregate_quality_results
from autotune.arm_tune.quality.normalization import NormalizationPolicy
from autotune.arm_tune.quality.suite import load_quality_suite
from autotune.arm_tune.quality.validators import validate_quality_case


GENERIC_SUITE = {
    "schema_version": "1.0",
    "suite_id": "generic_correctness_v1",
    "description": "Generic command-output correctness suite.",
    "cases": [
        {
            "case_id": "json_health",
            "category": "generic_json",
            "prompt": "Return a JSON health object.",
            "response_format": "json",
            "required": True,
            "maximum_predicted_tokens": 64,
            "expected": {
                "case_id": "json_health",
                "status": "ok",
                "required_evidence_line_ids": ["E001"],
                "allowed_evidence_line_ids": ["E001", "E002"],
            },
            "validators": [
                {"type": "nonempty_output"},
                {"type": "valid_json"},
                {
                    "type": "json_schema",
                    "expected": {
                        "required": ["case_id", "status", "value", "evidence_line_ids"],
                        "properties": {
                            "case_id": {"type": "string"},
                            "status": {"type": "string"},
                            "value": {"type": "number"},
                            "evidence_line_ids": {"type": "array"},
                        },
                    },
                },
                {"type": "exact_field", "field": "case_id", "expected": "json_health"},
                {"type": "allowed_values", "field": "status", "expected": ["ok"]},
                {"type": "numeric_range", "field": "value", "expected": {"min": 1, "max": 10}},
                {
                    "type": "evidence_line_ids",
                    "field": "evidence_line_ids",
                    "expected": {"required": ["E001"], "allowed": ["E001", "E002"]},
                },
            ],
        }
    ],
}


class ArmTuneGenericQualityTest(unittest.TestCase):
    def test_load_generic_suite_and_digest(self) -> None:
        with unittest.mock.patch.object(Path, "read_text", return_value=json.dumps(GENERIC_SUITE)):
            suite_id, cases, manifest, payload = load_quality_suite(Path("generic_suite.json"))

        self.assertEqual(suite_id, "generic_correctness_v1")
        self.assertEqual(manifest.case_count, 1)
        self.assertEqual(manifest.required_case_count, 1)
        self.assertEqual(len(manifest.suite_digest), 64)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(cases[0].case_id, "json_health")

    def test_valid_response_passes_all_generic_validators(self) -> None:
        case = _load_case()
        response = json.dumps(
            {
                "case_id": "json_health",
                "status": "ok",
                "value": 5,
                "evidence_line_ids": ["E001"],
            }
        )

        result = validate_quality_case(case, response)

        self.assertEqual(result.status, "passed")
        self.assertTrue(all(item.status == "passed" for item in result.validator_results))
        self.assertEqual(result.parser_status, "raw_json_valid")

    def test_invalid_json_and_invented_evidence_fail_without_workload_categories(self) -> None:
        case = _load_case()
        invalid = validate_quality_case(case, "{not-json")
        invented = validate_quality_case(
            case,
            json.dumps(
                {
                    "case_id": "json_health",
                    "status": "ok",
                    "value": 5,
                    "evidence_line_ids": ["E001", "E999"],
                }
            ),
        )

        self.assertEqual(invalid.status, "failed")
        self.assertIn("invalid_json", invalid.failure_categories)
        self.assertNotIn("model_capability_failure", invalid.failure_categories)
        self.assertEqual(invented.status, "failed")
        self.assertIn("invented_evidence", invented.failure_categories)
        self.assertNotIn("wrong_root_cause", invented.failure_categories)

    def test_optional_single_fence_normalization_is_generic(self) -> None:
        case = _load_case()
        response = json.dumps(
            {
                "case_id": "json_health",
                "status": "ok",
                "value": 5,
                "evidence_line_ids": ["E001"],
            },
            separators=(",", ":"),
        )
        policy = NormalizationPolicy(
            enabled=True,
            strict=False,
            allow_markdown_fence=True,
            allow_prose_outside_fence=False,
        )

        result = validate_quality_case(case, "```json\n" + response + "\n```", normalization_policy=policy)

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.validation_input, "normalized")
        self.assertEqual(result.parser_status, "normalized_json_valid")
        self.assertEqual(result.normalization_policy_version, "generic-single-fence-v1")
        self.assertIn("extract_single_markdown_fenced_json_block", result.normalization_steps)

    def test_aggregate_quality_status(self) -> None:
        case = _load_case()
        passed = validate_quality_case(
            case,
            json.dumps(
                {
                    "case_id": "json_health",
                    "status": "ok",
                    "value": 5,
                    "evidence_line_ids": ["E001"],
                }
            ),
        )
        failed = validate_quality_case(case, "{not-json")

        passed_aggregate = aggregate_quality_results([passed])
        failed_aggregate = aggregate_quality_results([passed, failed])

        self.assertEqual(passed_aggregate.quality_gate_status, "passed")
        self.assertEqual(failed_aggregate.quality_gate_status, "failed")
        self.assertGreater(failed_aggregate.critical_validator_failures, 0)
        self.assertGreater(failed_aggregate.failure_category_histogram["invalid_json"], 0)

    def test_suite_rejects_duplicate_case_id(self) -> None:
        payload = dict(GENERIC_SUITE)
        payload["cases"] = list(GENERIC_SUITE["cases"]) + [dict(GENERIC_SUITE["cases"][0])]
        with self.assertRaisesRegex(ArmTuneConfigError, "duplicate quality case_id"):
            with unittest.mock.patch.object(Path, "read_text", return_value=json.dumps(payload)):
                load_quality_suite(Path("synthetic.json"))


def _load_case():
    with unittest.mock.patch.object(Path, "read_text", return_value=json.dumps(GENERIC_SUITE)):
        _, cases, _, _ = load_quality_suite(Path("generic_suite.json"))
    return cases[0]


if __name__ == "__main__":
    unittest.main()
