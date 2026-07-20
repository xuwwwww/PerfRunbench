from __future__ import annotations

import json
import platform
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from autotune.cli import main as autotune_main
from autotune.recommendation.apply import ApplyOptions, apply_recommendation_bundle, promote_recommendation_bundle, recommendation_status, rollback_recommendation_bundle
from autotune.recommendation.bundle import (
    create_recommendation_bundle,
    generate_bundle_from_source,
    recommendation_path,
    validate_recommendation_bundle,
    write_recommendation_bundle,
)
from autotune.recommendation.contamination import classify_trial_contamination
from autotune.recommendation.correctness import correctness_passed, evaluate_correctness_contract
from autotune.recommendation.fingerprints import collect_environment_fingerprint, compare_environment_fingerprints
from autotune.recommendation.schemas import RecommendationError
from autotune.recommendation.statistics import StatisticalGuardConfig, evaluate_statistical_evidence


class RecommendationSafetyTest(unittest.TestCase):
    def test_bundle_contains_executable_selected_config_and_rejects_fixture_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            passport = _write_passport(temp_path)
            bundle = _bundle(temp_path, passport)
            result = validate_recommendation_bundle(bundle)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(bundle["selected_configuration"]["launch_spec"]["argv"][0], sys.executable)

            bundle["provenance"]["result_origin"] = "test_fixture"
            result = validate_recommendation_bundle(bundle)
            self.assertEqual(result["status"], "fail")
            self.assertIn("fixture_source_rejected", _codes(result))

    def test_generator_rejects_inconclusive_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            passport = _write_passport(temp_path)
            selected_config = temp_path / "selected.json"
            selected_config.write_text(json.dumps(_configuration()), encoding="utf-8")
            source_result = temp_path / "source.json"
            source_result.write_text(
                json.dumps({"status": "inconclusive", "result_origin": "measured"}),
                encoding="utf-8",
            )
            with self.assertRaises(RecommendationError):
                generate_bundle_from_source(
                    passport_path=passport,
                    source_result_path=source_result,
                    source_run_id="run1",
                    selected_configuration_path=selected_config,
                )

    def test_environment_hard_mismatch_requires_retune(self) -> None:
        expected = collect_environment_fingerprint(
            executable_path=sys.executable,
            runtime_identity="generic-command",
            runtime_version=platform.python_version(),
        )
        observed = json.loads(json.dumps(expected))
        observed["hard"]["architecture"] = "different-arch"

        result = compare_environment_fingerprints(expected, observed)

        self.assertEqual(result["status"], "hard_mismatch")
        self.assertTrue(result["retune_required"])
        self.assertIn("stale_environment", _codes(result))

    def test_equivalent_allowed_cpu_masks_do_not_require_retune(self) -> None:
        expected = collect_environment_fingerprint(
            executable_path=sys.executable,
            runtime_identity="generic-command",
            runtime_version=platform.python_version(),
        )
        expected["hard"]["allowed_cpu_mask"] = "0-7"
        observed = json.loads(json.dumps(expected))
        observed["hard"]["allowed_cpu_mask"] = "0,1,2,3,4,5,6,7"

        result = compare_environment_fingerprints(expected, observed)

        self.assertEqual(result["status"], "match")
        self.assertFalse(result["retune_required"])
        self.assertNotIn("stale_environment", _codes(result))

    def test_hex_allowed_cpu_mask_matches_equivalent_cpu_list(self) -> None:
        expected = collect_environment_fingerprint(
            executable_path=sys.executable,
            runtime_identity="generic-command",
            runtime_version=platform.python_version(),
        )
        expected["hard"]["allowed_cpu_mask"] = "0xff"
        observed = json.loads(json.dumps(expected))
        observed["hard"]["allowed_cpu_mask"] = "0,1,2,3,4,5,6,7"

        result = compare_environment_fingerprints(expected, observed)

        self.assertEqual(result["status"], "match")
        self.assertFalse(result["retune_required"])
        self.assertNotIn("stale_environment", _codes(result))

    def test_correctness_contract_uses_allowlisted_validators(self) -> None:
        contract = {
            "validators": [
                {"type": "exit_code", "expected": 0},
                {"type": "required_metric", "metric": "throughput"},
                {"type": "metric_finite", "metric": "throughput"},
                {"type": "metric_min", "metric": "throughput", "min": 100.0},
                {"type": "metric_max", "metric": "latency_ms", "max": 10.0},
                {"type": "request_completed"},
                {"type": "stream_parsed"},
                {"type": "cleanup_passed"},
            ]
        }
        observation = {
            "exit_code": 0,
            "metrics": {"throughput": 120.0, "latency_ms": 8.0},
            "request_completed": True,
            "stream_parsed": True,
            "cleanup_passed": True,
        }
        results = evaluate_correctness_contract(contract, observation)
        self.assertTrue(correctness_passed(results))

        rejected = evaluate_correctness_contract({"validators": [{"type": "shell_command"}]}, observation)
        self.assertFalse(correctness_passed(rejected))
        self.assertEqual(rejected[0].code, "validator_not_allowed")

    def test_statistics_require_fresh_paired_confirmation(self) -> None:
        search_only = {
            "trials": [
                _trial("baseline", "search", "s0", 100.0),
                _trial("selected", "search", "s0", 130.0),
            ]
        }
        search_result = evaluate_statistical_evidence(search_only, StatisticalGuardConfig(metric="decode_tokens_per_second", min_valid_trials=2))
        self.assertEqual(search_result["status"], "inconclusive")
        self.assertIn("search_only_recommendation_rejected", _codes(search_result))

        evidence = {
            "trials": [
                _trial("baseline", "warmup", "w0", 50.0),
                _trial("baseline", "confirmation", "p0", 100.0),
                _trial("selected", "confirmation", "p0", 130.0),
                _trial("baseline", "confirmation", "p1", 101.0),
                _trial("selected", "confirmation", "p1", 131.0),
                _trial("selected", "confirmation", "p2", 132.0, failed=True),
                _trial("baseline", "confirmation", "p2", 102.0, trial_status="contaminated"),
                _trial("baseline", "confirmation", "p3", 100.0),
                _trial("selected", "confirmation", "p3", 129.0),
            ]
        }
        result = evaluate_statistical_evidence(evidence, StatisticalGuardConfig(metric="decode_tokens_per_second", min_valid_trials=2))
        self.assertEqual(result["status"], "recommended")
        self.assertGreater(result["improvement_percent"], 2.0)
        self.assertEqual(result["warmup_trial_count"], 1)
        self.assertEqual(len(result["failed_trials"]), 1)
        self.assertEqual(result["contaminated_trial_count"], 1)

    def test_contamination_keeps_not_available_and_marks_contaminated(self) -> None:
        result = classify_trial_contamination({"background_cpu_percent": 90.0})

        self.assertEqual(result["trial_status"], "contaminated")
        codes = _codes(result)
        self.assertIn("trial_contaminated", codes)
        self.assertIn("load_average_not_available", codes)

    def test_process_apply_canary_lease_promote_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            passport = _write_passport(temp_path)
            bundle = _bundle(temp_path, passport)
            bundle_path = recommendation_path("phase10fake", root=temp_path / "recommendations")
            write_recommendation_bundle(bundle, bundle_path)

            dry_run = apply_recommendation_bundle(bundle_path, options=ApplyOptions(dry_run=True, root=temp_path / "recommendations"))
            self.assertEqual(dry_run["status"], "dry_run_passed")

            applied = apply_recommendation_bundle(
                bundle_path,
                options=ApplyOptions(lease_minutes=1, canary_runs=1, root=temp_path / "recommendations"),
            )
            self.assertEqual(applied["status"], "applied")
            self.assertEqual(applied["canary"]["status"], "passed")

            status = recommendation_status(bundle_path, root=temp_path / "recommendations")
            self.assertEqual(status["lock_state"], "active")
            self.assertEqual(status["lease_state"], "active")

            promoted = promote_recommendation_bundle(bundle_path, root=temp_path / "recommendations")
            self.assertEqual(promoted["state"], "promoted")

            rolled_back = rollback_recommendation_bundle(bundle_path, root=temp_path / "recommendations")
            self.assertEqual(rolled_back["state"], "rolled_back")

    def test_top_level_recommendation_cli_dispatcher(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            temp_path = Path(temp_dir)
            passport = _write_passport(temp_path)
            selected_config = temp_path / "selected.json"
            selected_config.write_text(json.dumps(_configuration()), encoding="utf-8")
            source_result = temp_path / "source.json"
            source_result.write_text(
                json.dumps(
                    {
                        "status": "recommended",
                        "result_origin": "measured",
                        "statistical_evidence": {"status": "recommended", "classification": "improved"},
                    }
                ),
                encoding="utf-8",
            )
            output = temp_path / "bundle.json"

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = autotune_main(
                    [
                        "recommendation",
                        "generate",
                        "--passport",
                        str(passport),
                        "--run-id",
                        "cli-run",
                        "--source-result",
                        str(source_result),
                        "--selected-config",
                        str(selected_config),
                        "--objective-metric",
                        "throughput",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue(output.exists())
            with redirect_stdout(StringIO()):
                self.assertEqual(autotune_main(["recommendation", "validate", "--bundle", str(output)]), 0)


def _bundle(temp_path: Path, passport: Path) -> dict:
    return create_recommendation_bundle(
        source_run_id="phase10fake",
        source_passport_path=passport,
        status="eligible_preliminary",
        objective={"metric": "throughput", "direction": "maximize"},
        baseline_configuration=_configuration(configuration_id="baseline"),
        selected_configuration=_configuration(configuration_id="selected"),
        environment_fingerprint=collect_environment_fingerprint(
            executable_path=sys.executable,
            runtime_identity="generic-command",
            runtime_version=platform.python_version(),
        ),
        constraints={
            "canary": {
                "throughput_metric": "throughput",
                "baseline_throughput_floor": 100.0,
                "latency_metric": "latency_ms",
                "baseline_latency_ceiling": 10.0,
            }
        },
        statistical_evidence={"status": "recommended"},
        provenance={"result_origin": "measured"},
    )


def _configuration(*, configuration_id: str = "selected") -> dict:
    return {
        "configuration_id": configuration_id,
        "launch_spec": {
            "schema_version": "1.0",
            "adapter": "generic-command",
            "argv": [
                sys.executable,
                "-c",
                "import json; print(json.dumps({'throughput': 120.0, 'latency_ms': 8.0}))",
            ],
            "timeout_seconds": 10.0,
            "env": {},
            "metadata": {
                "runtime_identity": "generic-command",
                "runtime_version": platform.python_version(),
            },
            "correctness_contract": {
                "validators": [
                    {"type": "exit_code", "expected": 0},
                    {"type": "required_metric", "metric": "throughput"},
                    {"type": "metric_min", "metric": "throughput", "min": 100.0},
                    {"type": "metric_max", "metric": "latency_ms", "max": 10.0},
                    {"type": "request_completed"},
                    {"type": "stream_parsed"},
                    {"type": "cleanup_passed"},
                ]
            },
        },
    }


def _write_passport(temp_path: Path) -> Path:
    path = temp_path / "performance_passport.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "verification_status": "passed",
                "claim_eligibility_state": "eligible_preliminary",
                "result_origin": "measured",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _trial(role: str, phase: str, pair_id: str, value: float, *, failed: bool = False, trial_status: str = "valid") -> dict:
    return {
        "role": role,
        "phase": phase,
        "pair_id": pair_id,
        "failed": failed,
        "trial_status": trial_status,
        "metrics": {"decode_tokens_per_second": value},
    }


def _codes(result) -> set[str]:
    return {item["code"] for item in result["guard_results"]}


if __name__ == "__main__":
    unittest.main()
