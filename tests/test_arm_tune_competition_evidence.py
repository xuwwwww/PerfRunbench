from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.arm_tune.build_profiles.builder import run_build_profile, write_build_plan
from autotune.arm_tune.build_profiles.comparison import run_profile_comparison
from autotune.arm_tune.passport.generator import generate_passport
from autotune.arm_tune.passport.verification import verify_passport
from autotune.arm_tune.targets.protocol import atomic_write_json
from autotune.cli import main


class ArmTuneCompetitionEvidenceTest(unittest.TestCase):
    def test_build_plan_does_not_execute_and_build_profile_records_kleidiai_unverified(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            portable = _binary(root, "portable")
            native = _binary(root, "native")
            optimized = _binary(root, "optimized")
            config_path = _write_build_config(root, portable=portable, native=native, optimized=optimized, optimized_kleidiai_evidence=False)

            plan = write_build_plan(config_path=config_path, run_id="plan-run", output_dir=root / "plan")
            result = run_build_profile(config_path=config_path, run_id="build-run", output_dir=root / "build")

            self.assertFalse(plan.execute_build)
            self.assertEqual([profile.profile_id for profile in plan.profiles], ["portable-baseline", "arm-native", "arm-optimized"])
            self.assertTrue(all(profile.build_status == "completed" for profile in result.profiles))
            self.assertEqual(result.artifact_validation.validation_status, "passed")
            optimized_profile = next(profile for profile in result.profiles if profile.profile_id == "arm-optimized")
            self.assertEqual(optimized_profile.kleidiai_status, "requested_but_unverified")

    def test_build_profile_records_kleidiai_only_when_evidenced(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = _write_build_config(
                root,
                portable=_binary(root, "portable"),
                native=_binary(root, "native"),
                optimized=_binary(root, "optimized"),
                optimized_kleidiai_evidence=True,
            )

            result = run_build_profile(config_path=config_path, run_id="build-run", output_dir=root / "build")

            optimized_profile = next(profile for profile in result.profiles if profile.profile_id == "arm-optimized")
            self.assertEqual(optimized_profile.kleidiai_status, "enabled_and_evidenced")

    def test_profile_comparison_recommends_faster_evidence_valid_profile_and_rejects_bad_build_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_path = _build_result(root, optimized_kleidiai_evidence=False)
            config_path = root / "profile_comparison.json"
            atomic_write_json(
                config_path,
                {
                    "schema_version": "1.0",
                    "result_origin": "test_fixture",
                    "baseline_profile_id": "portable-baseline",
                    "noise_band_percent": 2.0,
                    "profiles": [
                        {
                            "profile_id": "portable-baseline",
                            "build_result_path": str(build_path),
                            "measurements": [{"ttft_ms": 100.0, "end_to_end_latency_ms": 190.0}],
                        },
                        {
                            "profile_id": "arm-native",
                            "build_result_path": str(build_path),
                            "measurements": [{"ttft_ms": 90.0, "end_to_end_latency_ms": 175.0}],
                        },
                        {
                            "profile_id": "arm-optimized",
                            "build_result_path": str(build_path),
                            "measurements": [{"ttft_ms": 80.0, "end_to_end_latency_ms": 170.0}],
                        },
                    ],
                },
            )

            comparison = run_profile_comparison(config_path=config_path, run_id="compare-run", output_dir=root / "comparison")

            self.assertEqual(comparison.result_origin, "test_fixture")
            self.assertEqual(comparison.recommendation.status, "recommended")
            self.assertEqual(comparison.recommendation.profile_id, "arm-native")
            self.assertTrue(comparison.recommendation.message.startswith("Best observed evidence-valid profile"))
            self.assertEqual(comparison.profiles[0].quality_gate_status, "not_evaluated")
            self.assertEqual(comparison.artifact_validation.validation_status, "passed")
            rejected = {item["profile_id"]: item for item in comparison.rejected_profiles}
            self.assertEqual(rejected["arm-optimized"]["eligibility_status"], "rejected_build_evidence")

    def test_profile_comparison_can_opt_in_to_quality_required_rejection(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_path = _build_result(root, optimized_kleidiai_evidence=True)
            failed_quality = _quality_result(root, "unsuitable", "failed")
            config_path = root / "profile_comparison.json"
            atomic_write_json(
                config_path,
                {
                    "schema_version": "1.0",
                    "result_origin": "test_fixture",
                    "baseline_profile_id": "portable-baseline",
                    "noise_band_percent": 2.0,
                    "profiles": [
                        {
                            "profile_id": "portable-baseline",
                            "build_result_path": str(build_path),
                            "quality_calibration_result_path": str(failed_quality),
                            "quality_required": True,
                            "measurements": [{"ttft_ms": 100.0, "end_to_end_latency_ms": 190.0}],
                        },
                        {
                            "profile_id": "arm-native",
                            "build_result_path": str(build_path),
                            "quality_calibration_result_path": str(failed_quality),
                            "quality_required": True,
                            "measurements": [{"ttft_ms": 90.0, "end_to_end_latency_ms": 175.0}],
                        },
                    ],
                },
            )

            comparison = run_profile_comparison(config_path=config_path, run_id="compare-run", output_dir=root / "comparison")

            self.assertEqual(comparison.recommendation.status, "no_eligible_profile")
            self.assertEqual({item["eligibility_status"] for item in comparison.rejected_profiles}, {"rejected_quality"})

    def test_passport_generation_verification_and_tamper_detection(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison_path = _comparison_result(root)

            passport, json_path, markdown_path = generate_passport(
                comparison_result_path=comparison_path,
                run_id="passport-run",
                output_dir=root / "passport",
            )
            verification = verify_passport(json_path)

            self.assertEqual(passport.claim_eligibility_state, "fixture_only")
            self.assertTrue(markdown_path.exists())
            self.assertEqual(verification.verification_status, "passed")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["comparison_result_sha256"] = "0" * 64
            tampered = root / "passport" / "tampered_passport.json"
            atomic_write_json(tampered, payload)

            self.assertEqual(verify_passport(tampered).verification_status, "failed")

    def test_public_cli_fake_competition_evidence_e2e(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_config = _write_build_config(
                root,
                portable=_binary(root, "portable"),
                native=_binary(root, "native"),
                optimized=_binary(root, "optimized"),
                optimized_kleidiai_evidence=False,
            )

            with _cwd(root):
                with redirect_stdout(io.StringIO()):
                    build_code = main(["arm", "build-profile", "--config", str(build_config)])
            build_result = next((root / ".autotuneai" / "runs").glob("*/armtune/build_profile/build_profile_result.json"))
            comparison_config = root / "profile_comparison.json"
            atomic_write_json(
                comparison_config,
                {
                    "schema_version": "1.0",
                    "result_origin": "test_fixture",
                    "baseline_profile_id": "portable-baseline",
                    "profiles": [
                        {
                            "profile_id": "portable-baseline",
                            "build_result_path": str(build_result),
                            "measurements": [{"ttft_ms": 100.0, "end_to_end_latency_ms": 190.0}],
                        },
                        {
                            "profile_id": "arm-native",
                            "build_result_path": str(build_result),
                            "measurements": [{"ttft_ms": 90.0, "end_to_end_latency_ms": 175.0}],
                        },
                        {
                            "profile_id": "arm-optimized",
                            "build_result_path": str(build_result),
                            "measurements": [{"ttft_ms": 80.0, "end_to_end_latency_ms": 170.0}],
                        },
                    ],
                },
            )
            with _cwd(root):
                with redirect_stdout(io.StringIO()):
                    compare_code = main(["arm", "profile-compare", "--config", str(comparison_config)])
            comparison_result = next((root / ".autotuneai" / "runs").glob("*/armtune/profile_comparison/profile_comparison_result.json"))
            with _cwd(root):
                with redirect_stdout(io.StringIO()):
                    passport_code = main(["arm", "passport", "--comparison-result", str(comparison_result)])
            passport_result = next((root / ".autotuneai" / "runs").glob("*/armtune/passport/performance_passport.json"))
            passport_payload = json.loads(passport_result.read_text(encoding="utf-8"))

            self.assertEqual(build_code, 0)
            self.assertEqual(compare_code, 0)
            self.assertEqual(passport_code, 0)
            self.assertEqual(passport_payload["claim_eligibility_state"], "fixture_only")
            self.assertEqual(verify_passport(passport_result).verification_status, "passed")


def _binary(root: Path, label: str) -> Path:
    path = root / f"{label}-llama-server"
    path.write_bytes(f"{label} fixture binary\n".encode("utf-8"))
    return path


def _write_build_config(root: Path, *, portable: Path, native: Path, optimized: Path, optimized_kleidiai_evidence: bool) -> Path:
    compile_evidence = [{"command": "cc -DKLEIDIAI=1"}] if optimized_kleidiai_evidence else []
    linked = ["libkleidiai.a"] if optimized_kleidiai_evidence else []
    path = root / "build_profiles.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "result_origin": "test_fixture",
            "source_repo": str(root),
            "build_root": str(root / "build"),
            "target_architecture": "aarch64",
            "toolchain": {
                "compiler_id": "fixture-cc",
                "compiler_version": "fixture",
                "cmake_version": "fixture",
                "generator": "fixture",
                "build_type": "Release",
            },
            "profiles": [
                "portable-baseline",
                "arm-native",
                {
                    "profile_id": "arm-optimized",
                    "effective_cmake_cache": {"LLAMA_KLEIDIAI": "ON"},
                    "compile_command_evidence": compile_evidence,
                    "linked_libraries": linked,
                },
            ],
            "binaries": {
                "portable-baseline": {"path": str(portable)},
                "arm-native": {"path": str(native)},
                "arm-optimized": {"path": str(optimized)},
            },
            "build": {"effective_cmake_cache": {"CMAKE_BUILD_TYPE": "Release"}},
        },
    )
    return path


def _build_result(root: Path, *, optimized_kleidiai_evidence: bool) -> Path:
    config = _write_build_config(
        root,
        portable=_binary(root, "portable"),
        native=_binary(root, "native"),
        optimized=_binary(root, "optimized"),
        optimized_kleidiai_evidence=optimized_kleidiai_evidence,
    )
    result = run_build_profile(config_path=config, run_id="build-run", output_dir=root / "build")
    path = root / "build" / "build_profile_result.json"
    self_check = json.loads(path.read_text(encoding="utf-8"))
    assert self_check["run_id"] == result.run_id
    return path


def _quality_result(root: Path, suitability: str, status: str) -> Path:
    path = root / f"quality_{suitability}.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "run_id": f"quality-{suitability}",
            "suitability_state": suitability,
            "quality_gate_status": status,
        },
    )
    return path


def _comparison_result(root: Path) -> Path:
    path = root / "profile_comparison_result.json"
    atomic_write_json(
        path,
        {
            "schema_version": "1.0",
            "run_id": "compare-run",
            "operation": "profile_comparison.arm_build_runtime",
            "result_origin": "test_fixture",
            "baseline_profile_id": "portable-baseline",
            "primary_metric": "ttft_ms",
            "secondary_metric": "end_to_end_latency_ms",
            "recommendation": {
                "schema_version": "1.0",
                "status": "recommended",
                "profile_id": "arm-native",
                "message": "Fixture-only recommendation.",
            },
            "profiles": [],
            "rejected_profiles": [],
        },
    )
    atomic_write_json(
        root / "artifact_manifest.json",
        {
            "schema_version": "1.0",
            "artifacts": {},
        },
    )
    return path


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
