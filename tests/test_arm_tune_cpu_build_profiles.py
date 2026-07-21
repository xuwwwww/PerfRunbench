from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.arm_tune.build_profiles.builder import run_build_profile
from autotune.arm_tune.build_profiles.comparison import run_profile_comparison
from autotune.arm_tune.build_profiles.compiler import duplicate_effective_profiles, effective_signature, probe_compiler_target
from autotune.arm_tune.build_profiles.portability import evaluate_native_binary_reuse
from autotune.arm_tune.build_profiles.schemas import NativeBinaryScope
from autotune.arm_tune.errors import ArmTuneConfigError
from autotune.arm_tune.passport.generator import generate_passport
from autotune.arm_tune.passport.verification import verify_passport
from autotune.arm_tune.targets.protocol import atomic_write_json


class ArmTuneCpuBuildProfileTest(unittest.TestCase):
    def test_build_profile_rejects_unknown_cmake_definition(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_config(root)
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["profiles"][0]["requested_definitions"] = {"CMAKE_C_FLAGS": "-Ofast"}
            atomic_write_json(config, payload)

            with self.assertRaises(ArmTuneConfigError):
                run_build_profile(config_path=config, run_id="run", output_dir=root / "out")

    def test_build_profile_rejects_shell_fragment_flag(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_config(root)
            payload = json.loads(config.read_text(encoding="utf-8"))
            payload["profiles"][0]["effective_compiler_flags"] = ["-mcpu=neoverse-n1; rm -rf /"]
            atomic_write_json(config, payload)

            with self.assertRaises(ArmTuneConfigError):
                run_build_profile(config_path=config, run_id="run", output_dir=root / "out")

    def test_kleidiai_requires_runtime_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_config(root, kleidiai_build=True, kleidiai_runtime=False)

            result = run_build_profile(config_path=config, run_id="run", output_dir=root / "out")

            profile = next(item for item in result.profiles if item.profile_id == "native-kleidiai")
            self.assertEqual(profile.kleidiai_status, "requested_but_runtime_unverified")
            self.assertEqual(profile.eligibility_state, "built_runtime_unverified")

    def test_kleidiai_enabled_only_with_build_and_runtime_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = _build_config(root, kleidiai_build=True, kleidiai_runtime=True)

            result = run_build_profile(config_path=config, run_id="run", output_dir=root / "out")

            profile = next(item for item in result.profiles if item.profile_id == "native-kleidiai")
            self.assertEqual(profile.kleidiai_status, "enabled_and_evidenced")
            self.assertEqual(profile.eligibility_state, "built_and_verified")

    def test_native_binary_portability_rejects_feature_mismatch_and_x86(self) -> None:
        scope = NativeBinaryScope(
            architecture="aarch64",
            portability_state="cpu_model_bound",
            required_cpu_features=["asimd", "dotprod"],
            cpu_model="Neoverse-N1",
            binary_sha256="a" * 64,
            startup_smoke_required=True,
        )

        feature_mismatch = evaluate_native_binary_reuse(
            scope,
            current_architecture="aarch64",
            current_features={"asimd"},
            current_cpu_model="Neoverse-N1",
            observed_binary_sha256="a" * 64,
            startup_smoke_status="passed",
        )
        x86 = evaluate_native_binary_reuse(
            scope,
            current_architecture="x86_64",
            current_features={"asimd", "dotprod"},
            current_cpu_model="Neoverse-N1",
            observed_binary_sha256="a" * 64,
            startup_smoke_status="passed",
        )

        self.assertFalse(feature_mismatch.launch_allowed)
        self.assertEqual(feature_mismatch.compatibility_status, "hard_drift")
        self.assertFalse(x86.launch_allowed)
        self.assertIn("architecture mismatch", x86.reasons[0])

    def test_native_binary_portability_accepts_exact_matching_t4g_scope(self) -> None:
        scope = NativeBinaryScope(
            architecture="aarch64",
            portability_state="cpu_model_bound",
            required_cpu_features=["asimd"],
            cpu_model="Neoverse-N1",
            binary_sha256="b" * 64,
            startup_smoke_required=True,
        )

        decision = evaluate_native_binary_reuse(
            scope,
            current_architecture="aarch64",
            current_features={"asimd", "fp16"},
            current_cpu_model="Neoverse-N1",
            observed_binary_sha256="b" * 64,
            startup_smoke_status="passed",
        )

        self.assertTrue(decision.launch_allowed)
        self.assertEqual(decision.compatibility_status, "compatible")

    def test_duplicate_effective_profile_detection(self) -> None:
        duplicates = duplicate_effective_profiles(
            [
                effective_signature(
                    profile_id="native-current-cpu",
                    cmake_definitions={"GGML_NATIVE": "ON"},
                    compile_flags=["-mcpu=neoverse-n1"],
                    cpu_target_policy="native_current_cpu",
                ),
                effective_signature(
                    profile_id="explicit-current-cpu",
                    cmake_definitions={"GGML_NATIVE": "ON"},
                    compile_flags=["-mcpu=neoverse-n1"],
                    cpu_target_policy="native_current_cpu",
                ),
            ]
        )

        self.assertEqual(duplicates, {"explicit-current-cpu": "native-current-cpu"})

    def test_compiler_probe_rejects_shell_fragment_arguments(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with self.assertRaises(ArmTuneConfigError):
                probe_compiler_target(
                    compiler=sys.executable,
                    arguments=["-c", "print('x') && echo bad"],
                    work_dir=temp_dir,
                )

    def test_decode_throughput_comparison_keeps_prefill_separate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            build_result = run_build_profile(config_path=_build_config(root), run_id="build", output_dir=root / "build")
            self.assertEqual(build_result.artifact_validation.validation_status, "passed")
            config = root / "comparison.json"
            atomic_write_json(
                config,
                {
                    "schema_version": "1.0",
                    "result_origin": "test_fixture",
                    "baseline_profile_id": "upstream-default",
                    "noise_band_percent": 5.0,
                    "primary_metric": "decode_tokens_per_second",
                    "secondary_metric": "prefill_tokens_per_second",
                    "metric_goal": "maximize",
                    "profiles": [
                        {
                            "profile_id": "upstream-default",
                            "build_result_path": str(root / "build" / "build_profile_result.json"),
                            "measurements": [
                                {"decode_tokens_per_second": 10.0, "prefill_tokens_per_second": 40.0},
                                {"decode_tokens_per_second": 10.2, "prefill_tokens_per_second": 41.0},
                            ],
                        },
                        {
                            "profile_id": "native-current-cpu",
                            "build_result_path": str(root / "build" / "build_profile_result.json"),
                            "measurements": [
                                {"decode_tokens_per_second": 12.0, "prefill_tokens_per_second": 35.0},
                                {"decode_tokens_per_second": 12.2, "prefill_tokens_per_second": 36.0},
                            ],
                        },
                    ],
                },
            )

            result = run_profile_comparison(config_path=config, run_id="compare", output_dir=root / "compare")

            self.assertEqual(result.primary_metric, "decode_tokens_per_second")
            self.assertEqual(result.secondary_metric, "prefill_tokens_per_second")
            self.assertEqual(result.recommendation.status, "recommended")
            self.assertEqual(result.recommendation.profile_id, "native-current-cpu")
            self.assertIn("decode_tokens_per_second", result.profiles[0].metrics)
            self.assertIn("prefill_tokens_per_second", result.profiles[0].metrics)

    def test_passport_optional_build_evidence_round_trip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comparison = root / "profile_comparison_result.json"
            atomic_write_json(
                comparison,
                {
                    "schema_version": "1.0",
                    "run_id": "compare",
                    "operation": "profile_comparison.arm_build_runtime",
                    "result_origin": "test_fixture",
                    "baseline_profile_id": "upstream-default",
                    "primary_metric": "decode_tokens_per_second",
                    "secondary_metric": "prefill_tokens_per_second",
                    "recommendation": {
                        "schema_version": "1.0",
                        "status": "recommended",
                        "profile_id": "native-current-cpu",
                        "message": "fixture",
                    },
                    "profiles": [
                        {
                            "schema_version": "1.0",
                            "profile_id": "native-current-cpu",
                            "build_status": "completed",
                            "kleidiai_status": "unavailable",
                            "eligibility_status": "eligible",
                            "metrics": {"decode_tokens_per_second": {"schema_version": "1.0"}},
                        }
                    ],
                    "rejected_profiles": [],
                },
            )
            atomic_write_json(root / "artifact_manifest.json", {"schema_version": "1.0", "artifacts": {}})

            passport, json_path, _ = generate_passport(
                comparison_result_path=comparison,
                run_id="passport",
                output_dir=root / "passport",
            )

            self.assertEqual(passport.build_evidence["profile_count"], 1)
            self.assertEqual(verify_passport(json_path).verification_status, "passed")


def _build_config(root: Path, *, kleidiai_build: bool = False, kleidiai_runtime: bool = False) -> Path:
    upstream = _binary(root, "upstream-default")
    native = _binary(root, "native-current-cpu")
    kleidiai = _binary(root, "native-kleidiai")
    config = root / "build_profiles.json"
    atomic_write_json(
        config,
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
                "linker_version": "fixture",
                "generator": "fixture",
                "build_type": "Release",
            },
            "profiles": [
                {
                    "profile_id": "upstream-default",
                    "effective_cmake_cache": {"GGML_NATIVE": "OFF"},
                },
                {
                    "profile_id": "native-current-cpu",
                    "effective_cmake_cache": {"GGML_NATIVE": "ON"},
                    "effective_compiler_flags": ["-mcpu=neoverse-n1"],
                    "native_binary_scope": {
                        "schema_version": "1.0",
                        "architecture": "aarch64",
                        "portability_state": "cpu_model_bound",
                        "required_cpu_features": ["asimd"],
                        "cpu_model": "Neoverse-N1",
                    },
                },
                {
                    "profile_id": "native-kleidiai",
                    "effective_cmake_cache": {"GGML_NATIVE": "ON", "GGML_CPU_KLEIDIAI": "ON"},
                    "compile_command_evidence": [{"command": "cc -Iggml/src/kleidiai"}] if kleidiai_build else [],
                    "linked_libraries": ["libkleidiai.a"] if kleidiai_build else [],
                    "runtime_backend_evidence": (
                        {"runtime_backend_status": "enabled_and_evidenced", "startup_line": "CPU_KLEIDIAI backend active"}
                        if kleidiai_runtime
                        else {}
                    ),
                },
            ],
            "binaries": {
                "upstream-default": {"path": str(upstream)},
                "native-current-cpu": {"path": str(native)},
                "native-kleidiai": {"path": str(kleidiai)},
            },
            "build": {"effective_cmake_cache": {"CMAKE_BUILD_TYPE": "Release"}},
        },
    )
    return config


def _binary(root: Path, label: str) -> Path:
    path = root / label
    path.write_bytes(f"{label} fixture binary for {os.name}\n".encode("utf-8"))
    return path


if __name__ == "__main__":
    unittest.main()
