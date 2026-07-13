from __future__ import annotations

import unittest

from autotune.arm_tune.benchmark.provenance import decide_benchmark_provenance
from autotune.arm_tune.schemas import (
    ArchitectureInfo,
    ArmCpuIdentity,
    ArmHostCapabilities,
    BenchmarkArtifactsSpec,
    BenchmarkExecutableSpec,
    BenchmarkFileManifest,
    BenchmarkModelSpec,
    CpuTopology,
    HostInfo,
    HostInspectionResult,
    LlamaBenchInspection,
    LlamaBenchJobSpec,
    LlamaBenchSettings,
    ProductionValidation,
)


class ArmTuneBenchmarkOriginTest(unittest.TestCase):
    def test_fake_binary_on_x86_is_test_fixture(self) -> None:
        provenance = decide_benchmark_provenance(
            _job(result_origin="measured", executable_path="/tmp/fake_llama_bench"),
            _host(system="Windows", machine="AMD64", valid=False),
        )

        self.assertEqual(provenance.result_origin, "test_fixture")
        self.assertTrue(provenance.fake_executable_marker)

    def test_fake_binary_on_arm_is_still_test_fixture(self) -> None:
        provenance = decide_benchmark_provenance(
            _job(result_origin="measured", executable_path="/opt/llama-bench"),
            _host(system="Linux", machine="aarch64", valid=True),
            executable=BenchmarkFileManifest(path="/opt/llama-bench", basename="llama-bench", size_bytes=1, sha256="f" * 64),
            inspection=_inspection(fixture_marker_detected=True),
        )

        self.assertEqual(provenance.result_origin, "test_fixture")
        self.assertTrue(provenance.host_production_ready)

    def test_test_only_host_override_is_test_fixture(self) -> None:
        provenance = decide_benchmark_provenance(
            _job(result_origin="measured"),
            _host(system="Linux", machine="aarch64", valid=True, fixture_identity=True),
        )

        self.assertEqual(provenance.result_origin, "test_fixture")
        self.assertTrue(provenance.test_only_host_override)

    def test_production_origin_cannot_be_forced_without_arm64_target(self) -> None:
        provenance = decide_benchmark_provenance(
            _job(result_origin="measured", executable_path="/opt/llama-bench"),
            _host(system="Windows", machine="AMD64", valid=False),
        )

        self.assertEqual(provenance.result_origin, "unverified")

    def test_missing_provenance_does_not_silently_become_measured(self) -> None:
        provenance = decide_benchmark_provenance(
            _job(result_origin="unverified", executable_path="/opt/llama-bench"),
            _host(system="Linux", machine="aarch64", valid=True),
        )

        self.assertEqual(provenance.result_origin, "unverified")

    def test_real_arm_request_without_fixture_markers_can_be_measured(self) -> None:
        provenance = decide_benchmark_provenance(
            _job(result_origin="measured", executable_path="/opt/llama-bench"),
            _host(system="Linux", machine="aarch64", valid=True),
            executable=BenchmarkFileManifest(path="/opt/llama-bench", basename="llama-bench", size_bytes=1, sha256="a" * 64),
            inspection=_inspection(fixture_marker_detected=False),
        )

        self.assertEqual(provenance.result_origin, "measured")


def _job(*, result_origin: str, executable_path: str = "/opt/llama-bench") -> LlamaBenchJobSpec:
    return LlamaBenchJobSpec(
        protocol_version="1.0",
        operation="benchmark.llama_bench",
        run_id="run",
        target_run_directory="/tmp/run",
        executable=BenchmarkExecutableSpec(path=executable_path),
        model=BenchmarkModelSpec(path="/models/model.gguf"),
        benchmark=LlamaBenchSettings(threads=1, prompt_tokens=16, generation_tokens=8, repetitions=1, timeout_seconds=5),
        artifacts=BenchmarkArtifactsSpec(),
        result_origin=result_origin,
    )


def _host(*, system: str, machine: str, valid: bool, fixture_identity: bool = False) -> HostInspectionResult:
    normalized = "aarch64" if machine.lower() in {"aarch64", "arm64"} else "amd64"
    host = HostInfo(
        architecture=ArchitectureInfo(system=system, machine=machine, normalized_machine=normalized, is_linux=system == "Linux", is_arm64=normalized == "aarch64"),
        cpu_identity=ArmCpuIdentity(source="fixture" if fixture_identity else "cpuinfo"),
        host_capabilities=ArmHostCapabilities(evidence_source="fixture" if fixture_identity else "cpuinfo"),
        topology=CpuTopology(source="fixture" if fixture_identity else "lscpu"),
        os_release="fixture-linux" if fixture_identity else "unknown",
    )
    return HostInspectionResult(
        target_type="local",
        host=host,
        production_validation=ProductionValidation(valid=valid, reason="fixture validation", observed_system=system, observed_machine=normalized),
    )


def _inspection(*, fixture_marker_detected: bool) -> LlamaBenchInspection:
    return LlamaBenchInspection(
        version_command=["llama-bench", "--version"],
        help_command=["llama-bench", "--help"],
        version_exit_code=0,
        help_exit_code=0,
        version_status="passed",
        help_status="passed",
        raw_version_stdout_path="raw/version.txt",
        raw_version_stderr_path="raw/version.err",
        raw_help_stdout_path="raw/help.txt",
        raw_help_stderr_path="raw/help.err",
        fixture_marker_detected=fixture_marker_detected,
    )


if __name__ == "__main__":
    unittest.main()
