from __future__ import annotations

from autotune.arm_tune.schemas import BenchmarkFileManifest, BenchmarkProvenance, HostInspectionResult, LlamaBenchInspection, LlamaBenchJobSpec


FIXTURE_MARKERS = ("fake", "fixture", "synthetic")


def decide_benchmark_provenance(
    job: LlamaBenchJobSpec,
    host_inspection: HostInspectionResult,
    *,
    executable: BenchmarkFileManifest | None = None,
    inspection: LlamaBenchInspection | None = None,
) -> BenchmarkProvenance:
    requested = job.result_origin
    evidence: list[str] = [f"requested_origin={requested}"]
    host_ready = host_inspection.production_validation.valid
    fake_marker = _has_fake_executable_marker(job, executable, inspection)
    test_override = _has_test_only_host_override(host_inspection)
    if host_ready:
        evidence.append("host_production_ready=true")
    else:
        evidence.append("host_production_ready=false")
    if fake_marker:
        evidence.append("fake_or_fixture_executable_marker=true")
    if test_override:
        evidence.append("test_only_host_override=true")
    if requested == "test_fixture" or fake_marker or test_override:
        return BenchmarkProvenance(
            result_origin="test_fixture",
            requested_origin=requested,
            decision_reason="fixture provenance marker overrides target architecture",
            evidence=evidence,
            host_production_ready=host_ready,
            fake_executable_marker=fake_marker,
            test_only_host_override=test_override,
        )
    if requested == "measured" and host_ready:
        return BenchmarkProvenance(
            result_origin="measured",
            requested_origin=requested,
            decision_reason="measured origin requires Linux Arm64 target and no fixture markers",
            evidence=evidence,
            host_production_ready=host_ready,
            fake_executable_marker=False,
            test_only_host_override=False,
        )
    return BenchmarkProvenance(
        result_origin="unverified",
        requested_origin=requested,
        decision_reason="missing measured provenance or unsupported production target",
        evidence=evidence,
        host_production_ready=host_ready,
        fake_executable_marker=fake_marker,
        test_only_host_override=test_override,
    )


def _has_fake_executable_marker(
    job: LlamaBenchJobSpec,
    executable: BenchmarkFileManifest | None,
    inspection: LlamaBenchInspection | None,
) -> bool:
    texts = [job.executable.path]
    if executable is not None:
        texts.extend([executable.basename, executable.path])
    if inspection is not None:
        texts.append(str(inspection.fixture_marker_detected))
        if inspection.fixture_marker_detected:
            return True
    return any(marker in text.lower() for text in texts for marker in FIXTURE_MARKERS)


def _has_test_only_host_override(host_inspection: HostInspectionResult) -> bool:
    host = host_inspection.host
    texts = [
        host.architecture.kernel_release,
        host.cpu_identity.source,
        host.host_capabilities.evidence_source,
        host.topology.source,
        host.os_release,
    ]
    return any("fixture" in text.lower() or "test" in text.lower() for text in texts)
