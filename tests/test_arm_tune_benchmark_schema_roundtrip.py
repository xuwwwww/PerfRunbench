from __future__ import annotations

import json
import unittest

from autotune.arm_tune.schemas import (
    ArchitectureInfo,
    ArmBenchmarkResult,
    ArmCpuIdentity,
    ArmHostCapabilities,
    ArtifactValidationResult,
    BenchmarkArtifactManifest,
    BenchmarkArtifactsSpec,
    BenchmarkExecutableSpec,
    BenchmarkFileManifest,
    BenchmarkModelSpec,
    BenchmarkProvenance,
    BenchmarkTrialResult,
    CpuTopology,
    HostInfo,
    LlamaBenchInspection,
    LlamaBenchInvocation,
    LlamaBenchJobSpec,
    LlamaBenchParseResult,
    LlamaBenchSettings,
    ParsedMetric,
)


class ArmTuneBenchmarkSchemaRoundTripTest(unittest.TestCase):
    def test_benchmark_job_round_trip(self) -> None:
        job = LlamaBenchJobSpec(
            protocol_version="1.0",
            operation="benchmark.llama_bench",
            run_id="run",
            target_run_directory="/tmp/run",
            executable=BenchmarkExecutableSpec(path="/bin/llama-bench", expected_sha256="a" * 64),
            model=BenchmarkModelSpec(path="/models/model.gguf", expected_sha256="b" * 64),
            benchmark=LlamaBenchSettings(threads=1, prompt_tokens=16, generation_tokens=8, repetitions=1, timeout_seconds=5),
            artifacts=BenchmarkArtifactsSpec(),
            result_origin="measured",
        )

        self.assertEqual(_normalized(job.to_dict()), _normalized(LlamaBenchJobSpec.from_dict(_json(job.to_dict())).to_dict()))

    def test_nested_benchmark_structures_round_trip(self) -> None:
        executable = BenchmarkFileManifest(path="/bin/llama-bench", basename="llama-bench", size_bytes=10, sha256="a" * 64)
        model = BenchmarkFileManifest(path="/models/model.gguf", basename="model.gguf", size_bytes=20, sha256="b" * 64)
        inspection = _inspection()
        invocation = _invocation()
        parse_result = _parse_result()
        artifact_manifest = BenchmarkArtifactManifest(artifacts={"stdout": {"label": "stdout", "relative_path": "raw/stdout.txt", "sha256": "c" * 64}})
        trial = BenchmarkTrialResult(
            trial_index=1,
            validation_status="passed",
            execution_status="completed",
            parse_status="parsed",
            benchmark_status="passed",
            message="benchmark completed",
            invocation=invocation,
            parse_result=parse_result,
        )

        self.assertEqual(_normalized(executable.to_dict()), _normalized(BenchmarkFileManifest(**_strip(_json(executable.to_dict()))).to_dict()))
        self.assertEqual(_normalized(model.to_dict()), _normalized(BenchmarkFileManifest(**_strip(_json(model.to_dict()))).to_dict()))
        self.assertEqual(_normalized(inspection.to_dict()), _normalized(LlamaBenchInspection(**_strip(_json(inspection.to_dict()))).to_dict()))
        self.assertEqual(_normalized(invocation.to_dict()), _normalized(LlamaBenchInvocation(**_strip(_json(invocation.to_dict()))).to_dict()))
        restored_parse = LlamaBenchParseResult(
            parse_status=parse_result.parse_status,
            metrics=[ParsedMetric(**_strip(metric)) for metric in _json(parse_result.to_dict())["metrics"]],
            parser_strategy=parse_result.parser_strategy,
            unparsed_lines=parse_result.unparsed_lines,
            message=parse_result.message,
        )
        self.assertEqual(_normalized(parse_result.to_dict()), _normalized(restored_parse.to_dict()))
        self.assertEqual(_normalized(artifact_manifest.to_dict()), _normalized(BenchmarkArtifactManifest(artifacts=_json(artifact_manifest.to_dict())["artifacts"]).to_dict()))
        restored_trial = BenchmarkTrialResult(
            trial_index=trial.trial_index,
            validation_status=trial.validation_status,
            execution_status=trial.execution_status,
            parse_status=trial.parse_status,
            benchmark_status=trial.benchmark_status,
            message=trial.message,
            invocation=LlamaBenchInvocation(**_strip(_json(invocation.to_dict()))),
            parse_result=restored_parse,
        )
        self.assertEqual(_normalized(trial.to_dict()), _normalized(restored_trial.to_dict()))

    def test_final_benchmark_result_round_trip_ignores_unknown_additive_top_level_field(self) -> None:
        result = _benchmark_result()
        payload = _json(result.to_dict())
        payload["future_additive_field"] = {"ignored": True}

        restored = ArmBenchmarkResult.from_dict(payload)

        self.assertEqual(_normalized(result.to_dict()), _normalized(restored.to_dict()))

    def test_rejects_unsupported_future_schema_version_and_missing_schema_version(self) -> None:
        payload = _json(_benchmark_result().to_dict())
        payload["schema_version"] = "2.0"
        with self.assertRaises(ValueError):
            ArmBenchmarkResult.from_dict(payload)
        payload.pop("schema_version")
        with self.assertRaises(ValueError):
            ArmBenchmarkResult.from_dict(payload)

    def test_rejects_missing_critical_field_invalid_enum_and_invalid_status_combination(self) -> None:
        payload = _json(_benchmark_result().to_dict())
        payload.pop("run_id")
        with self.assertRaises(ValueError):
            ArmBenchmarkResult.from_dict(payload)
        payload = _json(_benchmark_result().to_dict())
        payload["parse_status"] = "made_up"
        with self.assertRaises(ValueError):
            ArmBenchmarkResult.from_dict(payload)
        payload = _json(_benchmark_result().to_dict())
        payload["execution_status"] = "failed"
        with self.assertRaises(ValueError):
            ArmBenchmarkResult.from_dict(payload)

    def test_rejects_wrong_numeric_type_and_invalid_path_type_in_job(self) -> None:
        job_payload = _json(
            LlamaBenchJobSpec(
                protocol_version="1.0",
                operation="benchmark.llama_bench",
                run_id="run",
                target_run_directory="/tmp/run",
                executable=BenchmarkExecutableSpec(path="/bin/llama-bench"),
                model=BenchmarkModelSpec(path="/models/model.gguf"),
                benchmark=LlamaBenchSettings(threads=1, prompt_tokens=16, generation_tokens=8, repetitions=1, timeout_seconds=5),
            ).to_dict()
        )
        job_payload["benchmark"]["threads"] = "1"
        with self.assertRaises(ValueError):
            LlamaBenchJobSpec.from_dict(job_payload)
        job_payload["benchmark"]["threads"] = 1
        job_payload["executable"]["path"] = ["not", "a", "path"]
        with self.assertRaises(ValueError):
            LlamaBenchJobSpec.from_dict(job_payload)


def _benchmark_result() -> ArmBenchmarkResult:
    host = HostInfo(
        architecture=ArchitectureInfo(system="Linux", machine="aarch64", normalized_machine="aarch64", kernel_release="6.8", is_linux=True, is_arm64=True),
        cpu_identity=ArmCpuIdentity(model_name="Neoverse", source="cpuinfo"),
        host_capabilities=ArmHostCapabilities(features=["asimd"], neon=True, evidence_source="cpuinfo"),
        topology=CpuTopology(logical_cpus=2, source="lscpu"),
    )
    return ArmBenchmarkResult(
        run_id="run",
        operation="benchmark.llama_bench",
        target_type="local",
        result_origin="measured",
        validation_status="passed",
        execution_status="completed",
        parse_status="parsed",
        benchmark_status="passed",
        host=host,
        executable=BenchmarkFileManifest(path="/bin/llama-bench", basename="llama-bench", size_bytes=10, sha256="a" * 64),
        model=BenchmarkFileManifest(path="/models/model.gguf", basename="model.gguf", size_bytes=20, sha256="b" * 64),
        inspection=_inspection(),
        trial=BenchmarkTrialResult(
            trial_index=1,
            validation_status="passed",
            execution_status="completed",
            parse_status="parsed",
            benchmark_status="passed",
            message="benchmark completed",
            invocation=_invocation(),
            parse_result=_parse_result(),
        ),
        artifacts=BenchmarkArtifactManifest(artifacts={"stdout": {"label": "stdout", "relative_path": "raw/stdout.txt", "sha256": "c" * 64}}),
        provenance=BenchmarkProvenance(result_origin="measured", requested_origin="measured", decision_reason="real Arm", host_production_ready=True),
        artifact_validation=ArtifactValidationResult(validation_status="passed", checked_artifacts=1),
        artifact_directory="/tmp/run",
    )


def _inspection() -> LlamaBenchInspection:
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
        supported_flags=["-m", "-t", "-p", "-n", "-r"],
        fixture_marker_detected=False,
    )


def _invocation() -> LlamaBenchInvocation:
    return LlamaBenchInvocation(
        command=["llama-bench", "-m", "model.gguf"],
        started_at_utc="2026-01-01T00:00:00Z",
        ended_at_utc="2026-01-01T00:00:01Z",
        duration_seconds=1.0,
        exit_code=0,
        timed_out=False,
        raw_stdout_path="raw/stdout.txt",
        raw_stderr_path="raw/stderr.txt",
    )


def _parse_result() -> LlamaBenchParseResult:
    return LlamaBenchParseResult(
        parse_status="parsed",
        parser_strategy="pipe_table",
        metrics=[
            ParsedMetric(
                name="prefill_tokens_per_second",
                value=100.0,
                unit="tokens/s",
                source="stdout",
                parser_strategy="pipe_table",
                raw_source_line="| pp128 | 100.0 |",
                raw_line_reference=1,
                validation_status="passed",
            ),
            ParsedMetric(
                name="decode_tokens_per_second",
                value=50.0,
                unit="tokens/s",
                source="stdout",
                parser_strategy="pipe_table",
                raw_source_line="| tg64 | 50.0 |",
                raw_line_reference=2,
                validation_status="passed",
            ),
        ],
    )


def _json(payload: dict) -> dict:
    return json.loads(json.dumps(payload, sort_keys=True))


def _normalized(payload: dict) -> dict:
    return _json(payload)


def _strip(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "schema_version"}


if __name__ == "__main__":
    unittest.main()
