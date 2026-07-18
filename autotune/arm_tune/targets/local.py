from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from autotune.arm_tune.benchmark.artifact_validation import apply_artifact_validation, validate_artifact_manifest
from autotune.arm_tune.benchmark.llama_bench import ensure_benchmark_operation
from autotune.arm_tune.benchmark.llama_server import apply_server_artifact_validation, ensure_server_benchmark_operation, finalize_server_result_artifacts
from autotune.arm_tune.platform.environment import collect_host_inspection
from autotune.arm_tune.schemas import ArmBenchmarkResult, HostInspectionResult, LlamaBenchJobSpec, LlamaServerBenchmarkResult, LlamaServerJobSpec, ProductionValidation, TargetSpec
from autotune.arm_tune.targets.base import ArmTuneTarget
from autotune.arm_tune.targets.protocol import atomic_write_json
from autotune.arm_tune.targets.worker import run_worker


class LocalArmTarget(ArmTuneTarget):
    def __init__(self, spec: TargetSpec | None = None) -> None:
        super().__init__(spec or TargetSpec(type="local"))

    def inspect(self) -> HostInspectionResult:
        return collect_host_inspection(target_type="local")

    def validate_production_target(self) -> ProductionValidation:
        return self.inspect().production_validation

    def run_benchmark(self, job: LlamaBenchJobSpec, local_artifact_dir: str | Path) -> ArmBenchmarkResult:
        benchmark_dir = Path(local_artifact_dir)
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        local_job = replace(job, target_type="local", target_run_directory=str(benchmark_dir))
        ensure_benchmark_operation(local_job)
        job_path = benchmark_dir / "benchmark_job.json"
        atomic_write_json(job_path, local_job.to_dict())
        worker_result = run_worker(job_path, benchmark_dir)
        result = ArmBenchmarkResult.from_dict(worker_result.result)
        validation = validate_artifact_manifest(benchmark_dir)
        validated = apply_artifact_validation(result, validation, benchmark_dir)
        atomic_write_json(benchmark_dir / "benchmark_result.json", validated.to_dict())
        return validated

    def run_server_benchmark(self, job: LlamaServerJobSpec, local_artifact_dir: str | Path) -> LlamaServerBenchmarkResult:
        server_dir = Path(local_artifact_dir)
        server_dir.mkdir(parents=True, exist_ok=True)
        local_job = replace(job, target_type="local", target_run_directory=str(server_dir))
        ensure_server_benchmark_operation(local_job)
        job_path = server_dir / "server_job.json"
        atomic_write_json(job_path, local_job.to_dict())
        worker_result = run_worker(job_path, server_dir)
        result = LlamaServerBenchmarkResult.from_dict(worker_result.result)
        validation = validate_artifact_manifest(server_dir)
        validated = apply_server_artifact_validation(result, validation, server_dir)
        return finalize_server_result_artifacts(server_dir, validated)
