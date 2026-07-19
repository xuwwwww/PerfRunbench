from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pathlib import Path

from autotune.arm_tune.schemas import (
    ArmBenchmarkResult,
    CommandResult,
    HostInspectionResult,
    LlamaBenchJobSpec,
    LlamaServerBenchmarkResult,
    LlamaServerJobSpec,
    ProductionValidation,
    TargetSpec,
)


class CommandRunner(Protocol):
    def run(self, command: list[str], *, timeout_seconds: int) -> CommandResult:
        ...


class ArmTuneTarget(ABC):
    def __init__(self, spec: TargetSpec) -> None:
        self.spec = spec

    @abstractmethod
    def inspect(self) -> HostInspectionResult:
        raise NotImplementedError

    @abstractmethod
    def validate_production_target(self) -> ProductionValidation:
        raise NotImplementedError

    def run_benchmark(self, job: LlamaBenchJobSpec, local_artifact_dir: str | Path) -> ArmBenchmarkResult:
        raise NotImplementedError

    def run_server_benchmark(self, job: LlamaServerJobSpec, local_artifact_dir: str | Path) -> LlamaServerBenchmarkResult:
        raise NotImplementedError

    def prepare_server_prompt_suite(self, local_prompt_suite_path: str | Path, *, run_id: str) -> str:
        return str(Path(local_prompt_suite_path).resolve())
