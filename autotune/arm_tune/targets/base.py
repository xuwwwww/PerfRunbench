from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from autotune.arm_tune.schemas import CommandResult, HostInspectionResult, ProductionValidation, TargetSpec


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
