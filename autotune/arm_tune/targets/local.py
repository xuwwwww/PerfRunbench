from __future__ import annotations

from autotune.arm_tune.platform.environment import collect_host_inspection
from autotune.arm_tune.schemas import HostInspectionResult, ProductionValidation, TargetSpec
from autotune.arm_tune.targets.base import ArmTuneTarget


class LocalArmTarget(ArmTuneTarget):
    def __init__(self, spec: TargetSpec | None = None) -> None:
        super().__init__(spec or TargetSpec(type="local"))

    def inspect(self) -> HostInspectionResult:
        return collect_host_inspection(target_type="local")

    def validate_production_target(self) -> ProductionValidation:
        return self.inspect().production_validation
