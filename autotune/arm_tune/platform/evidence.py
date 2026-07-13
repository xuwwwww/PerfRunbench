from __future__ import annotations

from autotune.arm_tune.schemas import WorkloadExecutionEvidence


def no_workload_evidence() -> WorkloadExecutionEvidence:
    return WorkloadExecutionEvidence(status="not_collected")
