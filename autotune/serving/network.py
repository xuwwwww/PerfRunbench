from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune.schemas import to_plain
from autotune.serving import SERVING_SCHEMA_VERSION


@dataclass(frozen=True)
class NetworkCapabilityReport:
    interface: str = "planned"
    driver: str = "unknown"
    rx_queue_count: int | None = None
    tx_queue_count: int | None = None
    irq_affinity_writable: bool | None = None
    rps_available: bool | None = None
    xps_available: bool | None = None
    irqbalance_state: str = "unknown"
    socket_counter_sources: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NetworkQueueProfile:
    profile_id: str
    rps_state: str = "not_available"
    xps_state: str = "not_available"
    listen_backlog: int | None = None
    evidence_required: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NetworkIRQProfile:
    profile_id: str
    irq_affinity_policy: str
    reversible: bool
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NetworkExperimentPlan:
    topology: str
    capability_report: NetworkCapabilityReport
    queue_profiles: list[NetworkQueueProfile]
    irq_profiles: list[NetworkIRQProfile]
    loopback_results_excluded_from_remote_claims: bool = True
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NetworkRollbackReport:
    rollback_strategy: str
    exact_reversible: bool
    required_before_snapshot: bool = True
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


def create_phase21d2_network_plan() -> NetworkExperimentPlan:
    capability = NetworkCapabilityReport()
    return NetworkExperimentPlan(
        topology="c7g_server_t4g_remote_load_generator_planned",
        capability_report=capability,
        queue_profiles=[
            NetworkQueueProfile(
                profile_id="original-network-state",
                evidence_required=["interface_driver", "queue_counts", "drops", "retransmits", "listen_overflow"],
            ),
            NetworkQueueProfile(
                profile_id="evidence-based-listen-backlog",
                listen_backlog=None,
                evidence_required=["listen_overflow_or_backlog_saturation"],
            ),
        ],
        irq_profiles=[
            NetworkIRQProfile(profile_id="managed-irq-original", irq_affinity_policy="preserve_original", reversible=True),
            NetworkIRQProfile(profile_id="managed-irq-housekeeping", irq_affinity_policy="only_when_irq_writable_and_mapped", reversible=True),
        ],
    )
