from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune.schemas import to_plain
from autotune.serving import SERVING_SCHEMA_VERSION


@dataclass(frozen=True)
class RemoteHostRole:
    role_id: str
    target_class: str
    expected_architecture: str
    expected_cpu_class: str
    expected_logical_cpus: int | None
    responsibility: str
    connection_details_policy: str = "ignored_private_configuration_only"
    required_preflight: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class RemoteNetworkOrchestrationPlan:
    phase: str
    topology_id: str
    server_role: RemoteHostRole
    client_role: RemoteHostRole
    local_preparation_only: bool = True
    aws_accessed: bool = False
    remote_accessed: bool = False
    required_profile: str = "slots4-cont-on"
    server_cpu_policy: str = "full_capacity_server_cpu_set_no_local_load_generator_reservation"
    source_state_policy: list[str] = field(default_factory=list)
    artifact_policy: list[str] = field(default_factory=list)
    timing_policy: list[str] = field(default_factory=list)
    privacy_policy: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class NetworkBottleneckGate:
    gate_id: str = "network_tuning_requires_observed_bottleneck"
    default_decision: str = "skip_irq_rps_xps_until_network_bottleneck_is_observed"
    allowed_network_actions: list[str] = field(
        default_factory=lambda: [
            "irq_affinity_profile",
            "rps_profile",
            "xps_profile",
            "evidence_based_listen_backlog",
        ]
    )
    required_evidence: list[str] = field(default_factory=list)
    forbidden_without_evidence: list[str] = field(
        default_factory=lambda: [
            "blind_irq_reassignment",
            "blind_rps_or_xps_change",
            "network_sysctl_sweep",
            "persistent_network_configuration_change",
        ]
    )
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


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
    orchestration: RemoteNetworkOrchestrationPlan
    network_bottleneck_gate: NetworkBottleneckGate
    phase21d2b_execution_order: list[str] = field(default_factory=list)
    recommendation_reuse_policy: dict[str, Any] = field(default_factory=dict)
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
    orchestration = RemoteNetworkOrchestrationPlan(
        phase="21D2A",
        topology_id="c7g_server_t4g_remote_client",
        server_role=RemoteHostRole(
            role_id="server",
            target_class="c7g.2xlarge",
            expected_architecture="linux_aarch64",
            expected_cpu_class="neoverse_v1",
            expected_logical_cpus=8,
            responsibility="run pinned llama-server and preserve server-side artifacts",
            required_preflight=[
                "source_head_matches_controller",
                "tracked_worktree_clean",
                "slots4_cont_on_profile_inspectable",
                "pinned_llama_server_binary_digest_verified",
                "model_digest_verified",
                "server_ports_free_before_measurement",
            ],
        ),
        client_role=RemoteHostRole(
            role_id="remote_load_generator",
            target_class="t4g.small",
            expected_architecture="linux_aarch64",
            expected_cpu_class="neoverse_n1",
            expected_logical_cpus=2,
            responsibility="run deterministic remote load generator and preserve client-side artifacts",
            required_preflight=[
                "source_head_matches_controller",
                "tracked_worktree_clean",
                "client_clock_recorded_but_not_subtracted_from_server_clock",
                "network_path_recorded_without_public_ip_in_tracked_files",
                "client_resource_saturation_checked",
            ],
        ),
        source_state_policy=[
            "controller_head_must_match_server_head",
            "controller_head_must_match_client_head",
            "server_and_client_tracked_worktrees_must_be_clean",
        ],
        artifact_policy=[
            "server_manifest_validated_on_server",
            "client_manifest_validated_on_client",
            "combined_manifest_validated_after_download",
            "raw_server_logs_preserved",
            "raw_client_requests_preserved",
        ],
        timing_policy=[
            "client_observed_ttft_and_e2e_are_primary_for_remote_topology",
            "server_internal_timings_may_be_recorded_as_same-host_evidence_only",
            "do_not_subtract_unsynchronized_server_and_client_wall_clocks",
        ],
        privacy_policy=[
            "public_ipv4_not_written_to_tracked_files",
            "ssh_key_path_not_written_to_tracked_files",
            "host_key_fingerprint_private_evidence_only",
            "model_path_private_evidence_only",
        ],
    )
    gate = NetworkBottleneckGate(
        required_evidence=[
            "remote_client_ttft_or_e2e_regression_relative_to_loopback_shape",
            "server_cpu_not_saturated_while_client_observes_delay",
            "client_cpu_not_saturated",
            "socket_retransmits_or_drops_or_listen_overflow",
            "rx_tx_queue_or_softirq_hotspot",
            "before_and_after_network_counters_from_both_hosts",
        ]
    )
    return NetworkExperimentPlan(
        topology="c7g_server_t4g_remote_load_generator_planned",
        capability_report=capability,
        orchestration=orchestration,
        network_bottleneck_gate=gate,
        phase21d2b_execution_order=[
            "validate_dual_host_source_and_target_fingerprints",
            "take_server_and_client_read_only_snapshots",
            "confirm_slots4_cont_on_profile_on_remote_client_topology",
            "measure_full_capacity_remote_serving",
            "classify_server_client_network_bottleneck",
            "run_irq_rps_xps_profiles_only_if_network_bottleneck_gate_passes",
            "generate_remote_network_recommendation_or_inconclusive_result",
            "validate_server_client_artifacts_and_cleanup",
        ],
        recommendation_reuse_policy={
            "phase21d1_loopback_recommendations": "hard_drift_for_remote_network_topology",
            "remote_network_recommendation_requires": [
                "fresh_remote_client_baseline",
                "fresh_remote_client_candidate_confirmation",
                "matching_server_binary_model_capability_workload_and_protocol_digests",
                "matching_server_and_client_topology_fingerprints",
            ],
        },
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
