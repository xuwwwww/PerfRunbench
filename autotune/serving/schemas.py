from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from autotune.arm_tune.schemas import to_plain, utc_now
from autotune.serving import SERVING_SCHEMA_VERSION


UNKNOWN = "unknown"
NOT_AVAILABLE = "not_available"
TRAFFIC_MODES = {"closed-loop", "open-loop", "burst", "shared-prefix"}
RECOMMENDATION_CLASSES = {
    "interactive_latency",
    "interactive_latency_remote",
    "balanced_multi_user",
    "maximum_throughput",
    "maximum_throughput_remote",
    "shared_prefix",
    "memory_constrained",
}


def stable_digest(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ServerOptionDefinition:
    concept: str
    availability: str
    option_names: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    effective_value_source: str = UNKNOWN
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerEndpointCapability:
    endpoint: str
    method: str
    availability: str
    evidence: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerRuntimeIdentity:
    source_commit: str = UNKNOWN
    binary_sha256: str = UNKNOWN
    architecture: str = UNKNOWN
    backend: str = UNKNOWN
    kernel_identity: str = UNKNOWN
    build_evidence_digest: str = UNKNOWN
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerCapabilityReport:
    runtime_identity: ServerRuntimeIdentity
    options: dict[str, ServerOptionDefinition]
    endpoints: dict[str, ServerEndpointCapability]
    supported_request_fields: list[str] = field(default_factory=list)
    supported_flags: list[str] = field(default_factory=list)
    capability_digest: str = ""
    discovered_at_utc: str = field(default_factory=utc_now)
    schema_version: str = SERVING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.capability_digest:
            object.__setattr__(
                self,
                "capability_digest",
                stable_digest(
                    {
                        "runtime_identity": self.runtime_identity.to_dict(),
                        "options": {key: value.to_dict() for key, value in sorted(self.options.items())},
                        "endpoints": {key: value.to_dict() for key, value in sorted(self.endpoints.items())},
                        "supported_request_fields": sorted(self.supported_request_fields),
                        "supported_flags": sorted(self.supported_flags),
                    }
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingWorkloadProfile:
    workload_id: str
    description: str
    prompt_fixture: str
    prompt_digest: str
    expected_request_structure: dict[str, Any]
    max_context_tokens_required: int
    deterministic_seed: int
    requested_output_tokens: int
    quality_guard: dict[str, Any]
    timeout_seconds: float
    retry_policy: str = "no_retries_during_measured_trials"
    shared_prefix: str | None = None
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingTrafficProfile:
    traffic_mode: str
    target_concurrency: int = 1
    target_request_rate: float | None = None
    request_count: int = 1
    duration_seconds: float | None = None
    connection_reuse: bool = True
    timeout_seconds: float = 30.0
    warmup_requests: int = 0
    retry_policy: str = "none"
    schema_version: str = SERVING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.traffic_mode not in TRAFFIC_MODES:
            raise ValueError(f"traffic_mode must be one of {sorted(TRAFFIC_MODES)}")
        if self.target_concurrency < 1:
            raise ValueError("target_concurrency must be >= 1")
        if self.request_count < 1:
            raise ValueError("request_count must be >= 1")
        if self.retry_policy != "none":
            raise ValueError("measured serving traffic must not silently retry requests")

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerRuntimeProfile:
    profile_id: str
    parallel_slots: int | None = None
    continuous_batching: bool | None = None
    context_size: int | None = None
    effective_context_per_slot: int | None = None
    generation_threads: int | None = None
    batch_threads: int | None = None
    batch_size: int | None = None
    micro_batch_size: int | None = None
    process_cpu_affinity: list[int] | None = None
    scheduler_process_profile: str = "default"
    mmap: bool | None = None
    mlock: bool | None = None
    prompt_cache_policy: str = NOT_AVAILABLE
    kv_cache_type: str = NOT_AVAILABLE
    flash_attention: bool | None = None
    http_worker_settings: dict[str, Any] = field(default_factory=dict)
    server_host: str = "127.0.0.1"
    server_port: int | str = "auto"
    connection_policy: str = "loopback"
    exact_command_line: list[str] = field(default_factory=list)
    exact_effective_startup_state: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerMemoryProfile:
    model_memory_bytes: int | None = None
    kv_cache_allocation_bytes: int | None = None
    context_allocation_bytes: int | None = None
    peak_rss_bytes: int | None = None
    steady_rss_bytes: int | None = None
    available_memory_headroom_bytes: int | None = None
    swap_activity: str = UNKNOWN
    oom_evidence: str = "none_observed"
    page_faults: dict[str, int] = field(default_factory=dict)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServerNetworkProfile:
    mode: str = "loopback"
    topology: str = "local_client"
    network_capability_digest: str = NOT_AVAILABLE
    connection_policy: str = "reuse"
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingExperimentPlan:
    experiment_id: str
    workload: ServingWorkloadProfile
    traffic: ServingTrafficProfile
    candidates: list[ServerRuntimeProfile]
    capability_report: ServerCapabilityReport
    result_origin: str = "planned"
    candidate_cap: int | None = None
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingRequestEvidence:
    request_id: str
    workload_id: str
    client_id: str
    scheduled_timestamp: float
    submitted_timestamp: float
    first_token_timestamp: float | None
    completed_timestamp: float | None
    http_status: int | None
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    server_reported_timings: dict[str, Any] = field(default_factory=dict)
    client_observed_timings: dict[str, float] = field(default_factory=dict)
    error_category: str = ""
    response_digest: str = ""
    quality_guard_result: str = "not_evaluated"
    token_count_source: str = NOT_AVAILABLE
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingMetricSummary:
    metric_name: str
    valid_count: int
    failed_count: int
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: float | None = None
    max: float | None = None
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingExperimentResult:
    experiment_id: str
    candidate_id: str
    workload_id: str
    traffic_profile: ServingTrafficProfile
    runtime_profile: ServerRuntimeProfile
    result_origin: str
    request_count: int
    failed_request_count: int
    metrics: dict[str, ServingMetricSummary]
    request_evidence: list[ServingRequestEvidence] = field(default_factory=list)
    memory_profile: ServerMemoryProfile = field(default_factory=ServerMemoryProfile)
    network_profile: ServerNetworkProfile = field(default_factory=ServerNetworkProfile)
    quality_guard_status: str = "not_evaluated"
    cleanup_status: str = "not_evaluated"
    client_validity: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingRecommendation:
    recommendation_class: str
    status: str
    candidate_id: str | None
    message: str
    guard_results: list[dict[str, Any]] = field(default_factory=list)
    selected_metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SERVING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.recommendation_class not in RECOMMENDATION_CLASSES:
            raise ValueError(f"unknown recommendation_class: {self.recommendation_class}")

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingParetoPoint:
    candidate_id: str
    aggregate_output_tokens_per_second: float | None
    ttft_p95_ms: float | None
    e2e_p95_ms: float | None
    peak_rss_bytes: int | None
    failure_rate: float
    quality_guard_status: str
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingParetoResult:
    experiment_id: str
    frontier_points: list[ServingParetoPoint]
    recommendations: dict[str, ServingRecommendation]
    baseline_candidate_id: str
    candidate_count: int
    eligible_candidate_count: int
    result_origin: str
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)


@dataclass(frozen=True)
class ServingRecommendationBundle:
    bundle_id: str
    source_experiment_id: str
    recommendation_class: str
    status: str
    serving_workload_id: str
    traffic_mode: str
    concurrency: int
    request_rate: float | None
    prompt_digest: str
    input_output_token_distribution: dict[str, Any]
    runtime_profile: ServerRuntimeProfile
    network_profile: ServerNetworkProfile
    client_identity: dict[str, Any]
    latency_percentiles: dict[str, Any]
    throughput: dict[str, Any]
    failure_rate: float
    fairness: dict[str, Any]
    peak_rss_bytes: int | None
    quality_evidence: dict[str, Any]
    server_binary_digest: str
    model_digest: str
    server_capability_digest: str
    hard_drift_keys: list[str] = field(default_factory=list)
    schema_version: str = SERVING_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)
