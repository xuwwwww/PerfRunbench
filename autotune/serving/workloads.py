from __future__ import annotations

from autotune.serving.schemas import ServingWorkloadProfile, stable_digest


def standard_workloads() -> dict[str, ServingWorkloadProfile]:
    workloads = [
        _workload(
            workload_id="interactive-short",
            description="Short prompt, short-to-medium output for interactive TTFT and E2E latency.",
            prompt="Summarize why measured benchmark provenance matters in two concise sentences.",
            requested_output_tokens=96,
            max_context_tokens_required=512,
        ),
        _workload(
            workload_id="decode-heavy",
            description="Short prompt, long output for decode throughput under concurrent requests.",
            prompt="Write a compact technical checklist for validating a CPU inference benchmark.",
            requested_output_tokens=384,
            max_context_tokens_required=768,
        ),
        _workload(
            workload_id="prefill-heavy",
            description="Long prompt, short output for prompt-processing and queueing behavior.",
            prompt=(
                "You are validating a benchmark pipeline. Evidence must preserve raw output, "
                "artifact digests, target identity, parser status, cleanup status, and statistical "
                "classification. Explain which three fields most directly prevent fixture data from "
                "becoming a measured performance claim."
            ),
            requested_output_tokens=80,
            max_context_tokens_required=1536,
        ),
        _workload(
            workload_id="shared-prefix",
            description="Repeated large prefix with controlled suffix for prefix-cache qualification.",
            prompt=(
                "Shared prefix: PerfRunbench records source identity, model digest, protocol digest, "
                "target fingerprint, raw stdout, raw stderr, artifact manifests, and cleanup evidence. "
                "Request suffix: compare the current request against the shared benchmark policy."
            ),
            requested_output_tokens=160,
            max_context_tokens_required=2048,
            shared_prefix=(
                "PerfRunbench records source identity, model digest, protocol digest, target fingerprint, "
                "raw stdout, raw stderr, artifact manifests, and cleanup evidence."
            ),
        ),
    ]
    return {item.workload_id: item for item in workloads}


def get_standard_workload(workload_id: str) -> ServingWorkloadProfile:
    workloads = standard_workloads()
    try:
        return workloads[workload_id]
    except KeyError as exc:
        raise ValueError(f"unknown serving workload: {workload_id}") from exc


def _workload(
    *,
    workload_id: str,
    description: str,
    prompt: str,
    requested_output_tokens: int,
    max_context_tokens_required: int,
    shared_prefix: str | None = None,
) -> ServingWorkloadProfile:
    expected_request = {
        "schema_version": "1.0",
        "endpoint": "/v1/chat/completions",
        "stream": True,
        "temperature": 0.0,
        "seed": 42,
        "retry_policy": "none",
    }
    return ServingWorkloadProfile(
        workload_id=workload_id,
        description=description,
        prompt_fixture=prompt,
        prompt_digest=stable_digest({"workload_id": workload_id, "prompt": prompt, "shared_prefix": shared_prefix}),
        expected_request_structure=expected_request,
        max_context_tokens_required=max_context_tokens_required,
        deterministic_seed=42,
        requested_output_tokens=requested_output_tokens,
        quality_guard={"schema_version": "1.0", "type": "nonempty_output_digest"},
        timeout_seconds=30.0,
        shared_prefix=shared_prefix,
    )
