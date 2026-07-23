from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from autotune.serving import SERVING_SCHEMA_VERSION
from autotune.serving.schemas import ServerNetworkProfile, ServerRuntimeProfile, stable_digest
from autotune.serving.workloads import get_standard_workload


PHASE21D2B_SOURCE_COMMIT = "b46460ee989baf54ebbeff6275ac9fb0bf866f3f"
PHASE21D2B_EVIDENCE_COMMIT = "d89bada8088cdc933d53081e4e730d00325d16b0"
LLAMA_CPP_SOURCE_COMMIT = "b15ca938ad00aa6b3ee6c2edda7363fd02826b18"
SERVER_BINARY_SHA256 = "6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e"
MODEL_SHA256 = "1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49"
SERVER_CAPABILITY_DIGEST = "46c690b0a2c9aa6e1adc49dd26bf9cb2b7bbc9017b79af46a82a8e59a155a612"
SERVER_CPU_MODEL = "Neoverse-V1"
SERVER_ARCHITECTURE = "aarch64"
SERVER_LOGICAL_CPUS = 8
SERVER_KERNEL_CLASS = "linux_aws_4k"
REMOTE_CLIENT_CPU_MODEL = "Neoverse-N1 class"
REMOTE_CLIENT_LOGICAL_CPUS = 2
REMOTE_TOPOLOGY_ID = "c7g_server_t4g_remote_client"
REMOTE_NETWORK_PATH = "direct_private_network"
CANONICAL_PROFILE_ID = "slots4-cont-on"
CANONICAL_CONCURRENCY = 4

MEASURED_REQUESTS_TOTAL = 608
PAIRED_BLOCK_PAIRS = 6
QUALITY_FAILURES = 0
REQUEST_FAILURES = 0
REQUEST_RETRIES = 0

CANONICAL_PROFILE = {
    "schema_version": SERVING_SCHEMA_VERSION,
    "profile_id": CANONICAL_PROFILE_ID,
    "parallel_slots": 4,
    "continuous_batching": True,
    "context_size": 2048,
    "effective_context_per_slot": 512,
    "generation_threads": 8,
    "batch_threads": 8,
    "batch_size": 512,
    "micro_batch_size": 128,
    "process_cpu_affinity": list(range(8)),
    "scheduler_process_profile": "full_capacity",
    "mmap": True,
    "mlock": False,
    "prompt_cache_policy": "disabled",
    "kv_cache_type": "not_available_for_recommendation",
    "flash_attention": False,
    "server_host": "<server-private-interface>",
    "server_port": "<benchmark_port>",
    "connection_policy": REMOTE_NETWORK_PATH,
    "exact_command_line": [
        "<llama-server>",
        "--model",
        "<model.gguf>",
        "--host",
        "<server-private-interface>",
        "--port",
        "<benchmark_port>",
        "--threads",
        "8",
        "--threads-batch",
        "8",
        "--ctx-size",
        "2048",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
        "--parallel",
        "4",
        "--cont-batching",
    ],
    "exact_effective_startup_state": {
        "schema_version": SERVING_SCHEMA_VERSION,
        "parallel_slots": 4,
        "continuous_batching": "enabled",
        "ctx_size": 2048,
        "batch_size": 512,
        "ubatch_size": 128,
        "flash_attention": "off",
        "slots_endpoint_confirmed": True,
    },
}

REMOTE_RESULT_MATRIX: dict[str, dict[str, Any]] = {
    "interactive-short": {
        "stage": "J1 full eight CPU",
        "baseline_output_tok_s": 17.032,
        "candidate_output_tok_s": 31.355,
        "throughput_delta_percent": 84.092,
        "baseline_ttft_p95_ms": 3055.682,
        "candidate_ttft_p95_ms": 271.088,
        "ttft_p95_reduction_percent": 91.128,
        "bootstrap_p05_delta_percent": 83.467,
        "bootstrap_median_delta_percent": 84.105,
        "bootstrap_p95_delta_percent": 84.742,
    },
    "decode-heavy": {
        "stage": "J1 full eight CPU",
        "baseline_output_tok_s": 17.046,
        "candidate_output_tok_s": 31.790,
        "throughput_delta_percent": 86.490,
        "baseline_ttft_p95_ms": 10126.795,
        "candidate_ttft_p95_ms": 202.424,
        "ttft_p95_reduction_percent": 98.001,
        "bootstrap_p05_delta_percent": 86.231,
        "bootstrap_median_delta_percent": 86.668,
        "bootstrap_p95_delta_percent": 87.104,
    },
    "prefill-heavy": {
        "stage": "J1 full eight CPU",
        "baseline_output_tok_s": 16.563,
        "candidate_output_tok_s": 28.054,
        "throughput_delta_percent": 69.375,
        "baseline_ttft_p95_ms": 3967.424,
        "candidate_ttft_p95_ms": 341.693,
        "ttft_p95_reduction_percent": 91.388,
        "bootstrap_p05_delta_percent": 68.817,
        "bootstrap_median_delta_percent": 69.547,
        "bootstrap_p95_delta_percent": 70.281,
    },
}

CONTROL_RESULT_MATRIX: dict[str, dict[str, Any]] = {
    "interactive-short": {
        "stage": "J2 seven CPU",
        "baseline_output_tok_s": 15.334,
        "candidate_output_tok_s": 28.295,
        "throughput_delta_percent": 84.529,
        "baseline_ttft_p95_ms": 3389.856,
        "candidate_ttft_p95_ms": 300.557,
        "ttft_p95_reduction_percent": 91.134,
        "bootstrap_p05_delta_percent": 83.957,
        "bootstrap_median_delta_percent": 84.778,
        "bootstrap_p95_delta_percent": 85.599,
    }
}

HARD_DRIFT_FIELDS = [
    "server_binary_digest",
    "model_digest",
    "server_capability_digest",
    "architecture",
    "cpu_model",
    "logical_cpus",
    "memory_class",
    "network_topology",
    "network_path",
    "serving_workload_id",
    "traffic_mode",
    "concurrency",
    "cpu_assignment",
    "parallel_slots",
    "continuous_batching",
    "context_size",
    "artifact_manifest_digest",
    "cleanup_status",
]


def canonical_remote_recommendations() -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for workload_id, result in REMOTE_RESULT_MATRIX.items():
        recommendations.append(_recommendation("maximum_throughput_remote", workload_id, result))
        recommendations.append(_recommendation("interactive_latency_remote", workload_id, result))
    return recommendations


def get_remote_recommendation(
    *,
    workload_id: str,
    concurrency: int,
    recommendation_class: str | None = None,
    recommendation_id: str | None = None,
) -> dict[str, Any]:
    for recommendation in canonical_remote_recommendations():
        if recommendation_id and recommendation["recommendation_id"] == recommendation_id:
            return recommendation
        if recommendation_id:
            continue
        if recommendation["serving_workload_id"] != workload_id:
            continue
        if int(recommendation["concurrency"]) != concurrency:
            continue
        if recommendation_class and recommendation["recommendation_class"] != recommendation_class:
            continue
        return recommendation
    if recommendation_id:
        raise ValueError(f"unknown serving recommendation: {recommendation_id}")
    raise ValueError(
        "no validated remote serving recommendation for "
        f"workload={workload_id!r}, concurrency={concurrency!r}, class={recommendation_class!r}"
    )


def recommendation_bundle(recommendation: dict[str, Any]) -> dict[str, Any]:
    workload = get_standard_workload(str(recommendation["serving_workload_id"]))
    payload = {
        "schema_version": SERVING_SCHEMA_VERSION,
        "bundle_id": recommendation["recommendation_id"],
        "source_experiment_id": "phase21d2b_c7g_t4g_remote_serving",
        "recommendation_class": recommendation["recommendation_class"],
        "status": "eligible_preliminary",
        "serving_workload_id": workload.workload_id,
        "traffic_mode": "closed-loop",
        "concurrency": CANONICAL_CONCURRENCY,
        "request_rate": None,
        "prompt_digest": workload.prompt_digest,
        "input_output_token_distribution": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "requested_output_tokens": workload.requested_output_tokens,
            "max_context_tokens_required": workload.max_context_tokens_required,
        },
        "runtime_profile": deepcopy(CANONICAL_PROFILE),
        "network_profile": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "mode": "remote-private",
            "topology": REMOTE_TOPOLOGY_ID,
            "network_capability_digest": "read_only_no_bottleneck_observed",
            "connection_policy": REMOTE_NETWORK_PATH,
        },
        "client_identity": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "cpu_model": REMOTE_CLIENT_CPU_MODEL,
            "logical_cpus": REMOTE_CLIENT_LOGICAL_CPUS,
            "model_present_on_client": False,
            "client_bottleneck_detected": False,
        },
        "latency_percentiles": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "ttft_p95_ms": {
                "baseline": recommendation["metrics"]["baseline_ttft_p95_ms"],
                "candidate": recommendation["metrics"]["candidate_ttft_p95_ms"],
                "reduction_percent": recommendation["metrics"]["ttft_p95_reduction_percent"],
            },
        },
        "throughput": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "aggregate_output_tokens_per_second": {
                "baseline": recommendation["metrics"]["baseline_output_tok_s"],
                "candidate": recommendation["metrics"]["candidate_output_tok_s"],
                "improvement_percent": recommendation["metrics"]["throughput_delta_percent"],
            },
            "bootstrap_delta_percent": {
                "p05": recommendation["metrics"]["bootstrap_p05_delta_percent"],
                "median": recommendation["metrics"]["bootstrap_median_delta_percent"],
                "p95": recommendation["metrics"]["bootstrap_p95_delta_percent"],
            },
        },
        "failure_rate": 0.0,
        "fairness": {"schema_version": SERVING_SCHEMA_VERSION, "client_guard": "passed"},
        "peak_rss_bytes": None,
        "quality_evidence": {"schema_version": SERVING_SCHEMA_VERSION, "quality_guard_status": "passed", "quality_failures": 0},
        "server_binary_digest": SERVER_BINARY_SHA256,
        "model_digest": MODEL_SHA256,
        "server_capability_digest": SERVER_CAPABILITY_DIGEST,
        "hard_drift_keys": HARD_DRIFT_FIELDS,
    }
    payload["bundle_digest"] = stable_digest(payload)
    return payload


def serving_performance_passport(recommendation: dict[str, Any]) -> dict[str, Any]:
    bundle = recommendation_bundle(recommendation)
    passport = {
        "schema_version": SERVING_SCHEMA_VERSION,
        "passport_kind": "serving_performance_passport",
        "passport_version": "serving-v1",
        "status": "eligible_preliminary",
        "verification_status": "passed",
        "recommendation_id": recommendation["recommendation_id"],
        "recommendation_class": recommendation["recommendation_class"],
        "serving_workload_id": recommendation["serving_workload_id"],
        "topology": REMOTE_TOPOLOGY_ID,
        "result_origin": "measured",
        "protocols_combined": False,
        "measured_request_count": MEASURED_REQUESTS_TOTAL,
        "request_failures": REQUEST_FAILURES,
        "request_retries": REQUEST_RETRIES,
        "quality_failures": QUALITY_FAILURES,
        "artifact_validation": "passed",
        "cleanup": "passed",
        "network_tuning_status": "network_tuning_not_executed_evidence_absent",
        "source_evidence": _source_evidence(),
        "recommendation_bundle_digest": bundle["bundle_digest"],
        "recommendation_bundle": bundle,
        "historical_passport_compatibility": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "v0_2_armtune_passports": "preserved_as_separate_passport_kind",
        },
    }
    passport["passport_digest"] = stable_digest(passport)
    return passport


def default_current_context(recommendation: dict[str, Any]) -> dict[str, Any]:
    workload_id = str(recommendation["serving_workload_id"])
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "server_binary_digest": SERVER_BINARY_SHA256,
        "model_digest": MODEL_SHA256,
        "server_capability_digest": SERVER_CAPABILITY_DIGEST,
        "architecture": SERVER_ARCHITECTURE,
        "cpu_model": SERVER_CPU_MODEL,
        "logical_cpus": SERVER_LOGICAL_CPUS,
        "memory_class": "c7g_2xlarge_16gib_class",
        "network_topology": REMOTE_TOPOLOGY_ID,
        "network_path": REMOTE_NETWORK_PATH,
        "serving_workload_id": workload_id,
        "traffic_mode": "closed-loop",
        "concurrency": CANONICAL_CONCURRENCY,
        "cpu_assignment": "cpus_0_7",
        "parallel_slots": 4,
        "continuous_batching": True,
        "context_size": 2048,
        "artifact_manifest_digest": "phase21d2b_revalidated_manifests",
        "cleanup_status": "passed",
        "available_memory_guard": "passed",
        "port_free": True,
    }


def validate_recommendation_context(recommendation: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    expected = default_current_context(recommendation)
    guards = []
    failed_fields: list[str] = []
    missing_fields: list[str] = []
    for field in HARD_DRIFT_FIELDS:
        expected_value = expected.get(field)
        observed_value = current.get(field)
        if observed_value is None:
            guards.append(_guard(field, "failed", "missing", expected_value, observed_value))
            failed_fields.append(field)
            missing_fields.append(field)
        elif observed_value != expected_value:
            guards.append(_guard(field, "failed", "hard_drift", expected_value, observed_value))
            failed_fields.append(field)
        else:
            guards.append(_guard(field, "passed", "matched", expected_value, observed_value))
    if current.get("available_memory_guard") == "failed":
        guards.append(_guard("available_memory_guard", "failed", "memory_guard_failed", "passed", "failed"))
        failed_fields.append("available_memory_guard")
    if current.get("port_free") is False:
        guards.append(_guard("port_free", "failed", "port_not_free", True, False))
        failed_fields.append("port_free")
    state = _applicability_state(failed_fields, missing_fields)
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "state": state,
        "status": "passed" if state == "recommendation_applicable" else "failed",
        "run_best_performs_search": False,
        "guard_results": guards,
    }


def run_best_plan(
    *,
    recommendation: dict[str, Any],
    current_context: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    current = current_context or default_current_context(recommendation)
    validation = validate_recommendation_context(recommendation, current)
    process_started = False
    if dry_run:
        execution_status = "dry_run_no_mutation"
    elif validation["state"] == "recommendation_applicable":
        execution_status = "diagnostic_only_public_runtime_inputs_not_supplied"
    else:
        execution_status = "blocked_by_guard"
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "kind": "serving_run_best",
        "recommendation_id": recommendation["recommendation_id"],
        "recommendation_class": recommendation["recommendation_class"],
        "serving_workload_id": recommendation["serving_workload_id"],
        "concurrency": recommendation["concurrency"],
        "selected_profile": CANONICAL_PROFILE_ID,
        "dry_run": dry_run,
        "run_best_performs_search": False,
        "process_started": process_started,
        "execution_status": execution_status,
        "expected_command": deepcopy(CANONICAL_PROFILE["exact_command_line"]),
        "cpu_assignment": deepcopy(CANONICAL_PROFILE["process_cpu_affinity"]),
        "topology_requirements": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "server": "C7g / Neoverse-V1 / Linux aarch64 / 8 logical CPUs",
            "client": "T4g / Neoverse-N1 class / Linux aarch64 / 2 logical CPUs",
            "network_path": REMOTE_NETWORK_PATH,
            "public_benchmark_port_exposure": False,
        },
        "memory_context_preflight": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "state": current.get("available_memory_guard", "passed"),
            "context_size": 2048,
            "parallel_slots": 4,
        },
        "port_preflight": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "port_free_required": True,
            "observed_port_free": current.get("port_free", True),
        },
        "startup_semantics": deepcopy(CANONICAL_PROFILE["exact_effective_startup_state"]),
        "canary": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "required_before_ready": True,
            "status": "not_executed_in_public_diagnostic" if not dry_run else "planned",
        },
        "ready_state": "validated_from_preserved_evidence" if validation["state"] == "recommendation_applicable" else "not_ready",
        "cleanup": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "cleanup_required": True,
            "status": "no_process_started" if not process_started else "not_evaluated",
        },
        "validation": validation,
    }


def explain_recommendation(recommendation: dict[str, Any]) -> dict[str, Any]:
    metrics = recommendation["metrics"]
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "recommendation_id": recommendation["recommendation_id"],
        "recommendation_class": recommendation["recommendation_class"],
        "selected_profile": CANONICAL_PROFILE_ID,
        "status": "eligible_preliminary",
        "what_changes": [
            "Use four llama-server parallel slots.",
            "Enable continuous batching.",
            "Keep the recommendation bound to concurrency 4 and the direct private remote-client topology.",
        ],
        "why_eligible": {
            "schema_version": SERVING_SCHEMA_VERSION,
            "throughput_improvement_percent": metrics["throughput_delta_percent"],
            "ttft_p95_reduction_percent": metrics["ttft_p95_reduction_percent"],
            "bootstrap_interval_excludes_zero": True,
            "request_failures": REQUEST_FAILURES,
            "retries": REQUEST_RETRIES,
            "quality_failures": QUALITY_FAILURES,
            "artifact_validation": "passed",
            "cleanup": "passed",
        },
        "limits": [
            "This is not a universal llama-server default.",
            "This does not combine loopback, llama-bench, kernel, or build-profile protocols.",
            "Network tuning was not executed because bottleneck evidence was absent.",
        ],
    }


def recommendation_summary_matrix() -> dict[str, Any]:
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "source_evidence": _source_evidence(),
        "j1_full_capacity_remote": deepcopy(REMOTE_RESULT_MATRIX),
        "j2_seven_cpu_control": deepcopy(CONTROL_RESULT_MATRIX),
        "measured_requests_total": MEASURED_REQUESTS_TOTAL,
        "paired_block_pairs": PAIRED_BLOCK_PAIRS,
        "request_retries": REQUEST_RETRIES,
        "request_failures": REQUEST_FAILURES,
        "quality_failures": QUALITY_FAILURES,
        "network_tuning_status": "network_tuning_not_executed_evidence_absent",
    }


def sanitize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _recommendation(recommendation_class: str, workload_id: str, result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": SERVING_SCHEMA_VERSION,
        "recommendation_id": "",
        "recommendation_class": recommendation_class,
        "profile_id": CANONICAL_PROFILE_ID,
        "status": "eligible_preliminary",
        "serving_workload_id": workload_id,
        "traffic_mode": "closed-loop",
        "concurrency": CANONICAL_CONCURRENCY,
        "topology": REMOTE_TOPOLOGY_ID,
        "result_origin": "measured",
        "metrics": deepcopy(result),
        "server_binary_digest": SERVER_BINARY_SHA256,
        "model_digest": MODEL_SHA256,
        "server_capability_digest": SERVER_CAPABILITY_DIGEST,
        "source_evidence": _source_evidence(),
        "wording": (
            "Best observed eligible serving configuration under the tested C7g server, "
            "T4g remote-client topology, Qwen2.5-7B Q4_K_M model, pinned llama-server "
            "binary, workload, concurrency, and measurement protocol."
        ),
    }
    payload["recommendation_id"] = stable_digest(
        {
            "class": recommendation_class,
            "profile": CANONICAL_PROFILE_ID,
            "workload": workload_id,
            "concurrency": CANONICAL_CONCURRENCY,
            "topology": REMOTE_TOPOLOGY_ID,
            "server": SERVER_BINARY_SHA256,
            "model": MODEL_SHA256,
            "source": PHASE21D2B_SOURCE_COMMIT,
        }
    )[:16]
    return payload


def _source_evidence() -> dict[str, Any]:
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "source_commit": PHASE21D2B_SOURCE_COMMIT,
        "evidence_commit": PHASE21D2B_EVIDENCE_COMMIT,
        "llama_cpp_source_commit": LLAMA_CPP_SOURCE_COMMIT,
        "server_binary_sha256": SERVER_BINARY_SHA256,
        "model_sha256": MODEL_SHA256,
        "server_capability_digest": SERVER_CAPABILITY_DIGEST,
        "result_document": "docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md",
    }


def _guard(field: str, status: str, reason: str, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "field": field,
        "status": status,
        "reason": reason,
        "expected": expected,
        "observed": observed,
    }


def _applicability_state(failed_fields: list[str], missing_fields: list[str]) -> str:
    if not failed_fields:
        return "recommendation_applicable"
    if "available_memory_guard" in failed_fields:
        return "memory_guard_failed"
    if any(field in failed_fields for field in {"server_capability_digest", "server_binary_digest", "architecture"}):
        return "capability_mismatch"
    if any(field in failed_fields for field in {"cpu_model", "logical_cpus", "network_topology", "network_path", "cpu_assignment"}):
        return "topology_mismatch"
    if any(field in failed_fields for field in {"serving_workload_id", "concurrency", "traffic_mode"}):
        return "workload_mismatch"
    if missing_fields:
        return "diagnostic_only"
    return "hard_drift"

