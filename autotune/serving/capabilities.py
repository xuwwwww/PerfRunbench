from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from autotune.arm_tune.benchmark.hashes import sha256_file
from autotune.serving.schemas import (
    NOT_AVAILABLE,
    ServerCapabilityReport,
    ServerEndpointCapability,
    ServerOptionDefinition,
    ServerRuntimeIdentity,
)


CONCEPT_FLAGS: dict[str, list[str]] = {
    "parallel_slots": ["--parallel", "--slots"],
    "continuous_batching": ["--cont-batching", "--continuous-batching", "--no-cont-batching"],
    "generation_threads": ["--threads", "-t"],
    "batch_threads": ["--threads-batch", "--threads-batch"],
    "context_size": ["--ctx-size", "-c"],
    "batch_size": ["--batch-size", "-b"],
    "micro_batch_size": ["--ubatch-size", "--micro-batch-size", "-ub"],
    "prompt_cache_behavior": ["--prompt-cache", "--prompt-cache-all", "--prompt-cache-ro"],
    "slot_reuse": ["--slots", "--slot-save-path"],
    "kv_cache_configuration": ["--cache-type-k", "--cache-type-v", "--cache-reuse"],
    "flash_attention": ["--flash-attn", "--flash-attention"],
    "mmap": ["--mmap", "--no-mmap"],
    "mlock": ["--mlock"],
    "numa_options": ["--numa"],
    "http_worker_settings": ["--threads-http", "--http-threads"],
    "request_timeout_controls": ["--timeout", "--timeout-ms", "--timeout-seconds"],
    "speculative_decoding": ["--draft", "--model-draft", "--draft-max"],
}

ENDPOINT_CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "health": ("GET", ["/health"]),
    "metrics": ("GET", ["/metrics"]),
    "slots": ("GET", ["/slots"]),
    "completion": ("POST", ["/completion", "/v1/completions", "/v1/chat/completions"]),
}

REQUEST_FIELD_MARKERS = {
    "stream": ["stream"],
    "seed": ["seed"],
    "temperature": ["temperature", "temp"],
    "response_format": ["response_format"],
    "grammar": ["grammar", "json_schema"],
    "max_tokens": ["max_tokens", "n_predict"],
}


def discover_server_capabilities(
    *,
    executable: str | Path | None = None,
    help_text: str | None = None,
    startup_log_text: str = "",
    endpoint_probe: dict[str, bool] | None = None,
    supported_request_fields: list[str] | None = None,
    source_commit: str = "unknown",
    architecture: str = "unknown",
    backend: str = "unknown",
    kernel_identity: str = "unknown",
    build_evidence_digest: str = "unknown",
) -> ServerCapabilityReport:
    text = help_text if help_text is not None else _read_help(executable)
    combined = text + "\n" + startup_log_text
    supported_flags = sorted(_extract_flags(combined))
    flag_set = set(supported_flags)
    options = {
        concept: ServerOptionDefinition(
            concept=concept,
            availability="available" if any(flag in flag_set for flag in flags) else NOT_AVAILABLE,
            option_names=[flag for flag in flags if flag in flag_set],
            evidence=[f"help flag {flag}" for flag in flags if flag in flag_set],
            effective_value_source="exact_binary_help" if any(flag in flag_set for flag in flags) else NOT_AVAILABLE,
        )
        for concept, flags in sorted(CONCEPT_FLAGS.items())
    }
    endpoints = _endpoint_capabilities(combined, endpoint_probe or {})
    fields = supported_request_fields if supported_request_fields is not None else _request_fields(combined)
    binary_sha256 = sha256_file(Path(executable)) if executable and Path(executable).is_file() else "unknown"
    identity = ServerRuntimeIdentity(
        source_commit=_extract_commit(combined) if source_commit == "unknown" else source_commit,
        binary_sha256=binary_sha256,
        architecture=architecture,
        backend=backend,
        kernel_identity=kernel_identity,
        build_evidence_digest=build_evidence_digest,
    )
    return ServerCapabilityReport(
        runtime_identity=identity,
        options=options,
        endpoints=endpoints,
        supported_request_fields=sorted(set(fields)),
        supported_flags=supported_flags,
    )


def _read_help(executable: str | Path | None) -> str:
    if executable is None:
        return ""
    try:
        completed = subprocess.run(
            [str(executable), "--help"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or "") + "\n" + (completed.stderr or "")


def _extract_flags(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)(?:--[A-Za-z][A-Za-z0-9-]*|-[A-Za-z][A-Za-z0-9-]*)", text))


def _endpoint_capabilities(text: str, probe: dict[str, bool]) -> dict[str, ServerEndpointCapability]:
    endpoints: dict[str, ServerEndpointCapability] = {}
    lowered = text.lower()
    for concept, (method, paths) in ENDPOINT_CONCEPTS.items():
        available = bool(probe.get(concept)) or any(path.lower() in lowered for path in paths)
        endpoints[concept] = ServerEndpointCapability(
            endpoint=concept,
            method=method,
            availability="available" if available else NOT_AVAILABLE,
            evidence=[path for path in paths if path.lower() in lowered] + ([f"probe:{concept}=true"] if probe.get(concept) else []),
        )
    return endpoints


def _request_fields(text: str) -> list[str]:
    lowered = text.lower()
    fields: list[str] = []
    for field, markers in REQUEST_FIELD_MARKERS.items():
        if any(marker in lowered for marker in markers):
            fields.append(field)
    return fields


def _extract_commit(text: str) -> str:
    match = re.search(r"(?:commit|build)[:= ]+([0-9a-f]{7,40})", text, flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"
