from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event
from typing import Any

from autotune.arm_tune.targets.protocol import atomic_write_json
from autotune.serving.metrics import summarize_requests
from autotune.serving.schemas import (
    ServingExperimentResult,
    ServingRequestEvidence,
    ServingTrafficProfile,
    ServingWorkloadProfile,
    ServerRuntimeProfile,
)


def run_load_generator(
    *,
    base_url: str,
    workload: ServingWorkloadProfile,
    traffic: ServingTrafficProfile,
    runtime_profile: ServerRuntimeProfile,
    experiment_id: str,
    candidate_id: str,
    output_dir: str | Path | None = None,
    result_origin: str = "test_fixture",
) -> ServingExperimentResult:
    if traffic.retry_policy != "none":
        raise ValueError("serving load generator does not retry measured requests")
    requests = _run_requests(base_url=base_url.rstrip("/"), workload=workload, traffic=traffic)
    metrics = summarize_requests(requests)
    failed = sum(1 for item in requests if item.error_category or item.http_status is None or item.http_status >= 400)
    quality = "passed" if requests and failed == 0 and all(item.quality_guard_result == "passed" for item in requests) else "failed"
    result = ServingExperimentResult(
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        workload_id=workload.workload_id,
        traffic_profile=traffic,
        runtime_profile=runtime_profile,
        result_origin=result_origin,
        request_count=len(requests),
        failed_request_count=failed,
        metrics=metrics,
        request_evidence=requests,
        quality_guard_status=quality,
        cleanup_status="not_applicable_load_generator_only",
        client_validity=_client_validity(requests),
    )
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        _write_jsonl(root / "requests.jsonl", requests)
        atomic_write_json(root / "serving_experiment_result.json", result.to_dict())
    return result


def _run_requests(
    *,
    base_url: str,
    workload: ServingWorkloadProfile,
    traffic: ServingTrafficProfile,
) -> list[ServingRequestEvidence]:
    if traffic.traffic_mode == "open-loop":
        return _run_open_loop(base_url, workload, traffic)
    if traffic.traffic_mode == "burst":
        return _run_burst(base_url, workload, traffic)
    return _run_closed_loop(base_url, workload, traffic)


def _run_closed_loop(base_url: str, workload: ServingWorkloadProfile, traffic: ServingTrafficProfile) -> list[ServingRequestEvidence]:
    count = traffic.request_count
    workers = min(traffic.target_concurrency, count)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _single_request,
                base_url,
                workload,
                traffic,
                request_id=f"{workload.workload_id}-{index:04d}",
                client_id=f"client-{index % workers}",
                scheduled_timestamp=time.perf_counter(),
                start_gate=None,
            )
            for index in range(count)
        ]
        return [future.result() for future in as_completed(futures)]


def _run_open_loop(base_url: str, workload: ServingWorkloadProfile, traffic: ServingTrafficProfile) -> list[ServingRequestEvidence]:
    if traffic.target_request_rate is None or traffic.target_request_rate <= 0:
        raise ValueError("open-loop traffic requires target_request_rate > 0")
    start = time.perf_counter()
    futures = []
    with ThreadPoolExecutor(max_workers=traffic.target_concurrency) as executor:
        for index in range(traffic.request_count):
            scheduled = start + (index / traffic.target_request_rate)
            delay = scheduled - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            futures.append(
                executor.submit(
                    _single_request,
                    base_url,
                    workload,
                    traffic,
                    request_id=f"{workload.workload_id}-{index:04d}",
                    client_id=f"client-{index % traffic.target_concurrency}",
                    scheduled_timestamp=scheduled,
                    start_gate=None,
                )
            )
        return [future.result() for future in as_completed(futures)]


def _run_burst(base_url: str, workload: ServingWorkloadProfile, traffic: ServingTrafficProfile) -> list[ServingRequestEvidence]:
    count = traffic.request_count
    workers = min(traffic.target_concurrency, count)
    gate = Event()
    scheduled = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _single_request,
                base_url,
                workload,
                traffic,
                request_id=f"{workload.workload_id}-{index:04d}",
                client_id=f"client-{index % workers}",
                scheduled_timestamp=scheduled,
                start_gate=gate,
            )
            for index in range(count)
        ]
        gate.set()
        return [future.result() for future in as_completed(futures)]


def _single_request(
    base_url: str,
    workload: ServingWorkloadProfile,
    traffic: ServingTrafficProfile,
    *,
    request_id: str,
    client_id: str,
    scheduled_timestamp: float,
    start_gate: Event | None,
) -> ServingRequestEvidence:
    if start_gate is not None:
        start_gate.wait(timeout=traffic.timeout_seconds)
    submitted = time.perf_counter()
    first_token: float | None = None
    completed: float | None = None
    text_parts: list[str] = []
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    status: int | None = None
    error = ""
    server_timings: dict[str, Any] = {}
    body = {
        "model": "fixture-or-configured-model",
        "stream": True,
        "seed": workload.deterministic_seed,
        "temperature": 0.0,
        "max_tokens": workload.requested_output_tokens,
        "messages": [{"role": "user", "content": workload.prompt_fixture}],
    }
    try:
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=traffic.timeout_seconds) as response:
            status = int(response.status)
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if payload_text == "[DONE]":
                    continue
                payload = json.loads(payload_text)
                content, finish = _content_from_payload(payload)
                if content:
                    if first_token is None:
                        first_token = time.perf_counter()
                    text_parts.append(content)
                if finish:
                    finish_reason = finish
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    input_tokens = _optional_int(usage.get("prompt_tokens"))
                    output_tokens = _optional_int(usage.get("completion_tokens"))
                timings = payload.get("timings")
                if isinstance(timings, dict):
                    server_timings.update(timings)
            completed = time.perf_counter()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        error = "http_error"
        completed = time.perf_counter()
    except TimeoutError:
        error = "timeout"
        completed = time.perf_counter()
    except json.JSONDecodeError:
        error = "malformed_stream"
        completed = time.perf_counter()
    except Exception as exc:
        error = exc.__class__.__name__
        completed = time.perf_counter()
    generated = "".join(text_parts)
    if status is not None and status >= 400 and not error:
        error = "http_error"
    if status is not None and 200 <= status < 300 and not generated and not error:
        error = "incomplete_generation"
    timings = {}
    if first_token is not None:
        timings["ttft_ms"] = round((first_token - submitted) * 1000.0, 6)
    if completed is not None:
        timings["e2e_latency_ms"] = round((completed - submitted) * 1000.0, 6)
        timings["client_scheduling_delay_ms"] = round((submitted - scheduled_timestamp) * 1000.0, 6)
    digest = hashlib.sha256(generated.encode("utf-8")).hexdigest() if generated else ""
    quality = "passed" if generated and not error else "failed"
    token_source = "server_reported" if input_tokens is not None or output_tokens is not None else "not_available"
    return ServingRequestEvidence(
        request_id=request_id,
        workload_id=workload.workload_id,
        client_id=client_id,
        scheduled_timestamp=scheduled_timestamp,
        submitted_timestamp=submitted,
        first_token_timestamp=first_token,
        completed_timestamp=completed,
        http_status=status,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        server_reported_timings=server_timings,
        client_observed_timings=timings,
        error_category=error,
        response_digest=digest,
        quality_guard_result=quality,
        token_count_source=token_source,
    )


def _content_from_payload(payload: dict[str, Any]) -> tuple[str, str | None]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choice = choices[0]
    if not isinstance(choice, dict):
        return "", None
    finish = choice.get("finish_reason")
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content, str(finish) if finish else None
    text = choice.get("text")
    if isinstance(text, str):
        return text, str(finish) if finish else None
    return "", str(finish) if finish else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _client_validity(requests: list[ServingRequestEvidence]) -> dict[str, Any]:
    delays = [
        item.client_observed_timings["client_scheduling_delay_ms"]
        for item in requests
        if "client_scheduling_delay_ms" in item.client_observed_timings
    ]
    max_delay = max(delays) if delays else None
    overloaded = max_delay is not None and max_delay > 100.0
    return {
        "schema_version": "1.0",
        "client_overload_detected": overloaded,
        "max_scheduling_delay_ms": max_delay,
        "clock_source": "time.perf_counter",
        "request_count": len(requests),
    }


def _write_jsonl(path: Path, requests: list[ServingRequestEvidence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in requests:
            handle.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
