from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from autotune.serving.loadgen import run_load_generator
from autotune.serving.schemas import ServerRuntimeProfile, ServingTrafficProfile
from autotune.serving.workloads import get_standard_workload


class ServingLoadGeneratorTest(unittest.TestCase):
    def test_closed_loop_concurrency_records_jsonl_tokens_and_timings(self) -> None:
        server = _FixtureServingServer(delay=0.05)
        server.start()
        try:
            with TemporaryDirectory() as temp_dir:
                result = run_load_generator(
                    base_url=server.base_url,
                    workload=get_standard_workload("interactive-short"),
                    traffic=ServingTrafficProfile(traffic_mode="closed-loop", target_concurrency=2, request_count=4, timeout_seconds=2),
                    runtime_profile=ServerRuntimeProfile(profile_id="baseline"),
                    experiment_id="closed-loop",
                    candidate_id="baseline",
                    output_dir=temp_dir,
                )
                self.assertEqual(result.failed_request_count, 0)
                self.assertEqual(result.quality_guard_status, "passed")
                self.assertEqual(result.metrics["ttft_ms"].valid_count, 4)
                self.assertTrue(all(item.token_count_source == "server_reported" for item in result.request_evidence))
                self.assertTrue((Path(temp_dir) / "requests.jsonl").exists())
                self.assertGreaterEqual(server.max_inflight, 2)
        finally:
            server.stop()

    def test_open_loop_records_requested_and_actual_submission_times(self) -> None:
        server = _FixtureServingServer(delay=0.001)
        server.start()
        try:
            result = run_load_generator(
                base_url=server.base_url,
                workload=get_standard_workload("decode-heavy"),
                traffic=ServingTrafficProfile(
                    traffic_mode="open-loop",
                    target_concurrency=2,
                    target_request_rate=20.0,
                    request_count=3,
                    timeout_seconds=2,
                ),
                runtime_profile=ServerRuntimeProfile(profile_id="baseline"),
                experiment_id="open-loop",
                candidate_id="baseline",
            )
            self.assertEqual(result.failed_request_count, 0)
            self.assertTrue(all(item.submitted_timestamp >= item.scheduled_timestamp for item in result.request_evidence))
            self.assertFalse(result.client_validity["client_overload_detected"])
        finally:
            server.stop()

    def test_burst_and_shared_prefix_modes_preserve_result_semantics(self) -> None:
        server = _FixtureServingServer(delay=0.001)
        server.start()
        try:
            for mode, workload_id in [("burst", "interactive-short"), ("shared-prefix", "shared-prefix")]:
                result = run_load_generator(
                    base_url=server.base_url,
                    workload=get_standard_workload(workload_id),
                    traffic=ServingTrafficProfile(traffic_mode=mode, target_concurrency=3, request_count=3, timeout_seconds=2),
                    runtime_profile=ServerRuntimeProfile(profile_id="baseline"),
                    experiment_id=mode,
                    candidate_id="baseline",
                )
                self.assertEqual(result.failed_request_count, 0)
                self.assertEqual(result.result_origin, "test_fixture")
                self.assertEqual(result.cleanup_status, "not_applicable_load_generator_only")
        finally:
            server.stop()

    def test_http_error_and_malformed_stream_are_failures_not_zero_metrics(self) -> None:
        for mode, expected_error in [("http_error", "http_error"), ("malformed", "malformed_stream")]:
            server = _FixtureServingServer(mode=mode)
            server.start()
            try:
                result = run_load_generator(
                    base_url=server.base_url,
                    workload=get_standard_workload("interactive-short"),
                    traffic=ServingTrafficProfile(traffic_mode="closed-loop", target_concurrency=1, request_count=1, timeout_seconds=2),
                    runtime_profile=ServerRuntimeProfile(profile_id="baseline"),
                    experiment_id=mode,
                    candidate_id="baseline",
                )
                self.assertEqual(result.failed_request_count, 1)
                self.assertEqual(result.request_evidence[0].error_category, expected_error)
                self.assertIsNone(result.metrics["ttft_ms"].p50)
            finally:
                server.stop()

    def test_retry_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ServingTrafficProfile(traffic_mode="closed-loop", retry_policy="retry-on-timeout")


class _FixtureServingServer:
    def __init__(self, *, mode: str = "success", delay: float = 0.0) -> None:
        self.mode = mode
        self.delay = delay
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.inflight = 0
        self.max_inflight = 0

    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                with owner._lock:
                    owner.inflight += 1
                    owner.max_inflight = max(owner.max_inflight, owner.inflight)
                try:
                    if owner.mode == "http_error":
                        self.send_response(500)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    if owner.mode == "malformed":
                        self.wfile.write(b"data: {not-json}\n\n")
                        self.wfile.flush()
                        return
                    _event(self, {"choices": [{"delta": {"content": "Hello"}}]})
                    if owner.delay:
                        time.sleep(owner.delay)
                    _event(
                        self,
                        {
                            "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
                            "timings": {"server_ttft_ms": 1.5},
                        },
                    )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                finally:
                    with owner._lock:
                        owner.inflight -= 1

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _event(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    handler.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode("utf-8"))
    handler.wfile.flush()


if __name__ == "__main__":
    unittest.main()
