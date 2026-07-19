#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--version" in args:
        print("fake llama-server fixture build: b15ca938")
        return 0
    if "--help" in args:
        mode = os.environ.get("FAKE_LLAMA_SERVER_HELP_MODE", "full")
        flags = ["--model", "--host", "--port", "--threads", "--ctx-size", "--batch-size", "--ubatch-size", "--seed", "--temp"]
        if mode == "missing_port":
            flags.remove("--port")
        print("fake llama-server fixture help")
        print(" ".join(flags))
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--threads", "-t", type=int, default=1)
    parser.add_argument("--ctx-size", "-c", type=int, default=512)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ubatch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", "--temperature", dest="temperature", type=float, default=0.0)
    parsed = parser.parse_args(args)
    args_path = os.environ.get("FAKE_LLAMA_SERVER_ARGS_PATH")
    if args_path:
        Path(args_path).write_text(json.dumps(args), encoding="utf-8")
    mode = os.environ.get("FAKE_LLAMA_SERVER_MODE", "success")
    if mode == "exit_before_ready":
        print("fake server exiting before ready", flush=True)
        time.sleep(0.1)
        return 17
    if mode == "ignore_sigterm":
        signal.signal(signal.SIGTERM, lambda signum, frame: None)
    ready_at = time.monotonic() + float(os.environ.get("FAKE_LLAMA_SERVER_READY_DELAY", "0"))
    request_count = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            if mode == "startup_timeout" or time.monotonic() < ready_at:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"not ready")
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if mode == "http_error":
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"server error")
                return
            if mode == "request_timeout":
                time.sleep(5)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if mode == "malformed_sse":
                self._event("data: {not-json}")
                self._event("data: [DONE]")
                return
            quality_mode = os.environ.get("FAKE_LLAMA_SERVER_QUALITY_MODE", "")
            if quality_mode.startswith("boottrace"):
                response_text = _quality_response_text(body, parsed, quality_mode)
                delay_profile = os.environ.get("FAKE_LLAMA_SERVER_DELAY_PROFILE", "")
                if delay_profile == "quality":
                    time.sleep(_quality_ttft_delay(parsed))
                self._event('data: {"choices":[{"delta":{"role":"assistant"}}]}')
                self._event("data: " + json.dumps({"choices": [{"delta": {"content": response_text}}]}))
                self._event('data: {"choices":[{"finish_reason":"stop","delta":{}}]}')
                self._event("data: [DONE]")
                _maybe_shutdown(server, request_count)
                return
            self._event('data: {"choices":[{"delta":{"role":"assistant"}}]}')
            self._event('data: {"choices":[{"delta":{"content":""}}]}')
            if mode == "empty_content":
                self._event('data: {"choices":[{"finish_reason":"stop","delta":{}}]}')
                self._event("data: [DONE]")
                _maybe_shutdown(server, request_count)
                return
            self._event('data: {"choices":[{"delta":{"content":"Hello"}}]}')
            if mode == "mid_stream_disconnect":
                _maybe_shutdown(server, request_count)
                return
            self._event('data: {"choices":[{"delta":{"content":" world"}}]}')
            self._event('data: {"choices":[{"finish_reason":"stop","delta":{}}]}')
            self._event("data: [DONE]")
            _maybe_shutdown(server, request_count)

        def _event(self, line: str) -> None:
            self.wfile.write((line + "\n\n").encode("utf-8"))
            self.wfile.flush()
            time.sleep(float(os.environ.get("FAKE_LLAMA_SERVER_EVENT_DELAY", "0.01")))

    server = ThreadingHTTPServer((parsed.host, parsed.port), Handler)
    print(f"fake llama-server listening on {parsed.host}:{parsed.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


def _quality_response_text(body: bytes, parsed: argparse.Namespace, quality_mode: str) -> str:
    prompt = _prompt_from_request(body)
    case_id = _case_id_from_prompt(prompt)
    if quality_mode == "boottrace_invalid_json":
        return "{not-json"
    if quality_mode.startswith("boottrace_tuning"):
        return _tuning_quality_response_text(case_id, parsed, quality_mode)
    if quality_mode == "boottrace_fail_all" or (parsed.batch_size is not None and parsed.batch_size >= 256):
        return json.dumps(
            {
                "case_id": case_id,
                "status": "ok",
                "event_sequence_valid": False,
                "latency_ms": 12.5,
                "summary": "I invented one missing BootTrace line.",
                "evidence_line_ids": ["L001", "L999"],
            },
            sort_keys=True,
        )
    response = json.dumps(
        {
            "case_id": case_id,
            "status": _status_for_case(case_id),
            "event_sequence_valid": True,
            "latency_ms": _latency_for_case(case_id),
            "summary": _summary_for_case(case_id),
            "evidence_line_ids": _evidence_for_case(case_id),
        },
        sort_keys=True,
    )
    if quality_mode == "boottrace_fenced":
        return "```json\n" + response + "\n```"
    if quality_mode == "boottrace_fenced_with_prose":
        return "Here is the result:\n```json\n" + response + "\n```"
    return response


def _tuning_quality_response_text(case_id: str, parsed: argparse.Namespace, quality_mode: str) -> str:
    if quality_mode == "boottrace_tuning_invalid_json":
        return "{not-json"
    if quality_mode == "boottrace_tuning_fail_all" or (parsed.batch_size is not None and parsed.batch_size >= 256):
        response = json.dumps(
            {
                "case_id": case_id,
                "status": "success",
                "failure_stage": "none",
                "root_cause_category": "none",
                "evidence_line_ids": ["L999"],
            },
            sort_keys=True,
        )
    else:
        response = json.dumps(
            {
                "case_id": case_id,
                "status": _tuning_status_for_case(case_id),
                "failure_stage": _tuning_failure_stage_for_case(case_id),
                "root_cause_category": _tuning_root_cause_for_case(case_id),
                "evidence_line_ids": _tuning_evidence_for_case(case_id),
            },
            sort_keys=True,
        )
    if quality_mode == "boottrace_tuning_fenced":
        return "```json\n" + response + "\n```"
    if quality_mode == "boottrace_tuning_fenced_with_prose":
        return "Here is the result:\n```json\n" + response + "\n```"
    return response


def _prompt_from_request(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return ""
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts)


def _case_id_from_prompt(prompt: str) -> str:
    matches = re.findall(r"CASE_ID=([A-Za-z0-9_.-]+)", prompt)
    return matches[-1] if matches else "unknown"


def _status_for_case(case_id: str) -> str:
    if case_id == "boot_retry_recovered":
        return "warning"
    if case_id in {"boot_config_missing", "boot_device_timeout", "boot_firmware_component_failure"}:
        return "failure"
    return "success"


def _latency_for_case(case_id: str) -> float:
    return {
        "boot_success": 83.0,
        "boot_retry_recovered": 142.0,
        "boot_config_missing": 36.0,
        "boot_device_timeout": 842.0,
        "boot_firmware_component_failure": 315.0,
    }.get(case_id, 50.0)


def _summary_for_case(case_id: str) -> str:
    if case_id == "boot_retry_recovered":
        return "Boot completed after one retry using cited BootTrace evidence."
    if case_id == "boot_config_missing":
        return "Boot failed because a required configuration line is missing."
    if case_id == "boot_device_timeout":
        return "Boot failed because the NVMe controller timed out before recovery shell entry."
    if case_id == "boot_firmware_component_failure":
        return "Boot failed because a firmware component checksum failure was reported before rollback."
    return "Boot completed successfully using cited BootTrace evidence."


def _evidence_for_case(case_id: str) -> list[str]:
    return {
        "boot_success": ["L001", "L002", "L003"],
        "boot_retry_recovered": ["L010", "L011", "L012", "L013"],
        "boot_config_missing": ["L020", "L021"],
        "boot_device_timeout": ["L030", "L031", "L032"],
        "boot_firmware_component_failure": ["L040", "L041", "L042", "L043"],
    }.get(case_id, ["L001"])


def _tuning_status_for_case(case_id: str) -> str:
    return {
        "tuning_successful_boot": "success",
        "tuning_recoverable_warning": "warning",
        "tuning_device_initialization_timeout": "failure",
        "tuning_missing_boot_device": "failure",
        "tuning_firmware_component_failure": "failure",
        "tuning_incomplete_unknown": "unknown",
        "tuning_retry_recovered": "warning",
    }.get(case_id, "unknown")


def _tuning_failure_stage_for_case(case_id: str) -> str:
    return {
        "tuning_successful_boot": "none",
        "tuning_recoverable_warning": "device_initialization",
        "tuning_device_initialization_timeout": "device_initialization",
        "tuning_missing_boot_device": "boot_selection",
        "tuning_firmware_component_failure": "firmware_validation",
        "tuning_incomplete_unknown": "evidence_collection",
        "tuning_retry_recovered": "boot_selection",
    }.get(case_id, "evidence_collection")


def _tuning_root_cause_for_case(case_id: str) -> str:
    return {
        "tuning_successful_boot": "none",
        "tuning_recoverable_warning": "recoverable_device_warning",
        "tuning_device_initialization_timeout": "device_initialization_timeout",
        "tuning_missing_boot_device": "boot_device_missing",
        "tuning_firmware_component_failure": "firmware_component_failure",
        "tuning_incomplete_unknown": "insufficient_evidence",
        "tuning_retry_recovered": "boot_device_missing",
    }.get(case_id, "insufficient_evidence")


def _tuning_evidence_for_case(case_id: str) -> list[str]:
    return {
        "tuning_successful_boot": ["L101", "L102", "L103", "L104"],
        "tuning_recoverable_warning": ["L111", "L112", "L113", "L114"],
        "tuning_device_initialization_timeout": ["L121", "L122", "L123", "L124"],
        "tuning_missing_boot_device": ["L131", "L132", "L133", "L134"],
        "tuning_firmware_component_failure": ["L141", "L142", "L143", "L144"],
        "tuning_incomplete_unknown": ["L151", "L152", "L153", "L154"],
        "tuning_retry_recovered": ["L161", "L162", "L163", "L164", "L165"],
    }.get(case_id, ["L151"])


def _quality_ttft_delay(parsed: argparse.Namespace) -> float:
    if parsed.batch_size is not None and parsed.batch_size >= 256:
        return 0.002
    if parsed.threads == 1:
        return 0.005
    return 0.05


def _maybe_shutdown(server: ThreadingHTTPServer, request_count: dict[str, int]) -> None:
    try:
        limit = int(os.environ.get("FAKE_LLAMA_SERVER_EXIT_AFTER_REQUESTS", "0"))
    except ValueError:
        limit = 0
    if limit <= 0:
        return
    request_count["count"] += 1
    if request_count["count"] >= limit:
        threading.Thread(target=server.shutdown, daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
