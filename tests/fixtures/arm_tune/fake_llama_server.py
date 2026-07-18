#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
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
        flags = ["--model", "--host", "--port", "--threads", "--ctx-size", "--batch-size", "--seed", "--temp"]
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
            _ = self.rfile.read(length)
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
            self._event('data: {"choices":[{"delta":{"role":"assistant"}}]}')
            self._event('data: {"choices":[{"delta":{"content":""}}]}')
            if mode == "empty_content":
                self._event('data: {"choices":[{"finish_reason":"stop","delta":{}}]}')
                self._event("data: [DONE]")
                return
            self._event('data: {"choices":[{"delta":{"content":"Hello"}}]}')
            if mode == "mid_stream_disconnect":
                return
            self._event('data: {"choices":[{"delta":{"content":" world"}}]}')
            self._event('data: {"choices":[{"finish_reason":"stop","delta":{}}]}')
            self._event("data: [DONE]")

        def _event(self, line: str) -> None:
            self.wfile.write((line + "\n\n").encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.01)

    server = ThreadingHTTPServer((parsed.host, parsed.port), Handler)
    print(f"fake llama-server listening on {parsed.host}:{parsed.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
