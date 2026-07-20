from __future__ import annotations

from pathlib import Path
from typing import Any

from autotune.arm_tune.adapters.base import ServerRuntimeAdapter
from autotune.arm_tune.benchmark.llama_server import inspect_llama_server, run_llama_server_worker


class LlamaServerAdapter(ServerRuntimeAdapter):
    name = "llama_server"

    def inspect_runtime(self, job: Any, run_dir: Path) -> Any:
        return inspect_llama_server(job, run_dir)

    def validate_configuration(self, job: Any, inspection: Any | None = None) -> None:
        return None

    def prepare(self, job: Any, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, job: Any, run_dir: Path) -> Any:
        return run_llama_server_worker(job, run_dir)

    def collect_artifacts(self, run_dir: Path) -> dict[str, dict[str, Any]]:
        result = self.parse_result(run_dir)
        return result.get("artifacts", {}).get("artifacts", {}) if isinstance(result, dict) else {}

    def parse_result(self, run_dir: Path) -> Any:
        import json

        path = run_dir / "benchmark_result.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def cleanup(self, run_dir: Path) -> Any:
        return None

    def start_server(self, job: Any, run_dir: Path) -> Any:
        raise NotImplementedError("server lifecycle is owned by the worker vertical slice")

    def wait_until_ready(self, server: Any, job: Any) -> Any:
        raise NotImplementedError("server lifecycle is owned by the worker vertical slice")

    def run_request(self, server: Any, request: Any) -> Any:
        raise NotImplementedError("server request execution is owned by the worker vertical slice")

    def stop_server(self, server: Any, job: Any) -> Any:
        raise NotImplementedError("server lifecycle is owned by the worker vertical slice")
