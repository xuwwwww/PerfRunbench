from __future__ import annotations

from pathlib import Path
from typing import Any

from autotune.arm_tune.adapters.base import RuntimeAdapter
from autotune.arm_tune.benchmark.llama_bench import inspect_llama_bench, run_llama_bench_worker


class LlamaBenchAdapter(RuntimeAdapter):
    name = "llama_bench"

    def inspect_runtime(self, job: Any, run_dir: Path) -> Any:
        return inspect_llama_bench(job, run_dir)

    def validate_configuration(self, job: Any, inspection: Any | None = None) -> None:
        return None

    def prepare(self, job: Any, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, job: Any, run_dir: Path) -> Any:
        return run_llama_bench_worker(job, run_dir)

    def collect_artifacts(self, run_dir: Path) -> dict[str, dict[str, Any]]:
        result = self.parse_result(run_dir)
        return result.get("artifacts", {}).get("artifacts", {}) if isinstance(result, dict) else {}

    def parse_result(self, run_dir: Path) -> Any:
        import json

        path = run_dir / "benchmark_result.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def cleanup(self, run_dir: Path) -> Any:
        return None
