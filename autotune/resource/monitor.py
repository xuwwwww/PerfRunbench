from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

from autotune.resource.budget import ResourceBudget


@dataclass
class ResourceSample:
    rss_mb: float
    available_memory_mb: float
    process_cpu_percent: float
    system_cpu_percent: float


@dataclass
class ResourceMonitor:
    budget: ResourceBudget
    interval_seconds: float = 0.1
    samples: list[ResourceSample] = field(default_factory=list)

    def __enter__(self) -> "ResourceMonitor":
        self._stop = threading.Event()
        self._process = None
        self._psutil = None
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except ModuleNotFoundError:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sample_once()

    def summary(self) -> dict:
        if not self.samples:
            return {
                "peak_rss_mb": 0.0,
                "available_memory_after_mb": None,
                "average_process_cpu_percent": 0.0,
                "peak_process_cpu_percent": 0.0,
                "average_system_cpu_percent": 0.0,
                "peak_system_cpu_percent": 0.0,
                "memory_budget_exceeded": False,
                "cpu_quota_exceeded": False,
            }
        rss_values = [sample.rss_mb for sample in self.samples]
        available_values = [sample.available_memory_mb for sample in self.samples]
        process_cpu = [sample.process_cpu_percent for sample in self.samples]
        system_cpu = [sample.system_cpu_percent for sample in self.samples]
        peak_rss = max(rss_values)
        total_memory_mb = None
        if self._psutil is not None:
            total_memory_mb = self._psutil.virtual_memory().total / (1024 * 1024)
        memory_budget = self.budget.effective_memory_budget_mb(total_memory_mb)
        cpu_quota = self.budget.cpu_quota_percent
        return {
            "peak_rss_mb": round(peak_rss, 3),
            "available_memory_after_mb": round(available_values[-1], 3),
            "average_process_cpu_percent": round(sum(process_cpu) / len(process_cpu), 3),
            "peak_process_cpu_percent": round(max(process_cpu), 3),
            "average_system_cpu_percent": round(sum(system_cpu) / len(system_cpu), 3),
            "peak_system_cpu_percent": round(max(system_cpu), 3),
            "memory_budget_exceeded": bool(memory_budget is not None and peak_rss > memory_budget),
            "cpu_quota_exceeded": bool(cpu_quota is not None and max(system_cpu) > cpu_quota),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            time.sleep(self.interval_seconds)

    def _sample_once(self) -> None:
        if self._psutil is not None and self._process is not None:
            memory = self._psutil.virtual_memory()
            self.samples.append(
                ResourceSample(
                    rss_mb=self._process.memory_info().rss / (1024 * 1024),
                    available_memory_mb=memory.available / (1024 * 1024),
                    process_cpu_percent=self._process.cpu_percent(interval=None),
                    system_cpu_percent=self._psutil.cpu_percent(interval=None),
                )
            )
        else:
            self.samples.append(
                ResourceSample(
                    rss_mb=0.0,
                    available_memory_mb=0.0,
                    process_cpu_percent=0.0,
                    system_cpu_percent=0.0,
                )
            )
