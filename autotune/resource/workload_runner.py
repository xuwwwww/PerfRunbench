from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from autotune.resource.affinity import apply_cpu_affinity
from autotune.resource.budget import ResourceBudget
from autotune.resource.run_state import RunManifest, create_run, finish_run, write_json


@dataclass
class ChildSample:
    timestamp: float
    rss_mb: float
    child_rss_mb: float
    available_memory_mb: float
    child_cpu_percent: float
    system_cpu_percent: float


def run_with_budget(
    command: list[str],
    budget: ResourceBudget,
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    run_dir: Path | None = None,
    manifest: RunManifest | None = None,
) -> tuple[int, Path]:
    if not command:
        raise ValueError("command cannot be empty")
    if run_dir is None or manifest is None:
        run_dir, manifest = create_run(command, budget)
    timeline: list[ChildSample] = []
    process = None
    return_code = 1
    status = "failed"
    try:
        affinity_context = apply_cpu_affinity(budget)
        manifest.notes.append(f"affinity_context={affinity_context}")
        process = subprocess.Popen(command)
        return_code = _monitor_child(process, budget, timeline, sample_interval_seconds, hard_kill)
        status = "completed" if return_code == 0 else "failed"
    except KeyboardInterrupt:
        status = "interrupted"
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        return_code = 130
    finally:
        write_json(run_dir / "resource_timeline.json", [asdict(sample) for sample in timeline])
        write_json(run_dir / "resource_summary.json", _summarize_timeline(timeline, budget))
        finish_run(run_dir, manifest, status, return_code)
    return return_code, run_dir


def _monitor_child(
    process: subprocess.Popen,
    budget: ResourceBudget,
    timeline: list[ChildSample],
    sample_interval_seconds: float,
    hard_kill: bool,
) -> int:
    try:
        import psutil
    except ModuleNotFoundError:
        return process.wait()

    child = psutil.Process(process.pid)
    child.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
    while process.poll() is None:
        try:
            sample = _sample_child(child, psutil)
            timeline.append(sample)
            if hard_kill and _exceeds_memory_budget(sample, budget, psutil):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return process.returncode if process.returncode is not None else 137
        except psutil.NoSuchProcess:
            break
        time.sleep(sample_interval_seconds)
    return process.wait()


def _sample_child(child, psutil) -> ChildSample:
    children = child.children(recursive=True)
    rss = child.memory_info().rss
    cpu = child.cpu_percent(interval=None)
    for item in children:
        try:
            rss += item.memory_info().rss
            cpu += item.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            pass
    memory = psutil.virtual_memory()
    return ChildSample(
        timestamp=time.time(),
        rss_mb=rss / (1024 * 1024),
        child_rss_mb=rss / (1024 * 1024),
        available_memory_mb=memory.available / (1024 * 1024),
        child_cpu_percent=cpu,
        system_cpu_percent=psutil.cpu_percent(interval=None),
    )


def _exceeds_memory_budget(sample: ChildSample, budget: ResourceBudget, psutil) -> bool:
    total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
    effective_budget = budget.effective_memory_budget_mb(total_memory_mb)
    return effective_budget is not None and sample.rss_mb > effective_budget


def _summarize_timeline(timeline: list[ChildSample], budget: ResourceBudget) -> dict:
    if not timeline:
        return {
            "samples": 0,
            "peak_rss_mb": 0.0,
            "average_child_cpu_percent": 0.0,
            "peak_child_cpu_percent": 0.0,
            "memory_budget_exceeded": False,
        }
    peak_rss = max(sample.rss_mb for sample in timeline)
    child_cpu = [sample.child_cpu_percent for sample in timeline]
    total_memory_mb = None
    try:
        import psutil

        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
    except ModuleNotFoundError:
        pass
    effective_budget = budget.effective_memory_budget_mb(total_memory_mb)
    return {
        "samples": len(timeline),
        "peak_rss_mb": round(peak_rss, 3),
        "available_memory_after_mb": round(timeline[-1].available_memory_mb, 3),
        "average_child_cpu_percent": round(sum(child_cpu) / len(child_cpu), 3),
        "peak_child_cpu_percent": round(max(child_cpu), 3),
        "memory_budget_mb": round(budget.memory_budget_mb, 3) if budget.memory_budget_mb is not None else None,
        "effective_memory_budget_mb": round(effective_budget, 3) if effective_budget is not None else None,
        "memory_budget_exceeded": bool(effective_budget is not None and peak_rss > effective_budget),
    }
