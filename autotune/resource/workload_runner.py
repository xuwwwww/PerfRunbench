from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from autotune.resource.affinity import apply_cpu_affinity
from autotune.resource.budget import ResourceBudget
from autotune.resource.cgroup_monitor import CgroupStats, read_cgroup_stats, wait_for_systemd_control_group
from autotune.resource.executor_capabilities import collect_executor_capabilities
from autotune.resource.run_state import RunManifest, create_run, finish_run, write_json
from autotune.resource.systemd_executor import build_systemd_run_command, make_systemd_scope_name


@dataclass
class ChildSample:
    timestamp: float
    rss_mb: float
    child_rss_mb: float
    available_memory_mb: float
    child_cpu_percent: float
    system_cpu_percent: float
    cgroup_path: str | None = None
    cgroup_memory_current_mb: float | None = None
    cgroup_memory_peak_mb: float | None = None
    cgroup_cpu_percent: float | None = None
    cgroup_cpu_usage_usec: int | None = None


def run_with_budget(
    command: list[str],
    budget: ResourceBudget,
    sample_interval_seconds: float = 0.5,
    hard_kill: bool = False,
    run_dir: Path | None = None,
    manifest: RunManifest | None = None,
    executor: str = "local",
    use_sudo: bool = False,
    allow_sudo_auto: bool = False,
) -> tuple[int, Path]:
    if not command:
        raise ValueError("command cannot be empty")
    if run_dir is None or manifest is None:
        run_dir, manifest = create_run(command, budget)
    manifest.budget = budget.to_record(total_cores=_visible_cpu_count(), total_memory_mb=_visible_memory_mb())
    timeline: list[ChildSample] = []
    process = None
    return_code = 1
    status = "failed"
    try:
        command_to_run = command
        selected_executor, selected_use_sudo, selection_notes = _resolve_executor(
            executor,
            use_sudo=use_sudo,
            allow_sudo_auto=allow_sudo_auto,
        )
        manifest.notes.extend(selection_notes)
        if selected_executor == "local":
            affinity_context = apply_cpu_affinity(budget)
            manifest.notes.append(f"affinity_context={affinity_context}")
        elif selected_executor == "systemd":
            unit_name = make_systemd_scope_name(manifest.run_id)
            systemd_command = build_systemd_run_command(
                command,
                budget,
                use_sudo=selected_use_sudo,
                unit_name=unit_name,
            )
            command_to_run = systemd_command.command
            manifest.notes.extend(systemd_command.notes)
            manifest.notes.append("systemd executor applies hard limits and samples the scope cgroup when available.")
        else:
            raise ValueError(f"unsupported executor: {selected_executor}")
        process = subprocess.Popen(command_to_run)
        if selected_executor == "systemd":
            return_code, control_group = _monitor_systemd_scope(
                process,
                unit_name,
                budget,
                timeline,
                sample_interval_seconds,
                hard_kill,
                use_sudo=selected_use_sudo,
            )
            if control_group:
                manifest.notes.append(f"systemd_control_group={control_group}")
            else:
                manifest.notes.append("systemd_control_group=unavailable")
        else:
            return_code = _monitor_child(process, budget, timeline, sample_interval_seconds, hard_kill)
        status = "completed" if return_code == 0 else "failed"
    except RuntimeError as exc:
        manifest.notes.append(f"runtime_error={exc}")
        raise
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


def _resolve_executor(executor: str, *, use_sudo: bool, allow_sudo_auto: bool) -> tuple[str, bool, list[str]]:
    if executor not in {"auto", "local", "systemd"}:
        raise ValueError(f"unsupported executor: {executor}")
    if executor != "auto":
        return executor, use_sudo, [
            f"requested_executor={executor}",
            f"selected_executor={executor}",
            f"sudo_used={use_sudo}",
        ]

    capabilities = collect_executor_capabilities(probe_systemd=True, check_sudo_cache=True)
    selected = capabilities.get("recommended_executor", "local")
    executors = capabilities.get("executors", {})
    notes = [
        "requested_executor=auto",
        f"selected_executor={selected}",
        f"executor_platform={capabilities.get('platform')}",
    ]
    if selected == "systemd":
        systemd = executors.get("systemd", {})
        requires_sudo = systemd.get("requires_sudo")
        sudo_available = systemd.get("sudo_available")
        if requires_sudo and not (allow_sudo_auto or use_sudo):
            raise RuntimeError(
                "auto executor selected systemd, but this machine requires sudo for transient scopes. "
                "Run sudo -v and pass --allow-sudo-auto, or explicitly pass --executor systemd --sudo."
            )
        if requires_sudo and not sudo_available:
            raise RuntimeError("auto executor selected systemd, but sudo is not available on this machine.")
        selected_use_sudo = bool(use_sudo or (requires_sudo and allow_sudo_auto))
        notes.extend(
            [
                f"systemd_requires_sudo={requires_sudo}",
                f"sudo_cached={systemd.get('sudo_cached')}",
                f"sudo_used={selected_use_sudo}",
            ]
        )
        return "systemd", selected_use_sudo, notes

    notes.append("sudo_used=False")
    return "local", False, notes


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


def _monitor_systemd_scope(
    process: subprocess.Popen,
    unit_name: str,
    budget: ResourceBudget,
    timeline: list[ChildSample],
    sample_interval_seconds: float,
    hard_kill: bool,
    *,
    use_sudo: bool,
) -> tuple[int, str | None]:
    try:
        import psutil
    except ModuleNotFoundError:
        psutil = None

    child = None
    if psutil is not None:
        try:
            child = psutil.Process(process.pid)
            child.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            child = None

    control_group = wait_for_systemd_control_group(unit_name)
    previous_stats: CgroupStats | None = None
    while process.poll() is None:
        stats = read_cgroup_stats(control_group) if control_group else None
        sample = _sample_systemd_scope(child, psutil, stats, previous_stats)
        timeline.append(sample)
        if stats is not None:
            previous_stats = stats
        if hard_kill and _exceeds_memory_budget(sample, budget, psutil):
            _terminate_systemd_scope(unit_name, process, use_sudo=use_sudo)
            return process.returncode if process.returncode is not None else 137
        time.sleep(sample_interval_seconds)
    return process.wait(), control_group


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


def _sample_systemd_scope(child, psutil, stats: CgroupStats | None, previous_stats: CgroupStats | None) -> ChildSample:
    process_rss_mb = 0.0
    process_cpu_percent = 0.0
    system_cpu_percent = 0.0
    available_memory_mb = 0.0
    if psutil is not None:
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        system_cpu_percent = psutil.cpu_percent(interval=None)
        if child is not None:
            try:
                process_sample = _sample_child(child, psutil)
                process_rss_mb = process_sample.rss_mb
                process_cpu_percent = process_sample.child_cpu_percent
            except psutil.NoSuchProcess:
                pass
    cgroup_memory_mb = stats.memory_current_mb if stats is not None else None
    return ChildSample(
        timestamp=time.time(),
        rss_mb=cgroup_memory_mb if cgroup_memory_mb is not None else process_rss_mb,
        child_rss_mb=process_rss_mb,
        available_memory_mb=available_memory_mb,
        child_cpu_percent=process_cpu_percent,
        system_cpu_percent=system_cpu_percent,
        cgroup_path=stats.cgroup_path if stats is not None else None,
        cgroup_memory_current_mb=cgroup_memory_mb,
        cgroup_memory_peak_mb=stats.memory_peak_mb if stats is not None else None,
        cgroup_cpu_percent=_cgroup_cpu_percent(stats, previous_stats),
        cgroup_cpu_usage_usec=stats.cpu_usage_usec if stats is not None else None,
    )


def _cgroup_cpu_percent(stats: CgroupStats | None, previous_stats: CgroupStats | None) -> float | None:
    if (
        stats is None
        or previous_stats is None
        or stats.cpu_usage_usec is None
        or previous_stats.cpu_usage_usec is None
    ):
        return None
    elapsed = stats.timestamp - previous_stats.timestamp
    if elapsed <= 0:
        return None
    delta_usec = stats.cpu_usage_usec - previous_stats.cpu_usage_usec
    if delta_usec < 0:
        return None
    return (delta_usec / 1_000_000) / elapsed * 100


def _terminate_systemd_scope(unit_name: str, process: subprocess.Popen, *, use_sudo: bool) -> None:
    command = ["systemctl", "kill", "--kill-who=all", unit_name]
    if use_sudo:
        command.insert(0, "sudo")
    try:
        subprocess.run(command, check=False, capture_output=True, timeout=5)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _exceeds_memory_budget(sample: ChildSample, budget: ResourceBudget, psutil) -> bool:
    total_memory_mb = psutil.virtual_memory().total / (1024 * 1024) if psutil is not None else None
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
    cgroup_memory = [
        sample.cgroup_memory_current_mb for sample in timeline if sample.cgroup_memory_current_mb is not None
    ]
    cgroup_memory_peak = [sample.cgroup_memory_peak_mb for sample in timeline if sample.cgroup_memory_peak_mb is not None]
    cgroup_cpu = [sample.cgroup_cpu_percent for sample in timeline if sample.cgroup_cpu_percent is not None]
    total_memory_mb = None
    try:
        import psutil

        total_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
    except ModuleNotFoundError:
        pass
    effective_budget = budget.effective_memory_budget_mb(total_memory_mb)
    summary = {
        "samples": len(timeline),
        "peak_rss_mb": round(peak_rss, 3),
        "available_memory_after_mb": round(timeline[-1].available_memory_mb, 3),
        "average_child_cpu_percent": round(sum(child_cpu) / len(child_cpu), 3),
        "peak_child_cpu_percent": round(max(child_cpu), 3),
        "memory_budget_mb": round(budget.memory_budget_mb, 3) if budget.memory_budget_mb is not None else None,
        "effective_memory_budget_mb": round(effective_budget, 3) if effective_budget is not None else None,
        "memory_budget_exceeded": bool(effective_budget is not None and peak_rss > effective_budget),
    }
    if cgroup_memory:
        summary["peak_cgroup_memory_current_mb"] = round(max(cgroup_memory), 3)
    if cgroup_memory_peak:
        summary["peak_cgroup_memory_peak_mb"] = round(max(cgroup_memory_peak), 3)
    if cgroup_cpu:
        summary["average_cgroup_cpu_percent"] = round(sum(cgroup_cpu) / len(cgroup_cpu), 3)
        summary["peak_cgroup_cpu_percent"] = round(max(cgroup_cpu), 3)
    cgroup_paths = [sample.cgroup_path for sample in timeline if sample.cgroup_path]
    if cgroup_paths:
        summary["cgroup_path"] = cgroup_paths[-1]
    return summary


def _visible_memory_mb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        return None


def _visible_cpu_count() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=True)
    except Exception:
        return None
