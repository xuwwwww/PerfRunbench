from __future__ import annotations

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def collect_hardware_info() -> dict[str, Any]:
    from autotune.resource.executor_capabilities import collect_executor_capabilities

    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "is_wsl": is_wsl(),
    }
    info.update(collect_cpu_memory_info())
    info["packages"] = collect_package_info()
    info["runtime"] = collect_runtime_info()
    info["limits"] = collect_limit_info()
    info["executor_capabilities"] = collect_executor_capabilities()
    info["notes"] = generate_notes(info)
    return info


def collect_cpu_memory_info() -> dict[str, Any]:
    result = {
        "cpu_count_logical": None,
        "cpu_count_physical": None,
        "cpu_affinity_supported": False,
        "current_cpu_affinity": None,
        "total_memory_mb": None,
        "available_memory_mb": None,
    }
    try:
        import psutil

        result["cpu_count_logical"] = psutil.cpu_count(logical=True)
        result["cpu_count_physical"] = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        result["total_memory_mb"] = round(memory.total / (1024 * 1024), 3)
        result["available_memory_mb"] = round(memory.available / (1024 * 1024), 3)
        process = psutil.Process()
        if hasattr(process, "cpu_affinity"):
            try:
                result["current_cpu_affinity"] = process.cpu_affinity()
                result["cpu_affinity_supported"] = True
            except (AttributeError, NotImplementedError, OSError):
                result["cpu_affinity_supported"] = False
    except ModuleNotFoundError:
        pass
    return result


def collect_package_info() -> dict[str, Any]:
    packages = {}
    for name in ["torch", "torchvision", "onnx", "onnxscript", "onnxruntime", "psutil", "numpy"]:
        packages[name] = package_version(name)
    return packages


def collect_runtime_info() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "torch_cuda_available": None,
        "torch_num_threads": None,
        "torch_num_interop_threads": None,
        "onnxruntime_providers": None,
    }
    try:
        import torch

        runtime["torch_cuda_available"] = bool(torch.cuda.is_available())
        runtime["torch_num_threads"] = int(torch.get_num_threads())
        runtime["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:
        pass
    try:
        import onnxruntime as ort

        runtime["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception:
        pass
    return runtime


def collect_limit_info() -> dict[str, Any]:
    return {
        "cgroup_memory_max_mb": read_cgroup_memory_limit_mb(),
        "wsl_config_visible": Path("/mnt/c/Users/User/.wslconfig").exists() if is_wsl() else False,
        "systemd_run_available": shutil.which("systemd-run") is not None,
        "systemd_state": read_systemd_state(),
    }


def package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def is_wsl() -> bool:
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return True
    version_path = Path("/proc/version")
    if version_path.exists():
        return "microsoft" in version_path.read_text(encoding="utf-8", errors="ignore").lower()
    return False


def read_cgroup_memory_limit_mb() -> float | None:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8").strip()
        if raw in {"", "max"}:
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value > 1 << 60:
            return None
        return round(value / (1024 * 1024), 3)
    return None


def read_systemd_state() -> str | None:
    if shutil.which("systemctl") is None:
        return None
    try:
        result = subprocess.run(["systemctl", "is-system-running"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return None
    state = (result.stdout or result.stderr).strip()
    return state or None


def generate_notes(info: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    total_memory = info.get("total_memory_mb")
    cgroup_memory = info.get("limits", {}).get("cgroup_memory_max_mb")
    packages = info.get("packages", {})
    runtime = info.get("runtime", {})
    executor_capabilities = info.get("executor_capabilities", {})

    if info.get("is_wsl") and total_memory is not None:
        notes.append(f"WSL environment detected; Linux-visible RAM is {total_memory:.1f} MB.")
    if cgroup_memory is not None and total_memory is not None and cgroup_memory < total_memory:
        notes.append(f"cgroup memory limit is lower than visible RAM: {cgroup_memory:.1f} MB.")
    if info.get("limits", {}).get("systemd_run_available") is False:
        notes.append("systemd-run is not available; systemd hard-limit executor cannot be used.")
    if not info.get("cpu_affinity_supported"):
        notes.append("CPU affinity is not supported in this environment.")
    if packages.get("torch") is None:
        notes.append("PyTorch is not installed; real PyTorch benchmarks will not run.")
    if packages.get("onnxruntime") is None:
        notes.append("ONNX Runtime is not installed; ONNX Runtime benchmarks will not run.")
    providers = runtime.get("onnxruntime_providers")
    if providers is not None and "CPUExecutionProvider" not in providers:
        notes.append("ONNX Runtime CPUExecutionProvider is not available.")
    if runtime.get("torch_cuda_available") is False:
        notes.append("torch reports CUDA unavailable; GPU benchmarks should be skipped unless another provider is configured.")
    recommended_executor = executor_capabilities.get("recommended_executor")
    if recommended_executor == "systemd":
        notes.append("systemd is the recommended executor for hard memory/CPU limits on this machine.")
    elif recommended_executor == "local":
        notes.append("local is the recommended executor; hard memory/CPU limits may be unavailable on this machine.")
    return notes


def write_hardware_info(output: str | Path, info: dict[str, Any] | None = None) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = info if info is not None else collect_hardware_info()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
