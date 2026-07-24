from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from autotune import __version__ as perfrunbench_version
from autotune.golden_path.config import PRESET_DEFAULTS, PUBLIC_PRESETS


SCHEMA_VERSION = "1.0"
VALIDATION_STATES = {
    "compatibility_verified",
    "performance_improvement_verified",
    "fixture_only",
    "unavailable",
    "failed",
}


def add_platform_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    platform_parser = subparsers.add_parser("platform", help="Inspect and validate portable local runtime compatibility.")
    platform_subparsers = platform_parser.add_subparsers(dest="platform_command")
    validate = platform_subparsers.add_parser("validate", help="Run bounded local compatibility validation.")
    validate.add_argument("--preset", choices=PUBLIC_PRESETS, default="quick")
    validate.add_argument("--output", default="validation.json")
    validate.set_defaults(handler=_cmd_platform_validate)


def validate_platform(*, preset: str = "quick") -> dict[str, Any]:
    if preset not in PUBLIC_PRESETS:
        raise ValueError(f"unsupported platform validation preset: {preset}")
    preset_budget_minutes = float(PRESET_DEFAULTS[preset]["budget_minutes"])
    host = _host_record()
    package = _package_record()
    nvidia = _detect_nvidia()
    validations = [_generic_command_smoke()]

    torch_probe = _probe_torch()
    if torch_probe["availability"] == "available":
        validations.append(_torch_cpu_smoke(torch_probe))
        if torch_probe.get("cuda_available") is True:
            validations.append(_torch_cuda_smoke(torch_probe))
        else:
            validations.append(_unavailable_validation("pytorch", "NVIDIA CUDA", "CUDA is not usable in the installed PyTorch runtime."))
    elif torch_probe["availability"] == "failed":
        validations.append(_failed_validation("pytorch", "CPU", str(torch_probe.get("error") or "PyTorch import failed.")))
        validations.append(_unavailable_validation("pytorch", "NVIDIA CUDA", "PyTorch CUDA validation requires a usable PyTorch install."))
    else:
        validations.append(_unavailable_validation("pytorch", "CPU", "PyTorch is not installed."))
        validations.append(_unavailable_validation("pytorch", "NVIDIA CUDA", "PyTorch is not installed."))

    onnx_probe = _probe_onnxruntime()
    if onnx_probe["availability"] == "available":
        validations.append(_onnxruntime_smoke(onnx_probe))
    elif onnx_probe["availability"] == "failed":
        validations.append(_failed_validation("onnxruntime", "CPU", str(onnx_probe.get("error") or "ONNX Runtime import failed.")))
    else:
        validations.append(_unavailable_validation("onnxruntime", "CPU", "ONNX Runtime is not installed."))

    digest_payload = {
        "host": host,
        "package": package,
        "nvidia": nvidia,
        "runtimes": [
            {
                "runtime": item["runtime"],
                "accelerator": item["accelerator"],
                "compatibility_status": item["compatibility_status"],
                "performance_status": item["performance_status"],
                "runtime_version": item.get("runtime_version", "unknown"),
            }
            for item in validations
        ],
    }
    failed = any(item["compatibility_status"] == "failed" for item in validations)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "portable_platform_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": preset,
        "preset_budget_minutes": preset_budget_minutes,
        "overall_status": "failed" if failed else "passed",
        "host": host,
        "nvidia_gpu": nvidia,
        "package": package,
        "validations": validations,
        "environment_digest": _stable_digest(digest_payload),
        "status_vocabulary": sorted(VALIDATION_STATES),
        "privacy": {
            "hostname_collected": False,
            "username_collected": False,
            "machine_identifier_collected": False,
            "absolute_paths_included": False,
        },
    }


def write_platform_validation(payload: dict[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _cmd_platform_validate(args: argparse.Namespace) -> int:
    payload = validate_platform(preset=args.preset)
    output = write_platform_validation(payload, args.output)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": payload["overall_status"],
                "preset": payload["preset"],
                "output": _display_output_path(output),
                "environment_digest": payload["environment_digest"],
                "validations": [
                    {
                        "runtime": item["runtime"],
                        "accelerator": item["accelerator"],
                        "compatibility_status": item["compatibility_status"],
                        "performance_status": item["performance_status"],
                    }
                    for item in payload["validations"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_status"] == "passed" else 1


def _host_record() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "os": platform.system() or "unknown",
        "os_release": platform.release() or "unknown",
        "architecture": platform.machine() or "unknown",
        "logical_cpu_count": max(1, os.cpu_count() or 1),
        "memory_total_bytes": int(memory.total),
        "memory_available_bytes": int(memory.available),
    }


def _package_record() -> dict[str, Any]:
    version = str(perfrunbench_version or "unknown")
    if version == "unknown":
        try:
            version = importlib.metadata.version("perfrunbench")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    source_commit, source_dirty = _source_state()
    return {
        "name": "perfrunbench",
        "version": version,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
    }


def _source_state() -> tuple[str, bool | str]:
    package_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(package_root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown", "unknown"
    if commit.returncode != 0:
        return "unknown", "unknown"
    return (commit.stdout or "").strip() or "unknown", bool((status.stdout or "").strip()) if status.returncode == 0 else "unknown"


def _detect_nvidia() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable:
        completed = _run_probe(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            timeout=5,
        )
        names = [line.strip() for line in str(completed.get("stdout") or "").splitlines() if line.strip()]
        if completed["return_code"] == 0 and names:
            return {"detected": True, "detection_source": "nvidia-smi", "device_names": names}
    if platform.system() == "Linux":
        devices = []
        for vendor in Path("/sys/bus/pci/devices").glob("*/vendor"):
            try:
                if vendor.read_text(encoding="ascii").strip().lower() == "0x10de":
                    devices.append("NVIDIA PCI device")
            except OSError:
                continue
        if devices:
            return {"detected": True, "detection_source": "linux-pci-sysfs", "device_names": devices}
    if platform.system() == "Windows":
        shell = shutil.which("powershell") or shutil.which("powershell.exe")
        if shell:
            completed = _run_probe(
                [
                    shell,
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                timeout=8,
            )
            names = [
                line.strip()
                for line in str(completed.get("stdout") or "").splitlines()
                if "nvidia" in line.lower()
            ]
            if completed["return_code"] == 0 and names:
                return {"detected": True, "detection_source": "windows-video-controller", "device_names": names}
    return {"detected": False, "detection_source": "bounded_local_probe", "device_names": []}


def _generic_command_smoke() -> dict[str, Any]:
    code = "import json; print(json.dumps({'status':'ok','value':sum(range(128))}))"
    completed = _run_probe([sys.executable, "-c", code], timeout=10)
    passed = completed["return_code"] == 0 and '"status": "ok"' in str(completed.get("stdout") or "")
    return _validation_record(
        runtime="generic-command",
        accelerator="CPU",
        compatibility_status="compatibility_verified" if passed else "failed",
        runtime_version=platform.python_version(),
        details={"probe": "bounded_subprocess_and_json_output", "return_code": completed["return_code"]},
        error=None if passed else completed.get("error"),
    )


def _probe_torch() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"availability": "unavailable"}
    code = (
        "import json, torch; "
        "print(json.dumps({'version':str(torch.__version__),"
        "'cuda_available':bool(torch.cuda.is_available()),"
        "'cuda_device_count':int(torch.cuda.device_count()) if torch.cuda.is_available() else 0}))"
    )
    completed = _run_probe([sys.executable, "-c", code], timeout=20)
    payload = _last_json_object(completed.get("stdout"))
    if completed["return_code"] != 0 or payload is None:
        return {"availability": "failed", "error": completed.get("error") or "PyTorch probe returned no JSON."}
    return {"availability": "available", **payload}


def _torch_cpu_smoke(probe: dict[str, Any]) -> dict[str, Any]:
    code = (
        "import json, math, torch; "
        "x=torch.arange(1024,dtype=torch.float32).reshape(32,32); "
        "value=float((x@x.T).sum().item()); "
        "print(json.dumps({'status':'ok' if math.isfinite(value) else 'failed'}))"
    )
    completed = _run_probe([sys.executable, "-c", code], timeout=30)
    payload = _last_json_object(completed.get("stdout"))
    passed = completed["return_code"] == 0 and payload is not None and payload.get("status") == "ok"
    return _validation_record(
        runtime="pytorch",
        accelerator="CPU",
        compatibility_status="compatibility_verified" if passed else "failed",
        runtime_version=str(probe.get("version") or "unknown"),
        details={"probe": "small_cpu_tensor_matmul", "return_code": completed["return_code"]},
        error=None if passed else completed.get("error"),
    )


def _torch_cuda_smoke(probe: dict[str, Any]) -> dict[str, Any]:
    code = (
        "import json, math, torch; "
        "x=torch.arange(1024,dtype=torch.float32,device='cuda').reshape(32,32); "
        "value=float((x@x.T).sum().item()); torch.cuda.synchronize(); "
        "print(json.dumps({'status':'ok' if math.isfinite(value) else 'failed'}))"
    )
    completed = _run_probe([sys.executable, "-c", code], timeout=30)
    payload = _last_json_object(completed.get("stdout"))
    passed = completed["return_code"] == 0 and payload is not None and payload.get("status") == "ok"
    return _validation_record(
        runtime="pytorch",
        accelerator="NVIDIA CUDA",
        compatibility_status="compatibility_verified" if passed else "failed",
        runtime_version=str(probe.get("version") or "unknown"),
        details={
            "probe": "small_cuda_tensor_matmul",
            "return_code": completed["return_code"],
            "cuda_device_count": int(probe.get("cuda_device_count") or 0),
        },
        error=None if passed else completed.get("error"),
    )


def _probe_onnxruntime() -> dict[str, Any]:
    if importlib.util.find_spec("onnxruntime") is None:
        return {"availability": "unavailable"}
    code = (
        "import json, onnxruntime as ort; "
        "print(json.dumps({'version':str(ort.__version__),'providers':list(ort.get_available_providers())}))"
    )
    completed = _run_probe([sys.executable, "-c", code], timeout=20)
    payload = _last_json_object(completed.get("stdout"))
    if completed["return_code"] != 0 or payload is None:
        return {"availability": "failed", "error": completed.get("error") or "ONNX Runtime probe returned no JSON."}
    return {"availability": "available", **payload}


def _onnxruntime_smoke(probe: dict[str, Any]) -> dict[str, Any]:
    code = (
        "import json, numpy as np, onnxruntime as ort; "
        "value=ort.OrtValue.ortvalue_from_numpy(np.asarray([1.0,2.0],dtype=np.float32)); "
        "print(json.dumps({'status':'ok' if value.shape()==[2] else 'failed'}))"
    )
    completed = _run_probe([sys.executable, "-c", code], timeout=20)
    payload = _last_json_object(completed.get("stdout"))
    passed = completed["return_code"] == 0 and payload is not None and payload.get("status") == "ok"
    return _validation_record(
        runtime="onnxruntime",
        accelerator="CPU",
        compatibility_status="compatibility_verified" if passed else "failed",
        runtime_version=str(probe.get("version") or "unknown"),
        details={
            "probe": "runtime_import_provider_and_tensor_boundary",
            "return_code": completed["return_code"],
            "providers": [str(item) for item in probe.get("providers", [])],
        },
        error=None if passed else completed.get("error"),
    )


def _validation_record(
    *,
    runtime: str,
    accelerator: str,
    compatibility_status: str,
    runtime_version: str,
    details: dict[str, Any],
    error: Any = None,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "accelerator": accelerator,
        "validation_type": "compatibility_smoke",
        "compatibility_status": compatibility_status,
        "performance_status": "unavailable",
        "evidence_origin": "local_bounded_validation",
        "runtime_version": runtime_version,
        "details": details,
        "claim_boundary": "Compatibility only; no performance improvement is claimed.",
    }
    if error:
        record["error"] = _sanitize_text(str(error))
    return record


def _unavailable_validation(runtime: str, accelerator: str, reason: str) -> dict[str, Any]:
    return _validation_record(
        runtime=runtime,
        accelerator=accelerator,
        compatibility_status="unavailable",
        runtime_version="unavailable",
        details={"reason": reason},
    )


def _failed_validation(runtime: str, accelerator: str, reason: str) -> dict[str, Any]:
    return _validation_record(
        runtime=runtime,
        accelerator=accelerator,
        compatibility_status="failed",
        runtime_version="unknown",
        details={"reason": _sanitize_text(reason)},
    )


def _run_probe(command: list[str], *, timeout: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"return_code": 124, "stdout": "", "error": "probe timed out"}
    except OSError as exc:
        return {"return_code": 1, "stdout": "", "error": _sanitize_text(str(exc))}
    error = _sanitize_text((completed.stderr or "").strip()) if completed.returncode != 0 else None
    return {"return_code": completed.returncode, "stdout": completed.stdout or "", "error": error}


def _last_json_object(raw: Any) -> dict[str, Any] | None:
    lines = str(raw or "").strip().splitlines()
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _sanitize_text(value: str) -> str:
    sanitized = value.replace(str(Path.home()), "<home>").replace(str(Path.cwd()), "<cwd>")
    sanitized = re.sub(r"(?i)(?:[a-z]:\\|/)(?:[^ \r\n\t:]+[\\/])+[^ \r\n\t:]*", "<path>", sanitized)
    return sanitized[:500]


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_output_path(path: Path) -> str:
    return path.name if path.is_absolute() else path.as_posix()
