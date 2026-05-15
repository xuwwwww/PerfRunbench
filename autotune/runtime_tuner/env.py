from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from autotune.resource.budget import ResourceBudget


class RuntimeEnvTuningError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeEnvPlan:
    profile: str
    env: dict[str, str]
    notes: list[str]


PROFILES = {
    "runtime-cpu-performance",
    "runtime-pytorch-gpu-performance",
    "runtime-pytorch-max-performance",
    "runtime-pytorch-aggressive",
}


def available_runtime_profiles() -> list[str]:
    return sorted(PROFILES)


def recommend_runtime_env(
    profile: str,
    budget: ResourceBudget | None = None,
    *,
    total_cores: int | None = None,
) -> dict[str, Any]:
    plan = build_runtime_env_plan(profile, budget or ResourceBudget(), total_cores=total_cores)
    return {
        "profile": plan.profile,
        "env": plan.env,
        "notes": plan.notes,
    }


def build_runtime_env_plan(
    profile: str,
    budget: ResourceBudget,
    *,
    total_cores: int | None = None,
) -> RuntimeEnvPlan:
    if profile not in PROFILES:
        raise RuntimeEnvTuningError(f"unknown runtime env profile: {profile}")
    visible_cores = total_cores or (os.cpu_count() or 1)
    allowed_threads = budget.allowed_threads(visible_cores) or visible_cores
    if profile == "runtime-cpu-performance":
        env = _cpu_env(allowed_threads)
        notes = [
            f"runtime_env_profile={profile}",
            f"runtime_env_allowed_threads={allowed_threads}",
            "runtime env sets OpenMP/BLAS thread counts for CPU-heavy PyTorch/NumPy/BLAS workloads.",
        ]
        return RuntimeEnvPlan(profile, env, notes)
    if profile == "runtime-pytorch-gpu-performance":
        env = {
            **_gpu_loader_env(max(1, min(4, allowed_threads))),
            **_pytorch_cuda_env(),
        }
        notes = [
            f"runtime_env_profile={profile}",
            f"runtime_env_cpu_helper_threads={env['OMP_NUM_THREADS']}",
            "runtime env favors GPU training by limiting CPU thread oversubscription and enabling CUDA/PyTorch throughput knobs.",
        ]
        return RuntimeEnvPlan(profile, env, notes)
    if profile == "runtime-pytorch-max-performance":
        env = {
            **_cpu_env(allowed_threads),
            **_pytorch_cuda_env(),
            "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "0",
        }
        notes = [
            f"runtime_env_profile={profile}",
            f"runtime_env_allowed_threads={allowed_threads}",
            "runtime env combines CPU throughput and PyTorch CUDA throughput settings; benchmark before keeping it.",
        ]
        return RuntimeEnvPlan(profile, env, notes)
    if profile == "runtime-pytorch-aggressive":
        helper_threads = max(1, min(4, allowed_threads))
        env = {
            **_gpu_loader_env(helper_threads),
            **_pytorch_cuda_env(),
            **_pytorch_throughput_unstable_env(),
            "TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "0",
        }
        notes = [
            f"runtime_env_profile={profile}",
            f"runtime_env_cpu_helper_threads={env['OMP_NUM_THREADS']}",
            "runtime env trades reproducibility and allocator headroom for throughput: cuDNN autotune, looser GC threshold, more concurrent CUDA connections, no strict OMP affinity.",
        ]
        return RuntimeEnvPlan(profile, env, notes)
    raise RuntimeEnvTuningError(f"unhandled runtime env profile: {profile}")


def apply_runtime_env_profile(
    env: dict[str, str],
    profile: str | None,
    budget: ResourceBudget,
    *,
    total_cores: int | None = None,
) -> dict[str, Any] | None:
    if profile is None:
        return None
    plan = build_runtime_env_plan(profile, budget, total_cores=total_cores)
    before = {key: env.get(key) for key in plan.env}
    env.update(plan.env)
    return {
        "profile": plan.profile,
        "env": plan.env,
        "before": before,
        "notes": plan.notes,
    }


def _cpu_env(threads: int) -> dict[str, str]:
    value = str(max(1, threads))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",
        "MALLOC_ARENA_MAX": "2",
    }


def _gpu_loader_env(cpu_threads: int) -> dict[str, str]:
    value = str(max(1, cpu_threads))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "MALLOC_ARENA_MAX": "2",
    }


def _pytorch_cuda_env() -> dict[str, str]:
    return {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.9",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
        "TORCH_NCCL_USE_COMM_NONBLOCKING": "1",
    }


def _pytorch_throughput_unstable_env() -> dict[str, str]:
    """Extra knobs that often improve throughput at the cost of determinism or VRAM headroom."""
    return {
        # Overrides PYTORCH_CUDA_ALLOC_CONF from _pytorch_cuda_env when merged later in the dict.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.96",
        "CUDNN_BENCHMARK": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    }
