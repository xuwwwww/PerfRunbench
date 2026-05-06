from __future__ import annotations

import argparse
import time

from training_workload import load_flat_config, summarize_step_latencies, write_training_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CUDA GPU pressure training-style workload.")
    parser.add_argument("--config", default="examples/gpu_training_pressure_config.yaml")
    args = parser.parse_args()

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for the GPU pressure workload.") from exc

    config = load_flat_config(args.config)
    require_cuda = bool(config.get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this benchmark, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or run in the WSL/server env with GPU access."
        )

    device_index = 0
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GPU pressure workload refuses to run on CPU because it would invalidate the benchmark.")

    dtype = _dtype(torch, str(config.get("dtype", "float16")))
    duration_seconds = float(config.get("duration_seconds", 60.0))
    matrix_size = int(config.get("matrix_size", 4096))
    batch_matrices = int(config.get("batch_matrices", 3))
    gpu_memory_target_mb = int(config.get("gpu_memory_target_mb", 3072))
    warmup_seconds = float(config.get("warmup_seconds", 3.0))
    latency_sample_limit = int(config.get("latency_sample_limit", 128))
    use_tf32 = bool(config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
    memory_blocks = _allocate_gpu_memory(torch, device, dtype, gpu_memory_target_mb)
    matrices = [
        torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
        for _ in range(max(2, batch_matrices))
    ]
    accumulator = torch.zeros((matrix_size, matrix_size), device=device, dtype=dtype)

    _run_matmul_loop(torch, matrices, accumulator, warmup_seconds)
    torch.cuda.synchronize(device_index)
    start = time.perf_counter()
    steps, latency_events = _run_matmul_loop(
        torch,
        matrices,
        accumulator,
        duration_seconds,
        latency_sample_limit=latency_sample_limit,
    )
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - start
    step_latencies = [
        start_event.elapsed_time(end_event) / 1000.0
        for start_event, end_event in latency_events
    ]

    # Touch memory blocks so allocation cannot be optimized away by lazy kernels.
    checksum = float(accumulator.float().mean().item())
    checksum += sum(float(block[:1].float().sum().item()) for block in memory_blocks[:4])
    flops_per_matmul = 2.0 * matrix_size * matrix_size * matrix_size
    matmuls = steps * max(1, len(matrices) - 1)
    metrics = {
        "duration_seconds": round(elapsed, 6),
        "samples_per_second": round(matmuls / max(elapsed, 1e-9), 3),
        "gpu_matmuls_per_second": round(matmuls / max(elapsed, 1e-9), 3),
        "gpu_tflops_estimate": round((matmuls * flops_per_matmul) / max(elapsed, 1e-9) / 1e12, 3),
        "step_time_mean_seconds": round(elapsed / max(1, steps), 6),
        **summarize_step_latencies(step_latencies),
        "epoch_time_mean_seconds": round(elapsed, 6),
        "epoch_time_max_seconds": round(elapsed, 6),
        "optimizer_steps": steps,
        "completed_epochs": 1,
        "feature_count": matrix_size,
        "train_samples": matmuls,
        "peak_batch_payload_mb": round((matrix_size * matrix_size * _dtype_size(dtype) * len(matrices)) / (1024 * 1024), 3),
        "gpu_memory_target_mb": gpu_memory_target_mb,
        "gpu_peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device_index) / (1024 * 1024), 3),
        "gpu_peak_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device_index) / (1024 * 1024), 3),
        "device": torch.cuda.get_device_name(device_index),
        "cuda_version": torch.version.cuda,
        "dtype": str(dtype).replace("torch.", ""),
        "allow_tf32": use_tf32,
        "checksum": round(checksum, 6),
        "dataset": "synthetic-gpu-pressure",
        "config_path": args.config,
    }
    write_training_metrics(metrics)
    print(
        "gpu pressure training completed "
        f"device={metrics['device']} "
        f"duration={metrics['duration_seconds']} "
        f"gpu_peak_memory_allocated_mb={metrics['gpu_peak_memory_allocated_mb']} "
        f"gpu_tflops_estimate={metrics['gpu_tflops_estimate']}"
    )
    return 0


def _run_matmul_loop(torch, matrices, accumulator, duration_seconds: float, *, latency_sample_limit: int = 0):
    deadline = time.perf_counter() + max(0.0, duration_seconds)
    steps = 0
    latency_events = []
    while time.perf_counter() < deadline:
        start_event = None
        end_event = None
        if len(latency_events) < max(0, latency_sample_limit):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        value = matrices[0]
        for other in matrices[1:]:
            value = torch.matmul(value, other)
        accumulator.mul_(0.95).add_(value, alpha=0.05)
        if start_event is not None and end_event is not None:
            end_event.record()
            latency_events.append((start_event, end_event))
        steps += 1
    return steps, latency_events


def _allocate_gpu_memory(torch, device, dtype, target_mb: int):
    blocks = []
    if target_mb <= 0:
        return blocks
    chunk_mb = 256
    elements_per_mb = (1024 * 1024) // _dtype_size(dtype)
    remaining = target_mb
    while remaining > 0:
        current_mb = min(chunk_mb, remaining)
        block = torch.empty((current_mb * elements_per_mb,), device=device, dtype=dtype)
        block.fill_(0.125)
        blocks.append(block)
        remaining -= current_mb
    return blocks


def _dtype(torch, name: str):
    normalized = name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise SystemExit(f"unsupported dtype for GPU pressure workload: {name}")


def _dtype_size(dtype) -> int:
    text = str(dtype)
    if "float16" in text or "bfloat16" in text:
        return 2
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
