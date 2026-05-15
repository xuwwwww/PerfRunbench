from __future__ import annotations

import argparse
import time

from training_workload import load_flat_config, summarize_step_latencies, write_training_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real timm model inference benchmark on CUDA.")
    parser.add_argument("--config", default="examples/timm_inference_benchmark_config.yaml")
    args = parser.parse_args()

    try:
        import timm
    except ModuleNotFoundError as exc:
        raise SystemExit("timm is required for this benchmark. Install it in the benchmark environment.") from exc
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for the timm benchmark.") from exc

    config = load_flat_config(args.config)
    require_cuda = bool(config.get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this benchmark, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or run in the server env with GPU access."
        )

    device_index = int(config.get("device_index", 0))
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("timm inference benchmark refuses to run on CPU because it would invalidate the comparison.")

    model_name = str(config.get("model_name", "resnet50"))
    batch_size = int(config.get("batch_size", 256))
    image_size = int(config.get("image_size", 224))
    channels = int(config.get("channels", 3))
    duration_seconds = float(config.get("duration_seconds", 300.0))
    warmup_seconds = float(config.get("warmup_seconds", 10.0))
    latency_sample_limit = int(config.get("latency_sample_limit", 512))
    use_tf32 = bool(config.get("allow_tf32", True))
    use_channels_last = bool(config.get("channels_last", True))
    amp_dtype_name = str(config.get("amp_dtype", "float16"))
    amp_dtype = _amp_dtype(torch, amp_dtype_name)
    extra_gpu_memory_target_mb = int(config.get("extra_gpu_memory_target_mb", 0))
    loader_workers = int(config.get("loader_workers", 4))
    pin_memory = bool(config.get("pin_memory", True))
    prefetch_factor = int(config.get("prefetch_factor", 4))
    persistent_workers = bool(config.get("persistent_workers", True))
    synthetic_dataset_size = int(config.get("synthetic_dataset_size", max(batch_size * 64, 4096)))
    use_torch_compile = bool(config.get("torch_compile", False))
    compile_mode = str(config.get("compile_mode", "reduce-overhead"))

    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.cuda.set_device(device_index)
    torch.cuda.reset_peak_memory_stats(device_index)

    model = timm.create_model(model_name, pretrained=False).to(device)
    model.eval()
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    if use_torch_compile:
        if not hasattr(torch, "compile"):
            raise SystemExit("torch.compile was requested, but this PyTorch build does not expose torch.compile.")
        model = torch.compile(model, mode=compile_mode)

    loader = _build_loader(
        torch,
        batch_size=batch_size,
        channels=channels,
        image_size=image_size,
        synthetic_dataset_size=synthetic_dataset_size,
        loader_workers=loader_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    extra_blocks = _allocate_gpu_memory(torch, device, extra_gpu_memory_target_mb)

    _run_loop(
        torch,
        model,
        loader,
        device,
        warmup_seconds,
        amp_dtype=amp_dtype,
        use_channels_last=use_channels_last,
        pin_memory=pin_memory,
    )
    torch.cuda.synchronize(device_index)
    start = time.perf_counter()
    steps, sample_count, latency_events, output_checksum = _run_loop(
        torch,
        model,
        loader,
        device,
        duration_seconds,
        amp_dtype=amp_dtype,
        latency_sample_limit=latency_sample_limit,
        return_checksum=True,
        use_channels_last=use_channels_last,
        pin_memory=pin_memory,
    )
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - start
    step_latencies = [
        start_event.elapsed_time(end_event) / 1000.0
        for start_event, end_event in latency_events
    ]

    metrics = {
        "duration_seconds": round(elapsed, 6),
        "samples_per_second": round(sample_count / max(elapsed, 1e-9), 3),
        "step_time_mean_seconds": round(elapsed / max(1, steps), 6),
        **summarize_step_latencies(step_latencies),
        "epoch_time_mean_seconds": round(elapsed, 6),
        "epoch_time_max_seconds": round(elapsed, 6),
        "optimizer_steps": steps,
        "completed_epochs": 1,
        "feature_count": image_size * image_size * channels,
        "train_samples": sample_count,
        "peak_batch_payload_mb": round((batch_size * channels * image_size * image_size * 4) / (1024 * 1024), 3),
        "gpu_peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device_index) / (1024 * 1024), 3),
        "gpu_peak_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device_index) / (1024 * 1024), 3),
        "device": torch.cuda.get_device_name(device_index),
        "cuda_version": torch.version.cuda,
        "model_name": model_name,
        "batch_size": batch_size,
        "image_size": image_size,
        "amp_dtype": amp_dtype_name,
        "allow_tf32": use_tf32,
        "channels_last": use_channels_last,
        "loader_workers": loader_workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor if loader_workers > 0 else None,
        "persistent_workers": persistent_workers if loader_workers > 0 else False,
        "synthetic_dataset_size": synthetic_dataset_size,
        "torch_compile": use_torch_compile,
        "compile_mode": compile_mode if use_torch_compile else None,
        "extra_gpu_memory_target_mb": extra_gpu_memory_target_mb,
        "checksum": round(output_checksum + _extra_checksum(extra_blocks), 6),
        "dataset": "timm-synthetic-inference",
        "config_path": args.config,
    }
    write_training_metrics(metrics)
    print(
        "timm inference benchmark completed "
        f"model={metrics['model_name']} "
        f"device={metrics['device']} "
        f"duration={metrics['duration_seconds']} "
        f"samples_per_second={metrics['samples_per_second']} "
        f"gpu_peak_memory_allocated_mb={metrics['gpu_peak_memory_allocated_mb']}"
    )
    return 0


def _run_loop(
    torch,
    model,
    loader,
    device,
    duration_seconds: float,
    *,
    amp_dtype,
    latency_sample_limit: int = 0,
    return_checksum: bool = False,
    use_channels_last: bool = False,
    pin_memory: bool = False,
):
    deadline = time.perf_counter() + max(0.0, duration_seconds)
    steps = 0
    sample_count = 0
    latency_events = []
    checksum = 0.0
    iterator = iter(loader)
    with torch.inference_mode():
        while time.perf_counter() < deadline:
            try:
                inputs, _labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                inputs, _labels = next(iterator)
            inputs = inputs.to(device, non_blocking=pin_memory)
            if use_channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            start_event = None
            end_event = None
            if len(latency_events) < max(0, latency_sample_limit):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(inputs)
            if start_event is not None and end_event is not None:
                end_event.record()
                latency_events.append((start_event, end_event))
            if return_checksum:
                checksum += float(outputs.float().mean().item())
            steps += 1
            sample_count += int(inputs.shape[0])
    return steps, sample_count, latency_events, checksum


def _build_loader(
    torch,
    *,
    batch_size: int,
    channels: int,
    image_size: int,
    synthetic_dataset_size: int,
    loader_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
):
    dataset = _SyntheticImageDataset(
        torch,
        sample_count=synthetic_dataset_size,
        channels=channels,
        image_size=image_size,
    )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": True,
        "num_workers": max(0, loader_workers),
        "pin_memory": pin_memory,
    }
    if loader_workers > 0:
        kwargs["prefetch_factor"] = max(2, prefetch_factor)
        kwargs["persistent_workers"] = persistent_workers
    return torch.utils.data.DataLoader(dataset, **kwargs)


class _SyntheticImageDataset:
    def __init__(self, torch, *, sample_count: int, channels: int, image_size: int) -> None:
        self._torch = torch
        self._sample_count = max(1, sample_count)
        self._shape = (channels, image_size, image_size)

    def __len__(self) -> int:
        return self._sample_count

    def __getitem__(self, index: int):
        generator = self._torch.Generator()
        generator.manual_seed(index)
        sample = self._torch.randn(self._shape, generator=generator, dtype=self._torch.float32)
        label = int(index % 1000)
        return sample, label


def _allocate_gpu_memory(torch, device, target_mb: int):
    blocks = []
    if target_mb <= 0:
        return blocks
    elements_per_mb = (1024 * 1024) // 2
    remaining = target_mb
    while remaining > 0:
        current_mb = min(256, remaining)
        block = torch.empty((current_mb * elements_per_mb,), device=device, dtype=torch.float16)
        block.fill_(0.03125)
        blocks.append(block)
        remaining -= current_mb
    return blocks


def _extra_checksum(blocks) -> float:
    if not blocks:
        return 0.0
    return sum(float(block[:1].float().sum().item()) for block in blocks[:4])


def _amp_dtype(torch, name: str):
    normalized = name.lower()
    if normalized in {"none", "fp32", "float32"}:
        return None
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise SystemExit(f"unsupported amp dtype for timm benchmark: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
