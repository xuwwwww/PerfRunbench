from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

from training_workload import load_flat_config, write_training_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a generic CPU and memory pressure training-style workload.")
    parser.add_argument("--config", default="examples/heavy_training_pressure_config.yaml")
    args = parser.parse_args()

    config = load_flat_config(args.config)
    duration_seconds = float(config.get("duration_seconds", 60.0))
    cpu_workers = int(config.get("cpu_workers", max(1, (os.cpu_count() or 2) - 1)))
    memory_target_mb = int(config.get("memory_target_mb", 4096))
    batch_size = int(config.get("batch_size", 256))
    feature_count = int(config.get("feature_count", 4096))
    samples_per_step = int(config.get("samples_per_step", batch_size))
    steps_per_epoch = int(config.get("steps_per_epoch", 100))

    memory = _allocate_memory(memory_target_mb)
    deadline = time.perf_counter() + duration_seconds
    result_queue: mp.Queue = mp.Queue()
    workers = [
        mp.Process(target=_worker, args=(deadline, worker_id, feature_count, batch_size, result_queue))
        for worker_id in range(max(1, cpu_workers))
    ]
    start = time.perf_counter()
    for worker in workers:
        worker.start()
    worker_results = []
    for worker in workers:
        worker.join()
    while not result_queue.empty():
        worker_results.append(result_queue.get())
    elapsed = time.perf_counter() - start
    total_steps = sum(item["steps"] for item in worker_results)
    total_samples = total_steps * samples_per_step
    metrics = {
        "duration_seconds": round(elapsed, 6),
        "samples_per_second": round(total_samples / max(elapsed, 1e-9), 3),
        "step_time_mean_seconds": round((elapsed * max(1, cpu_workers)) / max(1, total_steps), 6),
        "epoch_time_mean_seconds": round((elapsed * steps_per_epoch * max(1, cpu_workers)) / max(1, total_steps), 6),
        "epoch_time_max_seconds": round(elapsed, 6),
        "peak_batch_payload_mb": round((batch_size * feature_count * 8) / (1024 * 1024), 6),
        "optimizer_steps": total_steps,
        "completed_epochs": round(total_steps / max(1, steps_per_epoch * max(1, cpu_workers)), 6),
        "feature_count": feature_count,
        "train_samples": total_samples,
        "cpu_workers": cpu_workers,
        "memory_target_mb": memory_target_mb,
        "memory_touched_mb": len(memory) // (1024 * 1024),
        "config_path": args.config,
        "dataset": "synthetic-heavy-pressure",
    }
    write_training_metrics(metrics)
    print(
        "heavy pressure training completed "
        f"duration={metrics['duration_seconds']} "
        f"cpu_workers={cpu_workers} "
        f"memory_target_mb={memory_target_mb} "
        f"samples_per_second={metrics['samples_per_second']}"
    )
    return 0


def _allocate_memory(memory_target_mb: int) -> bytearray:
    memory = bytearray(max(0, memory_target_mb) * 1024 * 1024)
    page = 4096
    for index in range(0, len(memory), page):
        memory[index] = (index // page) % 251
    return memory


def _worker(deadline: float, worker_id: int, feature_count: int, batch_size: int, result_queue: mp.Queue) -> None:
    steps = 0
    value = float(worker_id + 1)
    width = max(16, min(feature_count, 8192))
    while time.perf_counter() < deadline:
        for sample_index in range(batch_size):
            base = (sample_index + 1) / batch_size
            for feature_index in range(width):
                angle = (feature_index + 1) * base
                value += math.sin(angle) * math.cos(angle / (worker_id + 1))
        steps += 1
    result_queue.put({"worker_id": worker_id, "steps": steps, "checksum": value})


if __name__ == "__main__":
    raise SystemExit(main())
