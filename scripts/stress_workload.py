from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CPU and memory load for AutoTuneAI resource guard tests.")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--memory-mb", type=int, default=0, help="Total memory to allocate and touch in the parent process.")
    parser.add_argument("--chunk-mb", type=int, default=64, help="Allocation chunk size used to touch memory gradually.")
    args = parser.parse_args()

    payload = allocate_memory(args.memory_mb, args.chunk_mb)
    stop_at = time.monotonic() + args.duration_seconds
    processes = [mp.Process(target=burn_cpu, args=(stop_at, index)) for index in range(args.workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    checksum = sum(block[0] for block in payload) if payload else 0
    print(
        f"stress_complete workers={args.workers} duration_seconds={args.duration_seconds} "
        f"memory_mb={args.memory_mb} checksum={checksum}"
    )


def allocate_memory(memory_mb: int, chunk_mb: int) -> list[bytearray]:
    if memory_mb <= 0:
        return []
    chunk_mb = max(1, chunk_mb)
    remaining = memory_mb
    payload: list[bytearray] = []
    while remaining > 0:
        current = min(chunk_mb, remaining)
        block = bytearray(current * 1024 * 1024)
        stride = 4096
        for offset in range(0, len(block), stride):
            block[offset] = 1
        payload.append(block)
        remaining -= current
    return payload


def burn_cpu(stop_at: float, worker_index: int) -> None:
    value = worker_index + 1
    while time.monotonic() < stop_at:
        value = (value * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    if value == -1:
        print(value)


if __name__ == "__main__":
    main()
