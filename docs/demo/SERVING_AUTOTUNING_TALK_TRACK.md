# Serving Autotuning Talk Track

## Opening

PerfRunbench is not just a benchmark launcher. It inspects the target, records
provenance, searches bounded candidates, applies statistical guards, and
packages a recommendation only when the measured protocol supports it.

For v0.3, the product focus is workload-aware `llama-server` serving
autotuning. The workload adapter is `llama.cpp`; the product identity is the
guarded benchmarking and recommendation workflow.

## Why Serving Was Different

The fixed single-stream `llama-bench` protocol did not produce a new
compiler, kernel, allocator, THP, HugeTLB, storage, or page-size
recommendation outside the guard. PerfRunbench preserved those negative and
inconclusive results instead of turning noise into a claim.

The remote serving workload exposed a different bottleneck: concurrent
requests needed the server to keep multiple slots active. `slots4-cont-on`
matched that workload shape.

## What Was Validated

The real validation used:

```text
C7g / Neoverse-V1 server
T4g / Neoverse-N1 class remote client
direct private network path
Qwen2.5-7B Q4_K_M
pinned llama-server binary
concurrency 4
fresh server restart between paired blocks
zero retries
zero failures
quality guard passed
artifact validation passed
dual-host cleanup passed
```

## Main Result

Under the J1 full-eight-CPU remote topology:

```text
interactive-short: 17.032 -> 31.355 output tok/s, +84.092%
decode-heavy: 17.046 -> 31.790 output tok/s, +86.490%
prefill-heavy: 16.563 -> 28.054 output tok/s, +69.375%
```

p95 TTFT also dropped by more than 91% for all three workloads.

## Guard Message

The important product behavior is not only the speedup. The important behavior
is that PerfRunbench knows when not to apply it:

- loopback topology is hard drift
- different workload is hard drift
- different concurrency is hard drift
- different binary, model, capability digest, CPU count, or CPU assignment is
  hard drift
- no network tuning recommendation is created without bottleneck evidence
- `run --best` never retunes

## Close

The recommendation is:

```text
Best observed eligible serving configuration under the tested C7g server,
T4g remote-client topology, Qwen2.5-7B Q4_K_M model, pinned llama-server
binary, workload, concurrency, and measurement protocol.
```

It is not a universal `llama-server` setting and not a cloud-wide Graviton
claim.

