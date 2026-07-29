# C7g Size-Scaling and Guarded Serving Validation

## Status

```text
result_origin: measured
status: passed
targets: c7g.xlarge, c7g.2xlarge, c7g.4xlarge
performance_passport_status: eligible_preliminary
cleanup_passed_on_all_targets: true
host_restored_on_all_targets: true
controller_artifact_failures: 0
```

This validation compares three AWS Graviton3 C7g sizes under one pinned model,
runtime bundle, source revision, binary identity, and measurement protocol.
It provides cross-size scaling evidence and a workload-aware runtime-tuning
result. It is not a universal speedup claim.

## Fixed Identity

| Item | Fixed value |
| --- | --- |
| PerfRunbench source | `93b179a1c8a2370c1d6d3042909054af7ccfe666` |
| llama.cpp source | `b15ca938ad00aa6b3ee6c2edda7363fd02826b18` |
| Model | Qwen2.5-7B Instruct Q4_K_M |
| Model SHA-256 | `1875fb29e8c91c86615c00e92d8b4114e56bc24359adb5a8db8b36452fae4a49` |
| llama-bench SHA-256 | `6d284a756e384dc5a3cacda20ca53171d5fc937aad837c72460fcdf7a8747320` |
| llama-server SHA-256 | `3378e0dce6f44c47aca983b792335e7482ec840890772041145d374ec26114c5` |
| Runtime bundle SHA-256 | `597a6e5c8f63223d1e23442bd4704ff0b89847a8b8cceb03ab0afdea7f395931` |

No model, source, binary, runtime, or protocol identity changed between target
sizes.

## Targets

One logical CPU was reserved for the local load generator. All remaining
logical CPUs were assigned to the server.

| Target | Logical CPUs | Memory | Client CPUs | Maximum server threads |
| --- | ---: | ---: | ---: | ---: |
| `c7g.xlarge` | 4 | 8 GiB | 1 | 3 |
| `c7g.2xlarge` | 8 | 16 GiB | 1 | 7 |
| `c7g.4xlarge` | 16 | 32 GiB | 1 | 15 |

The target class is Linux `aarch64`, AWS Graviton3 / Arm Neoverse-V1. The
topology is single-host loopback. These results are not remote-network
measurements.

## Protocol

### llama-bench scaling

```text
prompt test: pp512
generation test: tg128
warm-up processes per thread point: 1
measured repetitions per thread point: 5
server thread plans:
  c7g.xlarge: 1, 2, 3
  c7g.2xlarge: 1, 2, 3, 4, 6, 7
  c7g.4xlarge: 1, 2, 3, 4, 6, 7, 8, 12, 15
```

### llama-server tuning

```text
client CPUs: 1 reserved logical CPU
concurrency: 4
candidates:
  baseline: one slot
  candidate: two slots with continuous batching
  candidate: four slots with continuous batching
screening requests per workload/profile slice: 12
paired confirmation pairs: 3
measured requests per workload/block: 20
server restart between blocks: yes
deterministic randomized order: yes
retries: 0
noise band: 5%
bootstrap samples: 10,000
```

## Thread Scaling

The one-thread measurements were repeated independently on each target. The
largest point reserves one logical CPU for the client, so it uses 3, 7, or 15
server threads rather than all target CPUs.

| Target | Test | 1 thread tok/s | Max threads | Max tok/s | Speedup from 1 thread | Parallel efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `c7g.xlarge` | prefill pp512 | 6.705 | 3 | 16.444 | 2.453x | 81.8% |
| `c7g.xlarge` | decode tg128 | 2.706 | 3 | 7.101 | 2.624x | 87.5% |
| `c7g.2xlarge` | prefill pp512 | 6.713 | 7 | 38.993 | 5.808x | 83.0% |
| `c7g.2xlarge` | decode tg128 | 2.736 | 7 | 15.541 | 5.680x | 81.1% |
| `c7g.4xlarge` | prefill pp512 | 6.714 | 15 | 73.598 | 10.962x | 73.1% |
| `c7g.4xlarge` | decode tg128 | 2.745 | 15 | 29.871 | 10.883x | 72.6% |

Absolute throughput increases with target size. Parallel efficiency decreases
at the largest thread count, which is consistent with increasing shared
resource pressure, but this protocol does not include memory-controller
counters and therefore does not prove a specific hardware bottleneck.

## Formal Paired Serving Confirmation

Only the selected four-slot candidate entered formal paired confirmation.
Each row compares baseline and candidate on the same target, workload, model,
binary, concurrency, CPU allocation, and protocol.

| Target | Workload | Baseline tok/s | Selected tok/s | Ratio-of-medians delta | Bootstrap p05-p95 | Baseline p95 TTFT | Selected p95 TTFT | TTFT reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c7g.xlarge` | `interactive-short` | 6.204 | 11.470 | +84.866% | +83.610% to +85.978% | 10672.111 ms | 2774.197 ms | 74.005% |
| `c7g.xlarge` | `decode-heavy` | 6.237 | 12.136 | +94.585% | +94.321% to +95.699% | 28153.010 ms | 2554.761 ms | 90.925% |
| `c7g.2xlarge` | `interactive-short` | 13.554 | 24.539 | +81.040% | +80.962% to +81.186% | 4880.742 ms | 1208.990 ms | 75.229% |
| `c7g.2xlarge` | `decode-heavy` | 13.626 | 25.876 | +89.901% | +89.550% to +90.156% | 12879.642 ms | 1235.923 ms | 90.404% |
| `c7g.4xlarge` | `interactive-short` | 25.470 | 44.145 | +73.324% | +71.765% to +73.278% | 2607.447 ms | 585.763 ms | 77.535% |
| `c7g.4xlarge` | `decode-heavy` | 25.475 | 46.186 | +81.296% | +80.972% to +81.561% | 7058.402 ms | 612.340 ms | 91.325% |

All six bootstrap intervals exclude zero. All six comparisons passed execution,
parsing, quality, cleanup, minimum-sample, provenance, and 5% noise-band
guards.

## Workload-Aware Rejection

Four-slot serving did not help every workload. The screening stage rejected
`prefill-heavy` on all three target sizes:

| Target | Two-slot delta | Four-slot delta | Guard result |
| --- | ---: | ---: | --- |
| `c7g.xlarge` | -23.049% | -38.977% | rejected; TTFT regression |
| `c7g.2xlarge` | -22.287% | -38.039% | rejected; TTFT regression |
| `c7g.4xlarge` | -18.787% | -35.310% | rejected; TTFT regression |

The public recommendation therefore contains only `interactive-short` and
`decode-heavy`. PerfRunbench does not turn the successful workloads into a
global recommendation for prefill-heavy traffic.

## Evidence Integrity

| Target | Measured HTTP requests | Server starts | Downloaded artifacts revalidated | Digest failures |
| --- | ---: | ---: | ---: | ---: |
| `c7g.xlarge` | 349 | 10 | 156 | 0 |
| `c7g.2xlarge` | 349 | 10 | 174 | 0 |
| `c7g.4xlarge` | 349 | 10 | 184 | 0 |
| **Total** | **1,047** | **30** | **514** | **0** |

Every target completed with zero HTTP request failures, zero quality failures,
port release, no residual `llama-server` or `llama-bench` process, and verified
host restoration.

## Interpretation

Valid conclusion:

```text
Under the tested C7g size, pinned Qwen2.5-7B model and llama.cpp runtime,
single-host concurrency-4 protocol, and declared guards, four-slot continuous
batching was the best observed eligible configuration for interactive-short
and decode-heavy serving.
```

Invalid conclusions:

- The percentages are not a universal Arm or llama.cpp speedup.
- Hardware scaling and runtime-tuning percentages must not be added or
  multiplied.
- Loopback results are not remote-network results.
- The prefill-heavy workload is not eligible for this recommendation.
- Parallel-efficiency behavior does not identify a specific memory bottleneck
  without additional counters.

## Public Artifacts

- Sanitized structured evidence:
  `docs/results/c7g-size-scaling/c7g_size_scaling_validation.json`
- Thread-scaling CSV:
  `docs/results/c7g-size-scaling/thread_scaling.csv`
- Serving-comparison CSV:
  `docs/results/c7g-size-scaling/serving_comparisons.csv`
- Visual report:
  `docs/results/c7g-size-scaling/c7g_size_scaling_report.html`
- SVG charts:
  `docs/results/c7g-size-scaling/charts/`
- PNG review captures:
  `docs/results/c7g-size-scaling/screenshots/`

The public export is generated from preserved per-target summaries and
controller-side artifact revalidation records through:

```text
autotuneai report-arm-size-scaling
```

Private connection details, SSH trust material, cloud resource identifiers,
absolute paths, and raw model files are not included.
