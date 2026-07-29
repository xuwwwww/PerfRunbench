# PerfRunbench Validated Performance Summary

This summary separates measured performance results, guard evidence, and
unsupported or inconclusive mechanisms. Numbers from different targets,
protocols, or workload shapes must not be combined into a single speedup.

## Headline

PerfRunbench did not verify a compiler flag, backend flag, kernel profile,
page-size profile, allocator profile, or machine-wide Linux tuning profile
outside the configured guard for the fixed single-stream `llama-bench`
protocol.

The verified performance result is workload-aware serving autotuning:
`slots4-cont-on` improved remote-client `llama-server` throughput and p95 TTFT
for the tested C7g server, T4g remote-client topology, Qwen2.5-7B Q4_K_M model,
pinned binary, concurrency 4, and direct private network path.

A later, separate C7g.4xlarge loopback experiment adds larger-target scaling
and workload-aware guard evidence. It does not replace or extend the remote
network claim.

A newer same-protocol C7g size-scaling experiment compares xlarge, 2xlarge,
and 4xlarge with one pinned runtime and concurrency-4 loopback protocol. It
confirms the same workload-aware result across three sizes: four-slot
continuous batching is eligible for interactive/decode, while prefill-heavy is
rejected.

Recommended wording:

```text
Best observed eligible serving configuration under the tested C7g server,
T4g remote-client topology, Qwen2.5-7B Q4_K_M model, pinned llama-server
binary, workload, concurrency, and measurement protocol.
```

## Validated Matrix

| Area | Validated scope | Result | Claim boundary |
| --- | --- | --- | --- |
| Compiler/build profiles | C7g / Neoverse-V1, Qwen2.5-7B Q4_K_M, fixed `llama-bench` build-profile protocol | Earlier Phase 9 showed `portable-baseline` 10.825 tok/s, `arm-native` 17.530 tok/s, `arm-optimized` 17.380 tok/s; later guarded Phase 20 build/backend comparison did not verify a fresh difference outside the 5% noise band | Do not claim universal compiler/backend superiority |
| Backend evidence | Pinned `llama.cpp` binaries and startup evidence | KleidiAI or other optimized paths are recorded only when enabled and evidenced; requested flags alone are not sufficient | Host capability is separate from runtime backend evidence |
| Runtime threads/search | C7g, fixed `llama-bench` bounded runtime search | Phase 9 search baseline 13.630 tok/s to selected 17.480 tok/s, observed +28.246515% under that protocol; later Phase 20 CPU-aware run stayed inside noise | Protocol-specific only; do not combine with build-profile numbers |
| Reversible system tuning | C7g real host, one reversible family at a time | Phase 21B and 21B2 exhausted safe reversible families; no positive same-protocol candidate cleared the 5% noise band | Guard evidence, not a speedup |
| mmap / mlock / prefault | C7g real host, reversible checks | `no-mmap` best same-protocol point was +3.014%, inside the 5% guard; mlock/prefault lacked qualified positive evidence | No recommendation |
| Allocators | C7g real host, non-persistent extraction when available | No allocator profile produced eligible same-protocol evidence outside guard | No recommendation |
| THP / HugeTLB | C7g real host, bounded reversible checks | THP was exercised but stayed inside guard; HugeTLB consumer qualification did not create an eligible recommendation | No recommendation |
| Storage cold-start | C7g real host, cold-start-oriented profiles | Evidence was separated as cold-start/protocol-specific and not ranked against steady-state decode throughput | Diagnostic only |
| 4K boot isolation | C7g one-time 4K boot-parameter experiments | G1/G2 stayed inside guard; G3 regressed | No boot-isolation recommendation |
| 4K versus 64K kernel | C7g official Ubuntu AWS ARM64 64K validation | 4K versus 64K full and matched comparisons stayed inside the 5% noise band | No page-size recommendation |
| 64K isolation | C7g official 64K kernel plus isolation profiles | G5/G6 stayed inside guard; G7 regressed | No 64K isolation recommendation |
| Loopback serving | C7g local client/server loopback, `interactive-short` concurrency 4 | `slots4-cont-on` improved loopback output throughput by +67.602% and p95 TTFT by +91.064% | Loopback-only; not remote-network evidence |
| Remote serving | C7g server, T4g remote client, direct private path, concurrency 4 | `slots4-cont-on` improved remote output throughput by +69.375% to +86.490% across the three J1 workloads, with p95 TTFT reductions above 91% | Validated remote-client serving recommendation |
| Remote network tuning | C7g/T4g read-only network bottleneck evidence | No client or network bottleneck was observed, so IRQ/RPS/XPS/backlog tuning was not executed | No network recommendation |
| Larger-target thread scaling | C7g.4xlarge, 1-15 threads, pp512/tg128, pinned Qwen2.5-7B Q4_K_M and fresh llama.cpp build | Prefill `6.761 -> 73.669 tok/s` (`10.897x`); decode `2.759 -> 30.089 tok/s` (`10.907x`) | Separate scaling protocol; not a cross-instance speedup |
| Larger-target loopback serving | C7g.4xlarge, 15 server CPUs plus one client CPU, concurrency 8 | `slots4-cont-on` improved eligible interactive/decode throughput by `+70.749%` to `+80.331%`; prefill-heavy regressed `-21.025%` and was rejected | Loopback and workload scoped; not remote-network evidence |
| Same-protocol C7g size scaling | C7g.xlarge, C7g.2xlarge, and C7g.4xlarge; pinned model/runtime; one client CPU; concurrency 4 | Decode baseline-to-selected medians were `6.237 -> 12.136`, `13.626 -> 25.876`, and `25.475 -> 46.186 tok/s`; within-target deltas were `+94.585%`, `+89.901%`, and `+81.296%` | Runtime tuning within each target; do not add to hardware scaling |
| Cross-size workload guard | Same three-size protocol | Interactive/decode cleared the guard; four-slot prefill-heavy regressed `-38.977%`, `-38.039%`, and `-35.310%` and was rejected | No prefill-heavy recommendation |

## Remote Serving Result

All rows compare baseline versus `slots4-cont-on` within the same target,
workload, topology, concurrency, binary, model, and protocol.

| Stage | Workload | Baseline output tok/s | Candidate output tok/s | Throughput delta | Baseline TTFT p95 ms | Candidate TTFT p95 ms | TTFT p95 reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| J2 seven CPU control | `interactive-short` | 15.334 | 28.295 | +84.529% | 3389.856 | 300.557 | 91.134% |
| J1 full eight CPU | `interactive-short` | 17.032 | 31.355 | +84.092% | 3055.682 | 271.088 | 91.128% |
| J1 full eight CPU | `decode-heavy` | 17.046 | 31.790 | +86.490% | 10126.795 | 202.424 | 98.001% |
| J1 full eight CPU | `prefill-heavy` | 16.563 | 28.054 | +69.375% | 3967.424 | 341.693 | 91.388% |

Remote serving protocol checks:

```text
measured_requests_total: 608
paired_block_pairs: 6
retries: 0
request_failures: 0
quality_failures: 0
client_bottleneck_detected: false
network_tuning_status: network_tuning_not_executed_evidence_absent
artifact_validation: passed
dual_host_restored: true
```

## C7g.4xlarge Capacity Result

The larger-target run used a separate binary digest and protocol. Its numbers
must not be compared directly with, or added to, Phase 21D2B.

| Protocol | Workload | Baseline output tok/s | Candidate output tok/s | Delta | TTFT p95 reduction | Eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 7-CPU capacity match / c4 | `interactive-short` | 14.297 | 26.168 | +83.033% | 71.791% | yes |
| 15-CPU full capacity / c8 | `decode-heavy` | 25.779 | 46.488 | +80.331% | 67.736% | yes |
| 15-CPU full capacity / c8 | `interactive-short` | 25.600 | 43.712 | +70.749% | 58.950% | yes |
| 15-CPU full capacity / c8 | `prefill-heavy` | 19.734 | 15.585 | -21.025% | -109.684% | no |

The full-capacity recommendation contains only `decode-heavy` and
`interactive-short`. Eight slots failed long-prefill screening because the
request required 412 context tokens while only 256 were available per slot.
The run recorded 769 measured HTTP requests, zero eligible-confirmation
failures, complete cleanup, and 278 independently revalidated downloaded
artifacts.

## Three-Size C7g Result

The newer three-size experiment uses one pinned model, source, runtime bundle,
binary identity, CPU-allocation rule, and concurrency-4 loopback protocol.

| Target | Server threads | Workload | Baseline tok/s | Selected tok/s | Delta | p95 TTFT reduction |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `c7g.xlarge` | 3 | `interactive-short` | 6.204 | 11.470 | +84.866% | 74.005% |
| `c7g.xlarge` | 3 | `decode-heavy` | 6.237 | 12.136 | +94.585% | 90.925% |
| `c7g.2xlarge` | 7 | `interactive-short` | 13.554 | 24.539 | +81.040% | 75.229% |
| `c7g.2xlarge` | 7 | `decode-heavy` | 13.626 | 25.876 | +89.901% | 90.404% |
| `c7g.4xlarge` | 15 | `interactive-short` | 25.470 | 44.145 | +73.324% | 77.535% |
| `c7g.4xlarge` | 15 | `decode-heavy` | 25.475 | 46.186 | +81.296% | 91.325% |

All six 10,000-sample bootstrap intervals exclude zero. The protocol recorded
1,047 measured HTTP requests, 30 fresh server starts, zero request or quality
failures, complete cleanup, and 514 successful controller-side digest checks
after download.

The four-slot prefill-heavy candidate was rejected on every size. Detailed
methodology and sanitized data are in
`docs/results/C7G_SIZE_SCALING_VALIDATION.md` and
`docs/results/c7g-size-scaling/`.

## Recommendation Classes

Eligible public serving recommendation classes:

- `maximum_throughput_remote`
- `interactive_latency_remote`

Eligible workloads:

- `interactive-short`, concurrency 4
- `decode-heavy`, concurrency 4
- `prefill-heavy`, concurrency 4

Not created because evidence did not qualify:

- balanced multi-user recommendation
- shared-prefix recommendation
- memory-constrained recommendation
- remote network tuning recommendation

## Evidence Documents

- `docs/results/PHASE_21D1_C7G_LOOPBACK_SERVING_VALIDATION.md`
- `docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md`
- `docs/results/C7G4XLARGE_REALISTIC_ARM_INFERENCE_VALIDATION.md`
- `docs/results/C7G_SIZE_SCALING_VALIDATION.md`
- `docs/results/c7g4xlarge-realistic/`
- `docs/results/VALIDATION_MATRIX.md`

