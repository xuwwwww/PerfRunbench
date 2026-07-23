# PerfRunbench Architecture

PerfRunbench is a guarded benchmarking and autotuning system. Its core
responsibility is to bind a recommendation to reproducible evidence, validate
that evidence before reuse, and refuse unsafe launches when drift or missing
proof is detected.

## Core Flow

```text
controller CLI
-> environment and target inspection
-> workload adapter
-> bounded candidate generation
-> repeated benchmark execution
-> statistical guard
-> Performance Passport
-> RecommendationBundle
-> cache reuse or guarded apply
```

## Main Components

| Component | Responsibility |
| --- | --- |
| `autotune.golden_path` | Public `init`, `doctor`, `tune`, `run --auto`, and `run --best` workflows. |
| `autotune.recommendation` | RecommendationBundle generation, validation, canary, lease, status, rollback, and cache integration. |
| `autotune.training_adapter` | Dependency-free and PyTorch-backed bounded training slice adapter. |
| `autotune.arm_tune` | Arm target inspection, SSH/local target orchestration, `llama.cpp` benchmarking, runtime search, build profiles, and Performance Passport evidence. |
| `autotune.serving` | Workload-aware `llama-server` capability discovery, traffic profiles, deterministic load generation, Pareto selection, network plans, and serving recommendation guards. |
| `autotune.system_profiles` | Host state snapshots, Linux system-profile planning, reversible leases, rollback, protocol separation, and boot-profile planning using fake-root validation. |
| `autotune.resource` | Run directories, process execution, monitoring, CPU affinity, resource budgets, and cleanup. |
| `autotune.report` | Markdown and HTML report helpers. |

BootTraceBench is external and experimental. It does not participate in default
PerfRunbench recommendation eligibility.

## ArmTune Build/Runtime Path

```mermaid
flowchart LR
    controller[Controller] --> target[Arm target inspection]
    target --> source[Versioned runtime source lock]
    source --> builds[CPU-aware build profiles]
    builds --> verify[Build and backend evidence]
    verify --> compare[Fixed build-profile comparison]
    compare --> search[Bounded runtime search]
    search --> confirm[Fresh paired confirmation]
    confirm --> passport[Performance Passport]
    passport --> bundle[RecommendationBundle]
    bundle --> apply[Guarded apply]
```

`llama.cpp` is the first ArmTune workload adapter. It is not the product
identity. Phase 19 makes CPU-specific compilation a first-class tuning
dimension without rewriting PerfRunbench in another language.

## Phase 19 CPU-Aware Build Boundary

Phase 19 adds:

- build profile schema fields for CPU target policy, backend policy,
  optimization policy, effective flags, backend evidence, and native-binary
  portability scope
- compiler target probing without `shell=True`
- KleidiAI build/runtime evidence states
- decode-throughput build comparison while preserving prefill as a separate
  metric
- optional Performance Passport build evidence
- native binary reuse guards independent from workload identity

Phase 19 does not:

- reinterpret Phase 9 C7g results
- modify the v0.2.0 tag or release
- download new models or datasets
- add BootTrace qualification logic
- claim global Arm speedup from one target

## Phase 21A Host System Profile Boundary

The system-profile layer is separate from workload adapters. It owns:

```text
host snapshot
-> capability report
-> safe profile catalog
-> lease-bound apply
-> rollback verification
-> protocol-specific evidence
-> optional Passport/cache binding
```

Phase 21A validates this layer with fake Linux roots only. Phase 21B and
Phase 21B2 validate bounded reversible C7g real-host profiles with exact
rollback, artifact revalidation, and protocol-specific ranking. They do not
produce a performance recommendation because no positive same-protocol
candidate cleared the configured noise band.

Bootloader writes, kernel installation, reboot-required profiles, and 4K/64K
kernel comparisons remain future Phase 21C validation work.

## Phase 21D Serving Path

Serving optimization is a new workload scope after the fixed single-request
kernel/profile search completed for the current C7g target:

```mermaid
flowchart LR
    caps[Server capability discovery] --> workloads[Workload and traffic profiles]
    workloads --> loadgen[Deterministic load generator]
    loadgen --> evidence[Per-request JSONL evidence]
    evidence --> metrics[Throughput, latency, fairness, resource metrics]
    metrics --> search[Staged bounded search]
    search --> pareto[Pareto frontier]
    pareto --> recs[Named serving recommendations]
    recs --> passport[Serving Passport extension]
    passport --> safety[Hard-drift and cleanup guards]
```

Phase 21D0 was local implementation and fixture validation only. Phase 21D1
validated loopback serving on the C7g target. Phase 21D2B validated the fresh
remote-client topology with a T4g load generator and packaged `slots4-cont-on`
as a workload-scoped serving recommendation for concurrency 4.

Loopback and remote-client results remain separate protocols. The v0.3 serving
RecommendationBundle and Serving Performance Passport bind the recommendation
to model digest, server binary digest, server capability digest, workload ID,
concurrency, CPU assignment, parallel slot count, continuous batching state,
network path, artifact validation, and cleanup evidence. `serve run --best`
loads this evidence and never performs a search.

## Evidence Principles

- Fixture, mocked, and CI-runner values remain fixtures.
- Host capability evidence is separate from workload execution evidence.
- Backend usage such as KleidiAI requires build and runtime proof.
- Parser failures are never converted into zero-valued metrics.
- Build-profile, runtime-search, and complete-configuration improvements are
  separate protocols and must not be added together.
- Serving throughput, latency, capacity, shared-prefix, and remote-network
  claims are separate protocols and recommendation classes.

