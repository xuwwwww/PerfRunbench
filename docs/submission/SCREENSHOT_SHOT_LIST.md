# Final Screenshot Set

All screenshots are 16:9 PNG files under
`docs/submission/assets/devpost/`. They are generated from the checked-in
Slidev deck after its visual QA pass.

## Upload Order

### 1. Product Overview

- File: `01-perfrunbench-overview.png`
- Caption:
  `PerfRunbench turns bounded search, statistical confirmation, and guarded reuse into one workflow.`
- Evidence class: project overview plus scoped real-result headline

### 2. Real Arm Remote Serving Throughput

- File: `02-real-arm-serving-throughput.png`
- Caption:
  `On the tested C7g server, T4g remote client, pinned model/runtime, and concurrency-4 protocols, output throughput improved by 69.375%-86.490%.`
- Evidence class: measured
- Scope: C7g server, T4g remote client, Qwen2.5-7B Q4_K_M, pinned
  llama-server, direct private path, concurrency 4

### 3. Real Arm Remote Serving Latency And Reliability

- File: `03-real-arm-serving-latency.png`
- Caption:
  `The same scoped remote validation reduced p95 TTFT by 91.128%-98.001% across three workloads, with 608 measured requests and zero failures.`
- Evidence class: measured

### 4. Actual Config-Free Run

- File: `04-public-fixture-run-auto.png`
- Caption:
  `An actual public fixture run demonstrates config-free run --auto; fixture output is not performance evidence.`
- Evidence class: `test_fixture`
- Source command:

  ```bash
  autotuneai run --auto --preset quick -- python app.py
  ```

### 5. Recommendation Guards

- File: `05-recommendation-guards.png`
- Caption:
  `Hard drift, noise-band, cleanup, and no-evidence guards prevent unsafe recommendations.`
- Evidence class: product safety behavior

### 6. Arm And Platform Validation

- File: `06-arm-and-platform-validation.png`
- Caption:
  `Real Arm64 performance evidence is separated from installed-wheel compatibility validation on Windows, Linux x86-64, and Linux Arm64.`
- Evidence class: scoped Arm evidence plus compatibility, clearly separated

### 7. Real T4g Training Tuning Evidence

- File: `07-t4g-training-tuning-evidence.png`
- Caption:
  `On one T4g.small and one public synthetic CPU PyTorch protocol, 12 paired confirmations measured 14,430 to 33,275 samples/s while preserving failures, CPU-credit state, and guard evidence.`
- Evidence class: measured synthetic-workload guard and workflow evidence
- Scope: one T4g.small, CPU-only PyTorch 2.13.0, fixed effective batch size 32,
  5,000-step slices, 12 paired confirmations, T4g Unlimited surplus state

### 8. Larger Arm Capacity And Workload-Aware Guards

- File: `08-c7g4xlarge-workload-aware-evidence.png`
- Caption:
  `On one C7g.4xlarge and a pinned Qwen2.5-7B/llama.cpp loopback protocol, fifteen-thread throughput scaled about 10.9x over one thread; eligible concurrency-8 serving workloads improved 70.749%-80.331%, while prefill regression and context failures were rejected.`
- Evidence class: measured
- Scope: one C7g.4xlarge, Linux aarch64, Neoverse-V1, one pinned Qwen2.5-7B
  Q4_K_M artifact, one pinned native llama.cpp build, one client CPU, fifteen
  server CPUs, single-host loopback
- Claim boundary: thread scaling and runtime tuning are separate protocols and
  are not added; this is not remote-network evidence

### 9. Same-Protocol C7g Thread Scaling

- File: `09-c7g-size-thread-scaling.png`
- Caption:
  `With one pinned model and llama.cpp runtime, measured pp512/tg128 throughput scales across C7g.xlarge, C7g.2xlarge, and C7g.4xlarge while the largest thread point reserves one CPU for the client.`
- Evidence class: measured
- Scope: three AWS C7g sizes, Linux aarch64 / Neoverse-V1, one pinned
  Qwen2.5-7B Q4_K_M artifact, one pinned llama.cpp runtime bundle, one warm-up
  and five measured repetitions per thread point
- Claim boundary: absolute hardware scaling; not a runtime-tuning percentage

### 10. C7g Before And After With Workload Guards

- File: `10-c7g-size-before-after.png`
- Caption:
  `Across three tested C7g sizes, four-slot continuous batching improved eligible interactive and decode serving under the same concurrency-4 loopback protocol; prefill-heavy traffic was rejected separately.`
- Evidence class: measured
- Scope: three AWS C7g sizes, pinned identity, single-host loopback,
  concurrency 4, three randomized paired blocks and 20 measured requests per
  workload and side
- Claim boundary: loopback serving only; percentages are not added to
  cross-size scaling or remote-client results

## Supporting Assets

- Deck source:
  `docs/submission/arm-ai-challenge-deck/slides.md`
- Exported PDF:
  `docs/submission/arm-ai-challenge-deck/PerfRunbench-Arm-AI-Challenge.pdf`
- Raw fixture output:
  `docs/submission/evidence/public_fixture_run_auto.json`
- Screenshot provenance:
  `docs/submission/evidence/SCREENSHOT_PROVENANCE.md`
- C7g size-scaling report:
  `docs/results/C7G_SIZE_SCALING_VALIDATION.md`

## Never Include

- AWS public or private addresses
- SSH key paths or host-key fingerprints
- Security Group IDs
- cloud credentials or tokens
- model paths or model weights
- private artifact directories
- local absolute paths
- browser account-only controls
