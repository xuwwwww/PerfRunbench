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

## Supporting Assets

- Deck source:
  `docs/submission/arm-ai-challenge-deck/slides.md`
- Exported PDF:
  `docs/submission/arm-ai-challenge-deck/PerfRunbench-Arm-AI-Challenge.pdf`
- Raw fixture output:
  `docs/submission/evidence/public_fixture_run_auto.json`
- Screenshot provenance:
  `docs/submission/evidence/SCREENSHOT_PROVENANCE.md`

## Never Include

- AWS public or private addresses
- SSH key paths or host-key fingerprints
- Security Group IDs
- cloud credentials or tokens
- model paths or model weights
- private artifact directories
- local absolute paths
- browser account-only controls
