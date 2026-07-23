# Serving Autotuning Demo

This demo is designed for a public repository or recorded walkthrough. It does
not require live AWS access and does not include private hostnames, private IPs,
SSH key paths, model paths, or benchmark-port values.

## Story

PerfRunbench starts with a conservative `llama-server` baseline and validates a
workload-aware serving recommendation only after repeated paired measurements,
direct remote-client traffic, guard checks, artifact validation, and cleanup.

The validated recommendation is `slots4-cont-on`:

```text
parallel_slots: 4
continuous_batching: enabled
ctx_size: 2048
batch_size: 512
ubatch_size: 128
topology: C7g server + T4g remote client over direct private path
workloads: interactive-short, decode-heavy, prefill-heavy
concurrency: 4
```

## Demo Flow

1. Show the validated performance boundary.

   ```bash
   type docs/results/VALIDATED_PERFORMANCE_SUMMARY.md
   ```

2. Show a workload-scoped recommendation.

   ```bash
   autotuneai serve recommend \
     --workload interactive-short \
     --concurrency 4 \
     --recommendation-class maximum_throughput_remote
   ```

3. Explain why the recommendation is eligible.

   ```bash
   autotuneai serve explain --recommendation <recommendation_id>
   ```

4. Show safe `run --best` behavior without retuning.

   ```bash
   autotuneai serve run \
     --best \
     --workload interactive-short \
     --concurrency 4 \
     --dry-run
   ```

5. Show hard-drift refusal.

   ```bash
   autotuneai serve run \
     --best \
     --workload interactive-short \
     --concurrency 4 \
     --current-network-topology loopback \
     --json
   ```

6. Show the Serving Performance Passport output.

   ```bash
   autotuneai serve recommend \
     --workload decode-heavy \
     --concurrency 4 \
     --recommendation-class maximum_throughput_remote \
     --passport-output results/serving/serving_performance_passport.json \
     --json
   ```

## Measured Values To Show

Do not combine these values with loopback, `llama-bench`, kernel, or
build-profile measurements.

| Workload | Baseline output tok/s | Candidate output tok/s | Throughput delta | Baseline TTFT p95 ms | Candidate TTFT p95 ms | TTFT p95 reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `interactive-short` | 17.032 | 31.355 | +84.092% | 3055.682 | 271.088 | 91.128% |
| `decode-heavy` | 17.046 | 31.790 | +86.490% | 10126.795 | 202.424 | 98.001% |
| `prefill-heavy` | 16.563 | 28.054 | +69.375% | 3967.424 | 341.693 | 91.388% |

Validation checks:

```text
measured_requests_total: 608
retries: 0
request_failures: 0
quality_failures: 0
artifact_validation: passed
dual_host_restored: true
network_tuning_status: network_tuning_not_executed_evidence_absent
```
