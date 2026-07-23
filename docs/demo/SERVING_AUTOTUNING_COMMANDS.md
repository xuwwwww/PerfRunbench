# Serving Autotuning Commands

These commands are public and sanitized. They do not contain private IPs,
private paths, SSH keys, model paths, or cloud configuration.

## Inspect Built-In Workloads

```bash
autotuneai serve workload show interactive-short --json
autotuneai serve workload show decode-heavy --json
autotuneai serve workload show prefill-heavy --json
```

## Show Validated Recommendations

```bash
autotuneai serve recommend \
  --workload interactive-short \
  --concurrency 4 \
  --recommendation-class maximum_throughput_remote

autotuneai serve recommend \
  --workload decode-heavy \
  --concurrency 4 \
  --recommendation-class maximum_throughput_remote

autotuneai serve recommend \
  --workload prefill-heavy \
  --concurrency 4 \
  --recommendation-class interactive_latency_remote
```

## Generate A Serving Passport

```bash
autotuneai serve recommend \
  --workload interactive-short \
  --concurrency 4 \
  --recommendation-class maximum_throughput_remote \
  --output results/serving/remote_recommendation_bundle.json \
  --passport-output results/serving/serving_performance_passport.json \
  --json
```

## Dry-Run Run-Best

```bash
autotuneai serve run \
  --best \
  --workload interactive-short \
  --concurrency 4 \
  --dry-run \
  --json
```

Expected properties:

```text
run_best_performs_search: false
process_started: false
execution_status: dry_run_no_mutation
selected_profile: slots4-cont-on
parallel_slots: 4
continuous_batching: enabled
```

## Hard-Drift Refusal

Loopback topology is not remote-client evidence:

```bash
autotuneai serve run \
  --best \
  --workload interactive-short \
  --concurrency 4 \
  --current-network-topology loopback \
  --json
```

Expected result:

```text
state: topology_mismatch
run_best_performs_search: false
process_started: false
```

Different concurrency is workload drift:

```bash
autotuneai serve run \
  --best \
  --workload interactive-short \
  --concurrency 8 \
  --dry-run
```

Expected result:

```text
no validated remote serving recommendation
```

## Fixture Search Still Exists

Fixture mode remains available for public CI and local demos. It is not
measured Arm performance.

```bash
autotuneai serve plan \
  --workload interactive-short \
  --traffic closed-loop \
  --concurrency 4 \
  --candidate-cap 8

autotuneai serve search \
  --workload interactive-short \
  --candidate-cap 8 \
  --fixture
```

