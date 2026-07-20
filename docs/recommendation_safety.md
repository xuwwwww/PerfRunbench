# PerfRunbench Recommendation Safety

## Scope

PerfRunbench recommendations convert an audited optimization result into an
environment-bound launch or runtime configuration. The recommendation layer is
generic across workloads: ArmTune, `llama.cpp`, GPU tuning, and training
workloads may provide evidence, but the recommendation safety model does not
depend on a specific benchmark backend.

The default product flow is:

```text
inspect target
-> inspect workload runtime
-> benchmark baseline
-> build or configure profiles
-> search runtime candidates
-> confirm selected candidate
-> generate Performance Passport
-> generate RecommendationBundle
-> validate, dry-run, canary, lease, promote, or rollback
```

BootTrace-specific prompt contracts and firmware-log model qualification have
been split into a standalone BootTraceBench project. They are not part of
default PerfRunbench candidate eligibility.

## RecommendationBundle

Every persisted recommendation is a versioned top-level JSON object:

```text
schema_version: "1.0"
bundle_id
source_run_id
source_passport_path
source_passport_sha256
status
objective.metric
objective.direction
baseline_configuration
selected_configuration
environment_fingerprint
workload_fingerprint
constraints
statistical_evidence
apply_scope
rollback_configuration
created_at
expires_at
guard_results
```

`selected_configuration` must be complete enough to execute without resolving a
candidate ID through mutable search state. It includes the adapter identity,
working directory, argument vector, environment overrides, timeout, expected
artifacts, correctness contract, and workload metadata.

Bundles are rejected when the source is a fixture, synthetic example,
incomplete run, inconclusive comparison, invalid Passport, stale environment,
missing executable configuration, or failed artifact validation. Inconclusive
optimization results may remain useful evidence, but they must not become an
applyable recommendation.

Important recommendation JSON writes are atomic. A temporary file is written,
flushed, and replaced into place so partial writes cannot be mistaken for valid
recommendations.

## Guard Stages

Guard results are structured records, not one boolean:

```text
stage: preflight | candidate | trial | recommendation | apply | post_apply
status: pass | warn | fail | not_available
code
message
evidence_refs
recoverable
timestamp
```

The guard engine preserves warnings and unavailable observations. Missing
temperature, frequency, GPU occupancy, or platform telemetry is recorded as
`not_available`, not as zero.

## Environment Fingerprints

Hard-match fields:

- architecture
- CPU model
- logical CPU compatibility
- allowed CPU mask compatibility
- executable digest
- model digest
- runtime identity and version
- workload config digest
- container image digest
- required accelerator identity

Soft-match fields:

- kernel patch level
- available memory
- hostname or instance identity
- noncritical tool versions

A hard mismatch marks the bundle `stale_environment` and sets
`retune_required: true`. PerfRunbench must not silently apply stale
recommendations.

## Correctness Contracts

Adapters may use only allowlisted validators:

- `exit_code`
- `required_metric`
- `metric_finite`
- `metric_min`
- `metric_max`
- `output_file_exists`
- `output_nonempty`
- `artifact_digest_match`
- `request_completed`
- `stream_parsed`
- `cleanup_passed`

No adapter may run arbitrary validation shell commands. The initial contract
types are:

- generic command
- `llama-bench`
- `llama-server`

A faster candidate that fails correctness is recorded as
`rejected_correctness` and is not eligible for recommendation.

## Statistical Recommendation Guard

Recommendation eligibility requires:

- minimum valid trials
- failed trials preserved in the evidence
- warm-up trials excluded from medians
- paired baseline evidence
- fresh confirmation samples separate from search samples
- median comparison in the requested objective direction
- improvement outside the configured noise band
- variability under the configured threshold when one is provided
- consistent confirmation direction

A recommendation must not be based only on search samples. If improvement
evidence is missing or inside the configured noise band, the result is
`inconclusive`.

## Trial Contamination

Trial evidence may include load average, background CPU, CPU steal, available
memory, swap delta, free disk, target fingerprint stability, GPU occupancy,
frequency/governor state, and thermal state. A contaminated trial is marked:

```text
trial_status: contaminated
```

Contaminated trials are preserved and excluded from valid-sample statistics.
Retries are bounded. Unsupported telemetry remains `not_available`.

## Apply Scopes

`process` is the default apply scope. It creates an executable launch
specification from the selected workload configuration and does not modify
persistent system settings.

`system` scope may apply runtime system changes only through existing
PerfRunbench snapshot, active-state, restore, and interrupted-run recovery
mechanisms. Recommendation apply must not create a second system restore
implementation, execute arbitrary commands, or perform arbitrary sudo actions.

## Transactional Apply

Apply follows this sequence:

```text
validate recommendation and Passport
acquire ownership lock
revalidate environment fingerprint
snapshot current state
apply selected configuration
run canary
activate lease
record result
release lock when appropriate
```

On any failure, PerfRunbench rolls back through the recorded rollback
configuration, verifies rollback, records the failure, and releases or updates
the lock. A partial apply is never reported as success.

## Canary, Lease, Promote, Rollback

Commands are:

```text
autotuneai recommendation show --run-id <id>
autotuneai recommendation validate --run-id <id>
autotuneai recommendation apply --run-id <id> --scope process --lease-minutes <n>
autotuneai recommendation status --run-id <id>
autotuneai recommendation promote --run-id <id>
autotuneai recommendation rollback --run-id <id>
```

Dry-run apply prints the launch specification and guard summary without running
the workload.

Canary requirements:

- short warm-up
- bounded measured runs
- baseline throughput floor when configured
- baseline latency ceiling when configured
- zero required execution failures
- correctness contract pass
- cleanup pass

Process-scoped leases expire when the process ends or when the lease deadline is
reached. System-scoped leases restore through existing PerfRunbench mechanisms.

Ownership locks record owner process, hostname, created timestamp, updated
timestamp, lease expiration, bundle ID, scope, and state. Lock states are:

```text
active
stale
dead
expired
orphan
released
promoted
rolled_back
```

An active lock is not overwritten. Explicit takeover is required for stale,
dead, expired, or orphaned locks.

## Phase 9 Redacted Source Example

The first real ArmTune tuning result may be converted into a
RecommendationBundle only as an environment-bound preliminary recommendation.
The source evidence is recorded in
`docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md`.

The source result is measured Linux Arm64 evidence on one user-provisioned
C7g / Graviton3 target, one Qwen2.5-7B Q4_K_M model digest, one pinned
`llama.cpp` commit, and one benchmark protocol. Private target configuration,
public IP address, SSH key path, identity-file contents, private controller
paths, and model contents remain redacted.

Valid wording:

```text
Best observed eligible configuration under the tested Arm target, model,
inference workload, and measurement protocol.
```

Invalid wording:

```text
Global best Arm configuration.
Cross-instance performance guarantee.
External semantic-quality pass from a workload-specific project.
KleidiAI performance claim without build and runtime evidence.
```
