# ArmTune Limitations

## Release Scope

PerfRunbench ArmTune v0.1.0 focuses on Arm AI inference benchmarking and
guarded autotuning:

```text
inspect Arm target
-> inspect workload runtime
-> benchmark portable baseline
-> build Arm profiles
-> search bounded runtime candidates
-> confirm the selected candidate
-> generate and verify a Performance Passport
-> validate guarded recommendation apply
```

`llama.cpp` is the first workload adapter. It is not the product identity.

## Evidence Limits

- The public measured evidence is preliminary and single-target.
- The measured target was one Linux Arm64 C7g / Neoverse-V1 system.
- The measured model was Qwen2.5-7B Q4_K_M.
- The measured runtime used one pinned llama.cpp source revision.
- Results from `llama-bench` profile comparison, `llama-bench` runtime
  search, and `llama-server` request timing are separate protocols and must
  not be combined into one aggregate improvement.
- The Performance Passport state is `eligible_preliminary`, not a final
  production SLO, cloud-wide guarantee, or cross-instance portability claim.
- Fixture, mocked, synthetic, CI-runner, or fake-executable results remain
  `test_fixture` evidence and are not production Arm performance results.

## Real Phase 9 Result Boundary

The release evidence includes:

- `portable-baseline` decode median: `10.825 tok/s`
- `arm-native` decode median: `17.530 tok/s`
- `arm-optimized` decode median: `17.380 tok/s`
- runtime-search baseline decode median: `13.630 tok/s`
- selected runtime-search decode median: `17.480 tok/s`
- observed runtime-search improvement: `28.246515%`
- `portable-baseline` `llama-server` TTFT median: `97.260 ms`
- `arm-native` `llama-server` TTFT median: `62.826 ms`
- Performance Passport: `eligible_preliminary`, verification passed
- guarded apply validation: passed

These values support only bounded wording:

> Best observed eligible configuration under the tested Arm target, model,
> inference workload, and measurement protocol.

They do not prove global optimality, final competition-wide performance,
PerfRunbench behavior on other Arm systems, or stable production performance
under sustained load.

## Optimized-Path Claims

ArmTune does not infer optimized execution from requested flags or host
features alone. KleidiAI, Neon, DotProd, I8MM, BF16, SVE, SVE2, SME, SME2, or
another optimized path may be reported as enabled only when the recorded build
and runtime evidence supports that state.

Unverified states must remain explicit, such as:

- `requested_but_unverified`
- `runtime_unverified`
- `unavailable`
- `build_failed`

## Guarded Apply Limits

The first release validates process-scoped guarded apply with environment
validation, dry-run, canary, lease, rollback, artifact revalidation, and
cleanup. Permanent system-scoped apply remains out of scope for this release.

Guarded apply does not create a new benchmark result. Canary metrics are safety
evidence only and must not replace the repeated Phase 9 benchmark
distributions.

## Golden-Path CLI Limits

`autotuneai run --auto` is a convenience workflow for command identity,
bounded retuning, cache reuse, guarded process-scoped apply, and command
launch. It is not a guarantee that a cached recommendation is globally optimal
or portable to another machine, model, dataset, container image, or workload
configuration.

The recommendation cache is local. Cache match states distinguish exact
matches, soft drift, hard drift, expiry, revocation, corruption, ambiguity, and
missing entries, but the operator still owns workload-level correctness and
production SLO decisions.

The golden-path metric sources are allowlisted JSON/JSONL/stdout record inputs.
PerfRunbench does not infer application quality from arbitrary logs.

## External Workload Correctness

Workload-neutral correctness hooks remain in PerfRunbench: request completion,
nonempty output checks, JSON parsing, required metric checks, stream parsing,
cleanup checks, artifact digest checks, and recommendation guards.

Firmware-log model capability evaluation and related semantic suites were
split into the standalone BootTraceBench project after the
`v0.1.0-armtune` release. They are not part of default ArmTune candidate
eligibility in current main.

The `v0.1.0-armtune` tag preserves the earlier integrated experimental state
for historical review.

## Out Of Scope

- Automatic cloud provisioning or AWS lifecycle control.
- Downloading or selecting new models as part of default tuning.
- Updating llama.cpp source during a measured comparison.
- New benchmark backends beyond the existing llama.cpp adapter.
- New search algorithms beyond the bounded staged search implemented for the
  release.
- Performix evidence.
- Power measurement.
- Kubernetes orchestration.
- Hosted dashboards, user accounts, or SaaS workflows.
- Private firmware logs, customer data, NDA material, credentials, SSH keys,
  private target configs, or model weights in public artifacts.
