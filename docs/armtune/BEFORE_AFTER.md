# Before and After Tracking

## Competition Window

Official rules list:

- Registration period: 2026-06-10 09:00 Pacific Time to 2026-08-14 16:00
  Pacific Time.
- Submission period: 2026-06-10 09:00 Pacific Time to 2026-08-14 16:00
  Pacific Time.

## Pre-Challenge Baseline

Command used to identify the pre-challenge baseline:

```powershell
git log --all -1 --before="2026-06-10 09:00:00 -0700" --format="%H%n%aI%n%cI%n%an%n%cn%n%s"
```

Exact commit hash:

```text
bd7d4bc1f728482aa1d923d59e6a99a604be79f2
```

Author date:

```text
2026-05-15T18:12:04+08:00
```

Committer date:

```text
2026-05-15T18:12:04+08:00
```

Timezone:

```text
+08:00
```

Author:

```text
xuwwwww
```

Committer:

```text
xuwwwww
```

Subject:

```text
Add advanced tuning controls and richer timm benchmark
```

Current HEAD check:

```powershell
git rev-parse HEAD
```

Current HEAD at addendum time:

```text
bd7d4bc1f728482aa1d923d59e6a99a604be79f2
```

Current HEAD is the same commit as the pre-challenge baseline.

Post-start committed work check:

```powershell
git log --all --since="2026-06-10 09:00:00 -0700" --format="%H%x09%aI%x09%cI%x09%s"
```

Result:

```text
No commits were returned.
```

No post-start committed work currently exists in the audited git history.
Uncommitted or untracked working-tree files still require separate provenance
review before public before/after claims.

## Pre-Existing Functionality

The audited repository already had:

- `autotuneai` CLI.
- Resource budgets.
- Local, systemd, and Docker execution support.
- CPU affinity helpers.
- Run artifact storage under `.autotuneai/runs/<run_id>/`.
- Run manifests, git snapshots, and environment snapshots.
- Runtime/system/GPU tuning flows.
- Paired comparison and recommendation logic.
- Noise-band decision summaries.
- Markdown and HTML report utilities.
- Example CPU/GPU/training workloads.
- Unit tests for existing functionality.

## Phase 0 Challenge-Period Additions

Phase 0 added only planning and audit documentation under:

```text
docs/armtune/
```

No ArmTune runtime, SSH execution, llama.cpp adapter, benchmark engine,
passport generator, or measured Arm result was implemented in Phase 0.

## Phase 1 Challenge-Period Scope

Phase 1 adds ArmTune foundations only. The exact ArmTune challenge-period files
for the Phase 1 checkpoint are:

```text
AGENTS.md
autotune/cli.py
autotune/arm_tune/
docs/armtune/
tests/test_arm_tune_cli.py
tests/test_arm_tune_config.py
tests/test_arm_tune_platform.py
tests/test_arm_tune_protocol.py
tests/test_arm_tune_targets.py
```

Phase 1 implements configuration validation, versioned schemas, target
abstractions, local host inspection, structured SSH worker invocation, worker
protocol validation, `autotuneai arm doctor`, and `autotuneai arm
inspect-host`.

Phase 1 does not implement llama.cpp execution, candidate search, quality
gates, Arm Performix, build profiles, Performance Passport generation, cloud
provisioning, or measured Arm benchmark results.

## Working Tree Caveat

Before Phase 0, the working tree already contained modified and untracked files.
Those changes were not reverted and were not audited as committed
challenge-period work. Their provenance should be reviewed before any public
before/after claim.

The following files were observed as unrelated pre-existing modified files at
the Phase 1 hardening checkpoint and are not included in the ArmTune challenge
checkpoint:

```text
README.md
autotune/resource/run_state.py
autotune/resource/workload_runner.py
autotune/runtime_tuner/env.py
examples/timm_inference_benchmark.py
examples/timm_inference_benchmark_config.yaml
```

Additional untracked non-ArmTune paths observed at the same checkpoint include:

```text
.vscode/
docs/local_benchmark_matrix_2026-05-17.md
docs/validated_benchmark_results.md
examples/timm_inference_benchmark_sweep_config.yaml
examples/timm_inference_benchmark_tuner_config.yaml
tests/test_run_state.py
tests/test_runtime_env.py
```

## Significant Update Statement Draft

ArmTune is now a significant challenge-period update on top of the
pre-challenge PerfRunbench infrastructure. The challenge-period work added Arm
target validation, SSH worker execution, Arm host inspection, llama.cpp
workload adapters, benchmark artifact validation, build-profile evidence,
bounded runtime search, workload-neutral correctness hooks, Performance
Passport generation and verification, and guarded RecommendationBundle apply.

## Phase 11 Release Evidence Summary

The public release package should describe the product as PerfRunbench with
ArmTune as its Arm AI inference benchmarking and autotuning subsystem.

Default product flow:

```text
inspect Arm target
-> inspect workload runtime
-> benchmark portable baseline
-> build Arm profiles
-> search runtime candidates
-> confirm selected candidate
-> generate Performance Passport
-> validate and guarded-apply recommendation
```

Measured Phase 9 results, recorded in
`docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md`, are preliminary and
protocol-specific:

| protocol | before | after | result |
| --- | ---: | ---: | ---: |
| `llama-bench` build-profile decode median, `portable-baseline` to `arm-native` | `10.825 tok/s` | `17.530 tok/s` | profile comparison only |
| `llama-bench` runtime-search decode median | `13.630 tok/s` | `17.480 tok/s` | `+28.246515%` |
| `llama-server` TTFT median, `portable-baseline` to `arm-native` | `97.260 ms` | `62.826 ms` | server timing only |

These results must not be combined into one aggregate improvement.

Phase 9 generated a Performance Passport with:

```text
claim_eligibility_state: eligible_preliminary
verification_status: passed
```

Phase 10 validated guarded recommendation apply, recorded in
`docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md`:

```text
RecommendationBundle ID: 92b0eb69b189a06baae62f15
dry_run_status: dry_run_passed
canary_status: passed
rollback_state: rolled_back
artifact_revalidation: passed
cleanup_status: passed
```

Public claim wording:

> Best observed eligible configuration under the tested Arm target, model,
> inference workload, and measurement protocol.

Firmware-log model capability evaluation was split into BootTraceBench after
the `v0.1.0-armtune` release. It does not participate in default ArmTune
candidate eligibility in current main.
