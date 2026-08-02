# PerfRunbench

PerfRunbench is a reproducible AI benchmarking and autotuning toolkit. Its
current release focus is **ArmTune**, an Arm AI inference subsystem for:

- Arm platform inspection
- Reproducible AI benchmarking
- Build and runtime configuration search
- Statistical confirmation
- Performance Passport generation
- Guarded recommendation show, validate, dry-run apply, canary, lease, and
  rollback

ArmTune is the Arm-focused counterpart to the generic PerfRunbench resource and
recommendation infrastructure. `llama.cpp` is the first workload adapter, not
the product identity.

## ArmTune Guided Start

ArmTune provides a guided entry that keeps the detailed research controls
available without requiring a new user to understand every config field first:

```bash
autotuneai arm init --output .autotuneai/armtune-project --preset quick
autotuneai arm guide --project .autotuneai/armtune-project
autotuneai arm explain decode-heavy
```

`arm init` creates versioned `target.json`, `benchmark.json`, and
`optimize.json` starter files with strict SSH host-key checking, preserved raw
output, bounded candidate counts, confirmation runs, and a noise band. The
files are deliberately marked `result_origin: unverified` and contain only
placeholders. Replace the placeholders in a private location before accessing
a target.

`arm guide` does not connect to SSH or launch a workload. It reports missing or
placeholder fields and prints the exact sequence for target validation,
inspection, baseline measurement, and bounded optimization. Add `--json` for
automation. Use `--summary` with `arm doctor` and `arm inspect-host` for a
concise operator view while retaining the existing JSON output by default.

```bash
autotuneai arm doctor \
  --target-config .autotuneai/armtune-project/target.json \
  --require-production \
  --summary

autotuneai arm inspect-host \
  --target-config .autotuneai/armtune-project/target.json \
  --summary
```

The complete operator guide and plain-language workload glossary are in
[`docs/armtune/USER_GUIDE.md`](docs/armtune/USER_GUIDE.md) and
[`docs/armtune/GLOSSARY.md`](docs/armtune/GLOSSARY.md).

## Primary Workflow

The shortest public workflow for a new user is:

```bash
python -m pip install .
autotuneai run --auto --preset quick -- python app.py
autotuneai platform validate --preset quick --output validation.json
```

Config-free `run --auto` creates an in-memory generic command configuration,
uses process wall time as its objective, and tests only process-scoped CPU
thread settings. It writes no config unless `--save-config [PATH]` is present.
The JSON summary reports `cache_hit`, `newly_tuned`, or
`no_eligible_candidate`.

Use an explicit adapter config when the workload exposes richer metrics or
semantics:

```bash
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai doctor --config autotune.yaml -- python train.py
autotuneai run --auto --config autotune.yaml -- python train.py
```

`init` writes a public placeholder config for the selected adapter. `doctor`
performs non-destructive readiness checks without launching the workload,
tuning candidates, applying recommendations, or changing system settings. `run
--auto` fingerprints the command and environment, checks the local
RecommendationCache, retunes when policy requires it, validates/canaries the
selected process-scoped recommendation, launches the full workload, and cleans
up the apply lease. Use `autotuneai tune` when you want to tune and cache a
recommendation without launching the long workload afterward.

The `pytorch-training` adapter tunes bounded training slices with allowlisted
knob bindings. It does not require PyTorch as a core dependency and does not
rewrite user source files.

The primary product entry is:

```text
run --auto
-> platform validate
-> serving recommendations
-> ArmTune
```

```mermaid
flowchart LR
    controller[Controller CLI] --> target[Target inspection]
    target --> adapter[Workload adapter]
    adapter --> builds[Build profiles]
    builds --> search[Runtime search engine]
    search --> guard[Statistical guard]
    guard --> passport[Performance Passport]
    passport --> safety[Recommendation safety and apply]
```

Production Arm performance claims require a real Linux Arm64 target. Fixture,
synthetic, and CI-runner outputs are labeled separately and are not measured
Arm performance.

Preset search limits are `quick` (5 minutes, 3 candidates), `standard`
(20 minutes, 8 candidates), and `deep` (60 minutes, 16 candidates).
`--budget-minutes N` overrides the preset wall-clock limit without weakening
correctness, cleanup, drift, or failure guards.

## Install

PerfRunbench is not published to PyPI in this repository state. Install from a
source checkout or a locally built wheel:

```bash
python -m pip install .
```

Local wheel flow:

```bash
python -m pip install build
python -m build
python -m pip install dist/perfrunbench-<version>-py3-none-any.whl
```

Development environment:

```bash
conda env create -f environment.yml
conda activate autotuneai
python -m pip install -e .
python scripts/run_tests.py --fast
```

`environment.yml` is intentionally small. Use
`python -m pip install -e ".[benchmark]"` or `environment-benchmark.yml` for
PyTorch, ONNX, and GPU benchmark examples.

## ArmTune Quick Start

Public examples are placeholders. Keep real hostnames, identity files, model
paths, and private target configs outside git.

The commands below are the explicit advanced path. New users can create the
same bounded workflow with `autotuneai arm init` and inspect it with
`autotuneai arm guide` first.

Inspect the controller and an Arm SSH target:

```bash
autotuneai arm doctor \
  --target-config configs/armtune/target_ssh.example.json \
  --require-production

autotuneai arm inspect-host \
  --target-config configs/armtune/target_ssh.example.json
```

Run a fixed `llama-bench` baseline:

```bash
autotuneai arm benchmark \
  --config configs/armtune/llama_bench.example.json
```

Generate and inspect Arm build-profile evidence:

```bash
autotuneai arm build-doctor \
  --config configs/armtune/build_profiles.example.json

autotuneai arm build-plan \
  --config configs/armtune/build_profiles.example.json

autotuneai arm tune-build \
  --config configs/armtune/build_profiles.example.json
```

Run bounded runtime optimization:

```bash
autotuneai arm optimize \
  --config configs/armtune/llama_bench_optimize.example.json \
  --max-candidates 12 \
  --affinity-modes scheduler-default,compact,spread,reserve-one-core
```

Compare profiles and generate a Passport:

```bash
autotuneai arm compare-builds \
  --config configs/armtune/profile_comparison.example.json

autotuneai arm passport \
  --comparison-result .autotuneai/runs/<run_id>/armtune/profile_comparison/profile_comparison_result.json

autotuneai arm verify-passport \
  --passport .autotuneai/runs/<run_id>/armtune/passport/performance_passport.json
```

Show, validate, and dry-run a guarded recommendation:

```bash
autotuneai recommendation show \
  --bundle .autotuneai/recommendations/<bundle_id>.json

autotuneai recommendation validate \
  --bundle .autotuneai/recommendations/<bundle_id>.json

autotuneai recommendation apply \
  --bundle .autotuneai/recommendations/<bundle_id>.json \
  --scope process \
  --dry-run
```

For a canary apply with rollback support:

```bash
autotuneai recommendation apply \
  --bundle .autotuneai/recommendations/<bundle_id>.json \
  --scope process \
  --lease-minutes 30 \
  --canary-runs 1

autotuneai recommendation status \
  --bundle .autotuneai/recommendations/<bundle_id>.json

autotuneai recommendation rollback \
  --bundle .autotuneai/recommendations/<bundle_id>.json
```

## Serving Autotuning

Workload-aware serving tuning is separate from the fixed single-request
`llama-bench` protocol. v0.3 packages the validated `llama-server`
remote-client recommendation flow while preserving fixture mode for public CI:

```bash
autotuneai serve doctor --json
autotuneai serve workload show interactive-short --json
autotuneai serve plan \
  --workload interactive-short \
  --traffic closed-loop \
  --concurrency 4
autotuneai serve search \
  --workload interactive-short \
  --candidate-cap 8 \
  --fixture
autotuneai serve pareto --experiment results/serving/search
autotuneai serve network-plan
autotuneai serve recommend \
  --workload interactive-short \
  --concurrency 4 \
  --recommendation-class maximum_throughput_remote
autotuneai serve run \
  --best \
  --workload interactive-short \
  --concurrency 4 \
  --dry-run
```

`serve run --best` loads an existing validated serving recommendation and does
not search. The public dry-run path reports the selected profile, expected
command shape, drift checks, memory/context preflight, port preflight, canary
requirement, and cleanup plan without launching a private server.

Validated remote serving recommendation classes:

- `maximum_throughput_remote`
- `interactive_latency_remote`

No balanced, shared-prefix, memory-constrained, or remote-network tuning
recommendation is created without qualifying evidence. PerfRunbench does not
collapse unrelated serving metrics into one universal score.

Validated remote serving evidence:

| workload | baseline output tok/s | `slots4-cont-on` output tok/s | throughput delta | baseline TTFT p95 ms | `slots4-cont-on` TTFT p95 ms | TTFT p95 reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `interactive-short` | `17.032` | `31.355` | `+84.092%` | `3055.682` | `271.088` | `91.128%` |
| `decode-heavy` | `17.046` | `31.790` | `+86.490%` | `10126.795` | `202.424` | `98.001%` |
| `prefill-heavy` | `16.563` | `28.054` | `+69.375%` | `3967.424` | `341.693` | `91.388%` |

The validated topology was a C7g / Neoverse-V1 server and a T4g /
Neoverse-N1-class remote client over a direct private network path at
concurrency 4. The run recorded `608` measured requests, zero retries, zero
request failures, zero quality failures, passed artifact validation, and
dual-host cleanup.

Network tuning is explicitly not recommended:

```text
network_tuning_status: network_tuning_not_executed_evidence_absent
```

Detailed serving evidence and demo material:

```text
docs/results/VALIDATED_PERFORMANCE_SUMMARY.md
docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md
docs/demo/SERVING_AUTOTUNING_DEMO.md
docs/demo/SERVING_AUTOTUNING_COMMANDS.md
docs/demo/SERVING_AUTOTUNING_TALK_TRACK.md
```

## Same-Protocol C7g Size Scaling

A separate measured experiment used the same pinned Qwen2.5-7B Q4_K_M model,
llama.cpp runtime bundle, binaries, CPU-allocation rule, and concurrency-4
loopback protocol on three AWS Graviton3 sizes:

| Target | Logical CPUs | Server threads | Decode baseline tok/s | Tuned tok/s | Tuned delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c7g.xlarge` | 4 | 3 | 6.237 | 12.136 | +94.585% |
| `c7g.2xlarge` | 8 | 7 | 13.626 | 25.876 | +89.901% |
| `c7g.4xlarge` | 16 | 15 | 25.475 | 46.186 | +81.296% |

The same four-slot continuous-batching family also improved formally confirmed
`interactive-short` throughput by `+73.324%` to `+84.866%`. It was not
recommended for `prefill-heavy`: screening regressed by `-35.310%` to
`-38.977%` and triggered the TTFT guard on all three sizes.

![C7g size scaling and guarded serving](docs/results/c7g-size-scaling/charts/serving_before_after_by_size.svg)

Across the three targets, the protocol recorded `1,047` measured HTTP
requests, `30` fresh server starts, zero request or quality failures, complete
cleanup, and `514` successful controller-side artifact digest checks after
download.

These percentages describe runtime tuning within each target. They are not
added to hardware-size scaling, are not remote-network evidence, and are not a
universal Arm or llama.cpp speedup claim.

Detailed methodology, sanitized JSON/CSV data, charts, and the report are in:

```text
docs/results/C7G_SIZE_SCALING_VALIDATION.md
docs/results/c7g-size-scaling/
```

## Larger Arm Capacity And Workload-Aware Guards

A separate measured C7g.4xlarge experiment validates thread scaling and
full-capacity serving behavior on 16 logical Neoverse-V1 CPUs. It uses a pinned
Qwen2.5-7B Q4_K_M artifact, a pinned fresh llama.cpp build, one reserved local
client CPU, and 15 server CPUs.

The separate `llama-bench` pp512/tg128 protocol scaled from 1 to 15 threads:

| Metric | 1 thread | 15 threads | Speedup | 15-thread efficiency |
| --- | ---: | ---: | ---: | ---: |
| prefill | `6.761 tok/s` | `73.669 tok/s` | `10.897x` | `72.6%` |
| decode | `2.759 tok/s` | `30.089 tok/s` | `10.907x` | `72.7%` |

Full-capacity loopback concurrency-8 paired confirmation produced a
workload-scoped result:

| Workload | Baseline output tok/s | `slots4-cont-on` output tok/s | Delta | Result |
| --- | ---: | ---: | ---: | --- |
| `decode-heavy` | `25.779` | `46.488` | `+80.331%` | eligible |
| `interactive-short` | `25.600` | `43.712` | `+70.749%` | eligible |
| `prefill-heavy` | `19.734` | `15.585` | `-21.025%` | rejected |

The same run preserved two eight-slot long-prefill screening slices with HTTP
400 context-limit failures and skipped full-capacity concurrency 4 because no
candidate passed every screening guard. It recorded 769 measured HTTP
requests, 17 fresh server starts, passed cleanup, and independently validated
278 artifacts after download.

![C7g.4xlarge serving before and after](docs/results/c7g4xlarge-realistic/charts/serving_before_after.svg)

This is loopback evidence, not remote-network evidence. The hardware-scaling
and runtime-tuning percentages are separate and are not combined. Detailed
methodology, CSV data, charts, and the sanitized HTML report are in:

```text
docs/results/C7G4XLARGE_REALISTIC_ARM_INFERENCE_VALIDATION.md
docs/results/c7g4xlarge-realistic/
```

## Real Arm Evidence Snapshot

The current measured Arm evidence is a preliminary single-target result on a
user-provisioned Linux Arm64 C7g / Neoverse-V1 target using Qwen2.5-7B Q4_K_M
and a pinned llama.cpp source commit. It is not a global optimum and not a
cross-platform claim.

`llama-bench` build-profile comparison, fixed protocol:

| profile | decode median |
| --- | ---: |
| `portable-baseline` | `10.825 tok/s` |
| `arm-native` | `17.530 tok/s` |
| `arm-optimized` | `17.380 tok/s` |

`llama-bench` bounded runtime search, separate protocol:

| item | decode median |
| --- | ---: |
| search baseline | `13.630 tok/s` |
| selected candidate | `17.480 tok/s` |
| observed runtime-search improvement | `28.246515%` |

`llama-server` secondary latency evidence, separate protocol:

| profile | TTFT median |
| --- | ---: |
| `portable-baseline` | `97.260 ms` |
| `arm-native` | `62.826 ms` |

Performance Passport status:

```text
claim_eligibility_state: eligible_preliminary
verification_status: passed
guarded_apply_validation: passed
```

guarded apply validation passed.

These protocol-specific results must not be combined into a single aggregate
speedup. The supported wording is:

> Best observed eligible configuration under the tested Arm target, model,
> inference workload, and measurement protocol.

For CPU-aware native builds the claim boundary is narrower:

> Best observed eligible configuration under the tested CPU, model, runtime
> source commit, compiler, and measurement protocol.

Native binaries keep a separate portability scope. A CPU-specific binary
recommendation must validate architecture, required CPU features, CPU model
policy, binary digest, and startup smoke before reuse on another target.
KleidiAI is reported as `enabled_and_evidenced` only when both build evidence
and runtime backend evidence are present.

Detailed evidence is recorded in:

```text
docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md
docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md
```

## Low-Resource Arm Training Evidence

A separate measured T4g.small protocol validates the public CPU-only synthetic
PyTorch adapter, bounded search, failed-candidate preservation, paired
confirmation, cache reuse, budget exhaustion, rollback, and visual evidence
export.

| Metric | Baseline | Selected |
| --- | ---: | ---: |
| median throughput | `14,430.040 samples/s` | `33,274.800 samples/s` |
| median step time | `2.218 ms` | `0.962 ms` |

```text
paired confirmations: 12
ratio-of-medians improvement: 130.593958%
median paired improvement: 128.172176%
selected variability: 3.360982%
failed DataLoader-worker trials preserved: 4
```

![T4g paired training confirmation](docs/results/t4g-training-visual/charts/paired_before_after.png)

This is one burstable T4g.small and one public synthetic CPU PyTorch
measurement protocol. It is not a production training benchmark, must not be
compared directly with C7g, and is not a general PyTorch or Arm speedup claim.
The CPU credit and surplus state is recorded with the result.

Detailed evidence and reusable chart data:

```text
docs/results/T4G_TRAINING_VISUAL_EVIDENCE.md
docs/results/t4g-training-visual/
```

Phase 19 CPU-aware build tuning design documents:

```text
docs/CPU_AWARE_BUILD_TUNING.md
docs/NATIVE_BINARY_PORTABILITY.md
```

## Advanced Guarded System Experiments

Linux host, kernel, boot, and reboot-required profiles are opt-in guarded
experiments. They are not part of normal `run --auto`, and no public kernel
recommendation currently qualifies for general reuse.

Phase 21A Linux system and kernel profile planning documents:

```text
docs/LINUX_SYSTEM_AUTOTUNING.md
docs/KERNEL_PROFILE_AUTOTUNING.md
docs/HOST_STATE_ROLLBACK.md
docs/SYSTEM_BENCHMARK_PROTOCOLS.md
docs/ARM_4K_64K_KERNEL_COMPARISON.md
```

The Phase 21A planning and fake-root validation artifacts do not modify a real
host and do not create performance claims. Later real validations are
documented separately under `docs/results/`.

## Related Historical Experiment

BootTraceBench is maintained separately at
`https://github.com/xuwwwww/BootTraceBench` as an experimental firmware-log
model evaluation project. It is not part of PerfRunbench tuning or
recommendation eligibility. The `v0.1.0-armtune` tag preserves the former
integrated BootTrace state for historical review.

## Generic PerfRunbench Workflows

The repository also includes generic PerfRunbench resource-guard and workload
tuning commands:

```bash
autotuneai run --auto --preset quick -- python app.py
autotuneai platform validate --preset quick --output validation.json
autotuneai inspect
autotuneai executors --probe-systemd --probe-docker --check-sudo-cache
autotuneai doctor --config autotune.yaml -- python train.py
autotuneai run --auto --config autotune.yaml -- python train.py
autotuneai tune --config configs/examples/tune_command.yaml -- <command that writes metrics.json>
autotuneai run --best --config configs/examples/run_command.yaml -- <command that matches the cached identity>
autotuneai cache list
autotuneai run -- python examples/dummy_train.py
autotuneai tune-system
autotuneai tune-runtime --profile runtime-pytorch-max-performance
autotuneai optimize -- python examples/gpu_training_pressure.py --config examples/gpu_training_pressure_config.yaml
autotuneai report-comparison --input results/reports/tuning_comparison.json
```

Use `autotuneai restore --active` if an interrupted run left reversible
runtime tuning state in `.autotuneai/active_tuning_state.json`.

## Development Validation

Fast validation:

```bash
python scripts/run_tests.py --fast
```

Focused ArmTune validation:

```bash
python -m unittest discover -s tests -p "test_arm_tune*.py" -v
```

Compile check:

```bash
python -m compileall autotune
```

The public test suite does not require Arm hardware, SSH access, model files,
root privileges, cloud credentials, or network access.

## Repository Layout

```text
autotune/
  golden_path/    Command identity, metric-source parsing, bounded tuning,
                  recommendation cache lookup, and automatic reuse policy.
  platform_validation.py
                  Sanitized OS, architecture, accelerator, and installed
                  runtime compatibility validation.
  training_adapter/
                  PyTorch training adapter config validation, safe knob
                  bindings, bounded slices, effective-batch guards, and
                  training metric parsing.
  training_metrics.py
                  Dependency-free helper for structured training progress and
                  result records.
  arm_tune/       Arm target inspection, llama.cpp adapter, search, profiles,
                  Passport, and workload-neutral correctness hooks.
  serving/        Workload-aware llama-server capability discovery, traffic
                  profiles, deterministic load generation, Pareto search,
                  network plans, and serving recommendation guards.
  recommendation/ Generic RecommendationBundle generation, validation, guarded
                  apply, canary, lease, promote, and rollback.
  resource/       Run directories, process execution, monitoring, CPU affinity,
                  resource budgets, and lifecycle cleanup.
  report/         Markdown and HTML report helpers.
  runtime_tuner/  Process-level runtime environment profiles.
  system_tuner/   Reversible host tuning profiles.
configs/          Public placeholder configs and benchmark examples.
docs/armtune/     Architecture, methodology, evidence, limitations, Devpost,
                  and demo packaging.
tests/            Unit, fixture, and fake integration tests.
```

## Current Status

PerfRunbench v0.3.1 includes real C7g Arm64 build/runtime evidence, a verified
eligible preliminary Performance Passport, guarded process-scoped apply
validation, and real C7g server plus T4g remote-client serving evidence.
Protocol boundaries and release limitations are documented in
`docs/results/VALIDATED_PERFORMANCE_SUMMARY.md` and
`docs/armtune/LIMITATIONS.md`.

## License

PerfRunbench is available under the [MIT License](LICENSE).
