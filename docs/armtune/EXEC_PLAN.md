# ArmTune Execution Plan

## Phase 0 Status

Status: completed planning audit, pending user approval for Phase 1.

ArmTune is planned as an Arm-first subsystem inside PerfRunbench. The parent
repository remains PerfRunbench and existing `autotuneai` commands must remain
backward compatible.

## Current Repository Snapshot

- Branch: `main`
- HEAD: `bd7d4bc1f728482aa1d923d59e6a99a604be79f2`
- HEAD date: 2026-05-15 18:12:04 +0800
- HEAD subject: `Add advanced tuning controls and richer timm benchmark`
- Official challenge submission period found in the Devpost rules:
  2026-06-10 09:00 Pacific Time through 2026-08-14 16:00 Pacific Time.
- Pre-challenge baseline: current HEAD is before the official submission
  period. Existing committed PerfRunbench functionality must be treated as
  reused infrastructure, not new ArmTune challenge work.
- Working tree before Phase 0 already contained modified and untracked files.
  These were treated as user progress and not reverted.

## Phase 0 Addendum Governance

Commands:

```powershell
Test-Path -LiteralPath AGENTS.md
git ls-files --error-unmatch AGENTS.md
git status --short
git diff --stat
git branch --show-current
git rev-parse HEAD
```

Findings:

- `AGENTS.md` exists at the repository root.
- `AGENTS.md` is not currently tracked by git; `git ls-files --error-unmatch
  AGENTS.md` exited with code 1 and reported that the path is not known to git.
- Current branch is `main`.
- Current HEAD is `bd7d4bc1f728482aa1d923d59e6a99a604be79f2`.
- The working tree contains pre-existing modified and untracked files outside
  the ArmTune addendum.
- No destructive git operation was executed.
- No paid cloud resource was created, started, stopped, resized, or used.

## Fast Test Results

Pre-documentation command:

```powershell
python scripts\run_tests.py --fast
```

Result:

- 138 tests ran.
- Status: OK.
- Duration reported by unittest: 10.878 seconds.

This test run was executed before adding the Phase 0 ArmTune documentation.

Post-documentation command:

```powershell
python scripts\run_tests.py --fast
```

Result:

- 138 tests ran.
- Status: OK.
- Duration reported by unittest: 8.010 seconds.

This second test run was executed after adding the Phase 0 documentation.

Final verification command after adding the command log:

```powershell
python scripts\run_tests.py --fast
```

Result:

- 138 tests ran.
- Status: OK.
- Duration reported by unittest: 7.701 seconds.

## Repository Areas Inspected

- `pyproject.toml`
- `autotune/cli.py`
- `autotune/backends/`
- `autotune/profiler/`
- `autotune/resource/`
- `autotune/tuner/`
- `autotune/recommendation/`
- `autotune/report/`
- `autotune/runtime_tuner/`
- `autotune/system_tuner/`
- `autotune/utils/`
- `scripts/`
- `tests/`
- `configs/`
- `examples/`
- `README.md`
- `USER_GUIDE.md`
- `docs/architecture.md`
- `docs/methodology.md`
- `docs/validated_benchmark_results.md`
- `docs/local_benchmark_matrix_2026-05-17.md`
- Git status and commit history around the challenge start date.

## Phase 0 Command Log

The following terminal commands were executed during Phase 0:

```powershell
Get-Content -LiteralPath AGENTS.md
Get-Content -LiteralPath D:\CodexHome\attachments\b0b17aaa-ed18-4a0c-a475-318e6f83cab8\pasted-text.txt
Select-String -LiteralPath D:\CodexHome\memories\MEMORY.md -Pattern 'AutoTuneAI|ArmTune|PerfRunbench|optimize-performance|launch-performance' -Context 2,3
git status --short --branch
git rev-parse HEAD
git branch --show-current
rg --files
Get-ChildItem -LiteralPath docs\armtune -Force -ErrorAction SilentlyContinue
Get-Content -LiteralPath pyproject.toml
Get-Content -LiteralPath autotune\cli.py
Get-Content -LiteralPath README.md -TotalCount 220
Get-Content -LiteralPath USER_GUIDE.md -TotalCount 220
rg -n "autotuneai|\.autotuneai|runs|run_id|baseline|noise|within-noise|within_noise|paired|signal|SIGINT|cleanup|atexit|resource|affinity|cgroup|systemd|docker|report|compare" autotune tests scripts docs pyproject.toml README.md USER_GUIDE.md
python scripts\run_tests.py --fast
git log --oneline --decorate --date=short --pretty=format:"%h %ad %d %s" -n 30
git log --oneline --decorate --date=iso --pretty=format:"%H%x09%ad%x09%s" --before="2026-06-10 09:00:00 -0700" -n 5
git log --oneline --decorate --date=iso --pretty=format:"%H%x09%ad%x09%s" --since="2026-06-10 09:00:00 -0700" --reverse -n 10
git diff --stat
Get-ChildItem -LiteralPath autotune -Directory | Select-Object -ExpandProperty Name
Get-Content -LiteralPath scripts\run_tests.py
Get-Content -LiteralPath autotune\resource\run_state.py
Get-Content -LiteralPath autotune\resource\workload_runner.py -TotalCount 260
Get-Content -LiteralPath autotune\recommendation\optimizer.py -TotalCount 320
Get-Content -LiteralPath autotune\report\run_report.py -TotalCount 240
Get-Content -LiteralPath autotune\resource\comparison_runner.py -TotalCount 260
rg -n "noise|decision|thermal|paired|baseline|DEFAULT_NOISE|meaningful|within" autotune\recommendation\optimizer.py tests\test_optimizer.py docs\validated_benchmark_results.md
rg -n "def _interrupt_restore_guard|def _terminate_process_tree|def _tuning_lifecycle|restore|SIGINT|SIGTERM|finally|write_json|atomic" autotune\resource\workload_runner.py autotune\resource\run_state.py autotune\system_tuner\runtime.py autotune\source_tuner\transaction.py
rg -n "def build_parser|subparsers|add_parser|set_defaults\(handler" autotune\cli.py
rg -n "write_json|report|html|markdown|comparison|generate" autotune\report autotune\resource autotune\recommendation scripts tests | Select-Object -First 180
Get-ChildItem -LiteralPath tests -File | Select-Object -ExpandProperty Name
Get-Content -LiteralPath docs\architecture.md -TotalCount 240
Get-Content -LiteralPath docs\methodology.md -TotalCount 240
Get-Content -LiteralPath docs\validated_benchmark_results.md -TotalCount 160
Get-Content -LiteralPath docs\local_benchmark_matrix_2026-05-17.md -TotalCount 160
Get-Content -LiteralPath autotune\resource\budget.py -TotalCount 240
Get-Content -LiteralPath autotune\resource\affinity.py -TotalCount 200
Get-Content -LiteralPath requirements.txt
Get-Content -LiteralPath requirements-core.txt
Get-Content -LiteralPath requirements-dev.txt
Get-ChildItem -LiteralPath configs -Force | Select-Object Name,Length
Get-ChildItem -LiteralPath examples -Force | Select-Object Name,Length | Format-Table -AutoSize
Get-Content -LiteralPath autotune\__init__.py
rg -n "yaml|safe_load|toml|json|config" autotune tests scripts | Select-Object -First 160
rg -n "subprocess\.run|Popen|shell=True|timeout" autotune scripts tests | Select-Object -First 160
Get-Content -LiteralPath autotune\utils\config.py
Get-Content -LiteralPath autotune\profiler\hardware_info.py -TotalCount 240
New-Item -ItemType Directory -Force -Path docs\armtune | Out-Null
python scripts\run_tests.py --fast
git diff --stat
git status --short --branch
rg -n "measured Arm|No measured|unverified|Phase 0|bd7d4bc|2026-06-10|autotuneai arm|KleidiAI" docs\armtune
python scripts\run_tests.py --fast
Get-ChildItem -LiteralPath docs\armtune -File | Select-Object -ExpandProperty Name
git status --short --branch
git status --short docs\armtune
```

Official source pages were also opened for the challenge rules, Arm Performix,
llama.cpp, llama-bench, and KleidiAI; URLs are recorded in
`RESEARCH_NOTES.md`.

## Existing Reusable Components

- CLI entry point: `autotuneai = autotune.cli:main`.
- Run artifact root: `.autotuneai/runs/<run_id>/`, created by
  `autotune.resource.run_state`.
- Run provenance: git status, diff, HEAD, command, environment, manifest, and
  process state are already captured for PerfRunbench runs.
- Resource execution: `autotune.resource.workload_runner` supports local,
  systemd, and Docker execution through existing PerfRunbench abstractions.
- Lifecycle cleanup: workload runner restores system and GPU tuning in `finally`
  blocks and handles SIGINT/SIGTERM interruption through an interrupt guard.
- CPU and memory controls: `ResourceBudget`, `apply_cpu_affinity`, cgroup
  monitoring, systemd scopes, and Docker limits.
- Executor capability detection: `autotune.resource.executor_capabilities`.
- Comparison utilities: `autotune.resource.comparison_runner`.
- Paired baseline and noise-band logic: `autotune.recommendation.optimizer`
  contains rotating/interleaved trials, thermal-controlled pairs, and the
  default `+/-2%` noise-band decision summary.
- Generic report utilities: `autotune.report.run_report`,
  `autotune.report.comparison_report`, and `autotune.report.charts`.
- Configuration loader: `autotune.utils.config` already supports JSON and YAML,
  with a simple YAML fallback when PyYAML is unavailable.
- Tests: existing unittest suite covers CLI, resource budgets, executor
  construction, reports, comparison, run state recovery, runtime tuning, and
  system tuning.

## Existing Modules That Should Remain Generic

These modules should not receive Arm-specific business logic unless a small,
generic abstraction is clearly needed by PerfRunbench and ArmTune:

- `autotune.resource.*`
- `autotune.report.*`
- `autotune.recommendation.optimizer`
- `autotune.runtime_tuner.*`
- `autotune.system_tuner.*`
- `autotune.utils.config`
- `autotune.cli` outside of a thin `arm` command-group dispatcher

## Proposed Package Boundary

Create ArmTune under:

```text
autotune/arm_tune/
```

Proposed first-phase packages:

```text
autotune/arm_tune/
  __init__.py
  cli.py
  config.py
  schemas.py
  errors.py
  targets/
  platform/
```

Later phases add `builds/`, `adapters/`, `benchmark/`, `search/`, `quality/`,
`profiling/`, `passport/`, `ci/`, and `report/`.

## CLI Integration Proposal

Keep `autotune/cli.py` as the top-level entry point. Add one isolated
`autotuneai arm` command group with a thin dispatcher to
`autotune.arm_tune.cli`.

Phase 1 commands:

```text
autotuneai arm doctor
autotuneai arm inspect-host
```

Later commands:

```text
autotuneai arm inspect-build
autotuneai arm benchmark
autotuneai arm optimize
autotuneai arm profile
autotuneai arm passport
autotuneai arm compare
autotuneai arm ci-check
autotuneai arm demo
```

## Remote Target Strategy

ArmTune uses a controller/target model:

- Controller can run on Windows, WSL, Linux, or macOS on x86_64 or Arm64.
- Production benchmarks require a real Linux Arm64 target.
- Initial targets are local Arm64 Linux and remote Arm64 Linux over SSH.
- The controller uploads a versioned JSON job specification and invokes a fixed
  worker entry point on the target.
- The worker validates arguments and writes versioned JSON results plus raw
  artifacts.
- The controller downloads and validates target artifacts.

No automatic cloud provisioning is planned for the MVP.

## Dependency Decision

Phase 1 should use only the Python standard library plus existing dependencies
unless a dependency is already present in the environment.

- Use the existing JSON/YAML loader style for controller-side config.
- Do not add cloud SDKs.
- Do not require Paramiko in Phase 1; prefer the system OpenSSH client first,
  with strict host-key verification by default.
- Do not require Arm Performix; keep it optional for Phase 5.
- Do not require PyTorch, ONNX, or model runtime dependencies for ArmTune
  foundations.

## Run Artifact Plan

ArmTune artifacts should be nested inside the existing run store:

```text
.autotuneai/runs/<run_id>/armtune/
```

This prevents a second run lifecycle from bypassing existing manifest,
provenance, restore, and report conventions.

## Missing Or Unsafe Abstractions

- `write_json` is not atomic today. ArmTune important manifests should either
  add an atomic generic writer or use an ArmTune-local atomic writer until a
  generic improvement is justified.
- Existing timestamps are local time for PerfRunbench runs. ArmTune serialized
  schemas should use UTC timestamps.
- Existing `autotune/cli.py` is a large single file. ArmTune should not expand
  it broadly; only add the minimal dispatcher.
- Current hardware inspection is generic and controller-local. ArmTune needs
  target-side Linux Arm64 inspection that keeps host capability separate from
  workload execution evidence.
- Existing benchmark docs contain non-Arm results and must not be reused as
  Arm competition measurements.

## Major Risks

- Accidentally presenting x86 controller data as Arm evidence.
- Claiming optimized Arm execution from host feature presence alone.
- Comparing incompatible experiments across model hashes, prompts, token counts,
  binary commits, or CPU allocations.
- SSH command construction becoming unsafe or leaking identity-file paths or
  secrets.
- Adding Arm-specific logic to generic PerfRunbench modules.
- Creating benchmark claims before Phase 6 real Arm validation.

## Phase 1 Proposal

Phase 1 should implement:

- `autotune/arm_tune/schemas.py`
- `autotune/arm_tune/config.py`
- `autotune/arm_tune/errors.py`
- `autotune/arm_tune/targets/base.py`
- `autotune/arm_tune/targets/local.py`
- `autotune/arm_tune/targets/ssh.py`
- `autotune/arm_tune/targets/protocol.py`
- `autotune/arm_tune/targets/worker.py`
- `autotune/arm_tune/platform/architecture.py`
- `autotune/arm_tune/platform/cpu_identity.py`
- `autotune/arm_tune/platform/cpu_features.py`
- `autotune/arm_tune/platform/topology.py`
- `autotune/arm_tune/platform/evidence.py`
- `tests/test_arm_tune_*.py`

Focus only on configuration validation, target abstraction, local Arm64
rejection/acceptance rules, SSH command construction tests, host inspection
parsers, and `autotuneai arm doctor` / `inspect-host`.

Do not implement llama.cpp execution, search, passport, report, or Performix in
Phase 1.

## Phase 1 Hardening Checkpoint

Status: implemented and ready for selective checkpoint after packaging and test
validation.

Phase 1 hardening scope:

- Preserve existing `autotuneai` command compatibility through a thin
  `autotuneai arm` dispatcher.
- Keep the controller valid on Windows, WSL, Linux, or macOS on x86_64 or
  Arm64.
- Require a configured Linux Arm64 target before reporting production benchmark
  readiness.
- Separate controller diagnostics, local target support, configured target
  readiness, and production benchmark readiness in `autotuneai arm doctor`.
- Keep host capabilities separate from workload execution evidence.
- Use versioned top-level JSON structures.
- Use ArmTune-local atomic JSON writes for worker job/result manifests.
- Use system OpenSSH with strict host-key checking by default.
- Invoke the remote worker through structured fields:
  `remote_python`, `remote_module`, and `remote_workdir`.
- Reject arbitrary remote worker command fields.
- Avoid `shell=True`.
- Avoid logging identity-file contents or writing them into serialized ArmTune
  job artifacts.

The expected conceptual state on a Windows AMD64 controller with no configured
Arm target is:

```text
controller_status: ready
local_target_status: unsupported
configured_target_status: not_configured
production_benchmark_ready: false
```

`autotuneai arm doctor` may return exit code 0 when diagnostics complete even
when `production_benchmark_ready` is false. `--require-production` returns a
non-zero exit code when no configured Linux Arm64 production target is ready.

## Phase 2A Plan: Minimal llama-bench Vertical Slice

Status: implementation in progress after the Phase 2A plan checkpoint.

Phase 2A implements only the smallest end-to-end llama-bench path needed to
prove the controller/worker protocol, artifact handling, and defensive parsing
shape.

Allowed scope:

- Inspect an existing `llama-bench` binary.
- Capture binary version, help output, and system information.
- Record binary SHA-256.
- Validate a supplied model path.
- Record model SHA-256.
- Use a fixed benchmark configuration.
- Send a controller-to-worker JSON benchmark job.
- Execute through `LocalArmTarget` and `SshArmTarget`.
- Preserve raw stdout and stderr artifacts.
- Implement a defensive `llama-bench` parser.
- Capture prefill throughput.
- Capture decode throughput.
- Record trial status and parse status separately.
- Write a versioned benchmark result.
- Download result manifests and artifacts from SSH targets.
- Add `autotuneai arm benchmark`.
- Add fake executable integration tests.

Out of scope for Phase 2A:

- `llama-server`.
- HTTP or streaming benchmarks.
- Time to first token.
- Concurrency.
- Candidate search.
- Quality gates.
- Build profiles.
- Arm Performix.
- Performance Passport generation.
- Automatic model download.
- Cloud provisioning.

Planned file additions:

```text
autotune/arm_tune/benchmark/__init__.py
autotune/arm_tune/benchmark/llama_bench.py
autotune/arm_tune/benchmark/parser.py
autotune/arm_tune/benchmark/artifacts.py
autotune/arm_tune/benchmark/hashes.py
tests/test_arm_tune_benchmark.py
tests/test_arm_tune_llama_bench_parser.py
tests/fixtures/arm_tune/fake_llama_bench.py
```

Planned file updates:

```text
autotune/arm_tune/schemas.py
autotune/arm_tune/cli.py
autotune/arm_tune/targets/base.py
autotune/arm_tune/targets/local.py
autotune/arm_tune/targets/ssh.py
autotune/arm_tune/targets/protocol.py
autotune/arm_tune/targets/worker.py
docs/armtune/BENCHMARK_METHODOLOGY.md
docs/armtune/LIMITATIONS.md
docs/armtune/RESULTS_TEMPLATE.md
```

Phase 2A test requirements:

- Tests must use fake executables and synthetic fixture output only.
- Tests must not require Arm hardware, an SSH server, network access, a real
  model, root privileges, Arm Performix, or cloud credentials.
- Parser failures must produce explicit parse status, not zero-valued metrics.
- Host capability evidence and workload execution evidence must remain separate
  schemas.

Phase 2A completion criteria:

- `autotuneai arm benchmark --config <path>` runs one fixed llama-bench
  invocation through the versioned worker protocol.
- The worker preserves raw inspection and benchmark stdout/stderr artifacts.
- The final result records separate validation, execution, parse, and benchmark
  statuses.
- The CLI returns documented exit codes and prints `Not available` for missing
  metrics.
- Fake executable integration tests exercise the public CLI without Arm
  hardware, external SSH, a real model, real llama.cpp, internet access, root,
  Arm Performix, or cloud credentials.

## Phase 2B Plan: Native Arm64 CI Validation and Benchmark Pipeline Hardening

Status: completed. The GitHub-hosted native Arm64 workflow was executed
successfully after the Phase 2B implementation was pushed.

Phase 2B validates that the Phase 2A benchmark slice can run its fixture smoke
path on a GitHub-hosted native Arm64 Linux runner while hardening benchmark
provenance, schemas, parser behavior, and artifact trust boundaries.

Allowed scope:

- GitHub-hosted native Arm64 smoke validation.
- Linux Arm64 package and CLI validation.
- Fixture benchmark execution on a native Arm64 runner.
- Result-origin integrity.
- Artifact and digest hardening.
- Schema round-trip hardening.
- Parser robustness.
- Real-Arm integration-test scaffolding that is skipped by default.

Out of scope for Phase 2B:

- Real model benchmark.
- AWS or any other cloud provisioning.
- External SSH server validation.
- `llama-server`.
- Time to first token.
- Candidate search.
- Quality gate.
- Build profiles.
- Arm Performix.
- Performance Passport generation.
- Paid infrastructure actions.

Planned file additions:

```text
.github/workflows/armtune-arm64-smoke.yml
autotune/arm_tune/benchmark/artifact_validation.py
autotune/arm_tune/benchmark/provenance.py
scripts/run_armtune_real_smoke.py
tests/test_arm_tune_benchmark_origin.py
tests/test_arm_tune_benchmark_schema_roundtrip.py
tests/test_arm_tune_artifact_validation.py
tests/fixtures/arm_tune/llama_bench_outputs/*.txt
```

Planned file updates:

```text
autotune/arm_tune/benchmark/artifacts.py
autotune/arm_tune/benchmark/llama_bench.py
autotune/arm_tune/benchmark/parser.py
autotune/arm_tune/config.py
autotune/arm_tune/schemas.py
autotune/arm_tune/targets/local.py
autotune/arm_tune/targets/ssh.py
docs/armtune/ARCHITECTURE.md
docs/armtune/BENCHMARK_METHODOLOGY.md
docs/armtune/LIMITATIONS.md
docs/armtune/REMOTE_TARGETS.md
```

Phase 2B completion criteria:

- A GitHub Actions workflow exists for native Arm64 fixture smoke validation on
  `ubuntu-24.04-arm`.
- The workflow runs only for ArmTune-relevant path changes and manual dispatch.
- The workflow validates architecture, package installation, CLI help, doctor,
  inspect-host, focused ArmTune tests, and a public CLI fake llama-bench E2E.
- The fixture E2E remains `result_origin: test_fixture` even on native Arm64.
- Controller-side artifact validation checks downloaded manifests before a
  benchmark can be reported successful.
- Parser ambiguity, invalid numeric values, duplicate rows, and stderr-only
  metrics have explicit outcomes and do not become zero-valued metrics.
- Real-Arm smoke scaffolding is available but skipped by default and requires
  explicit user-supplied binary/model paths plus an acknowledgement flag.
- No real model, external SSH server, AWS/cloud resource, Performix, candidate
  search, quality gate, build profile, or Performance Passport is implemented.

Phase 2B validation evidence:

```text
Workflow: ArmTune Native Arm64 Smoke
Workflow file: .github/workflows/armtune-arm64-smoke.yml
GitHub run URL: https://github.com/xuwwwww/PerfRunbench/actions/runs/29270842566
GitHub run ID: 29270842566
GitHub job URL: https://github.com/xuwwwww/PerfRunbench/actions/runs/29270842566/job/86887839990
GitHub job ID: 86887839990
Event: push
Conclusion: success
Started: 2026-07-13T17:32:35Z
Tested commit: 2074dfac80e699ba22178774b28c66e79afae475
Runner OS: Linux
Runner architecture: aarch64
Kernel: 6.17.0-1018-azure
Uploaded artifact name: armtune-arm64-smoke-artifacts
Artifact run ID: 20260713_173254_650848
Focused ArmTune suite at tested commit: 94 tests
Downloaded artifact digest validation: passed, 13 artifacts checked
Benchmark result_origin: test_fixture
Benchmark parse_status: parsed
Benchmark status: passed
```

The fixture metrics parsed from the fake llama-bench output were
`321.5 tokens/s` for prefill and `87.25 tokens/s` for decode. These are
fixture constants and are not Arm performance results.

This evidence supports the following claim only:

> ArmTune's controller, worker, provenance, parser, and artifact-validation
> pipeline was validated natively on a GitHub-hosted Linux Arm64 runner.

This evidence does not show real llama.cpp execution, real model inference,
KleidiAI usage, candidate search, quality validation, Performix support, or an
ArmTune optimization result.

## Phase 2C Plan: Real SSH Arm64 Baseline Validation

Status: completed. A user-provisioned AWS T4g instance was used only as an
existing Linux Arm64 target; ArmTune did not provision, resize, start, stop, or
otherwise control cloud resources.

Phase 2C validated the real-target path:

```text
Windows controller
  -> system OpenSSH
  -> Linux Arm64 target
  -> real llama-bench binary
  -> real GGUF model
  -> measured result
  -> downloaded artifact digest validation
```

Phase 2C validation evidence:

```text
Evidence file: docs/armtune/results/PHASE_2C_AWS_GRAVITON2.md
Run ID: 20260716_231915_147773
Target type: ssh
Target architecture: aarch64
Target OS: Ubuntu 24.04.4 LTS
Kernel: 6.17.0-1017-aws
CPU model: Neoverse-N1
Result origin: measured
Validation status: passed
Execution status: completed
Parse status: parsed
Benchmark status: passed
Artifact validation: passed
Checked artifacts: 13
Fake executable marker: false
Prefill throughput: 43.8 tokens/s
Decode throughput: 30.13 tokens/s
PerfRunbench commit: 41b57b70b318a94977a5dd822314782e855295f6
llama.cpp commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama-bench SHA-256: 990d4eee8877ed2b6758afd198b61546c87092a606e2ee696d08c34f896b01b8
Model SHA-256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
```

This is a preliminary measured baseline used to validate the real SSH Arm64
execution pipeline on a burstable AWS T4g instance. It is not evidence of
optimization, KleidiAI execution, build-profile improvement, thread-search
improvement, quality preservation, stable production performance, or formal
competition benchmark results.

## Phase 3A Plan: Repeated Benchmark Search Foundation

Status: implemented. Phase 3A adds repeated `llama-bench` measurement and a
conservative Arm-aware candidate search foundation. Implementation tests use
fake executables and synthetic fixtures only.

A later Phase 3A.1 validation run exercised the implemented search pipeline on
a user-provisioned AWS Graviton2 development instance. ArmTune did not
provision, resize, start, stop, or otherwise control cloud resources.

Allowed scope:

- Configurable warm-up runs, search repetitions, confirmation repetitions,
  cooldown seconds, random seed, and paired-baseline insertion cadence.
- Separate raw stdout/stderr artifacts for every repetition.
- Randomized candidate order with the recorded seed.
- Periodic paired-baseline insertion during search.
- Trial failure preservation without converting failures to zero metrics.
- Metric distributions with valid count, failed count, median, p95 when
  meaningful, min, max, and IQR or standard deviation.
- Warnings when sample counts are too small for meaningful percentiles.
- Thread candidates generated from visible and allowed CPU topology.
- `threads-batch`, batch, and ubatch candidates only when the inspected
  `llama-bench` help output supports the relevant flags.
- Process-level affinity candidates: `scheduler-default`, `compact`, `spread`,
  `reserve-one-core`, and `numa-local` only when meaningful.
- Deterministic staged search:
  1. thread count
  2. threads-batch and batch parameters
  3. affinity
  4. confirmation of baseline and selected candidate
- Noise-band and inconclusive classification.
- Performance-only recommendations with `quality_gate_status:
  not_evaluated`.

Out of scope:

- `llama-server`, HTTP, streaming, and TTFT.
- BootTrace and semantic quality validation.
- Build-profile comparison.
- KleidiAI claims.
- Arm Performix.
- Performance Passport generation.
- Cloud provisioning or automatic instance control.
- Full Cartesian product search by default.

Default protocol controls:

```text
warmup_runs: 2
search_runs: 5
confirmation_runs: 10
cooldown_seconds: 2
random_seed: 42
insert_paired_baseline_every: 4
noise_band_percent: 2.0
```

Every repetition records candidate ID, repetition index, start/end UTC, raw
stdout/stderr artifact paths, execution status, parse status, parsed metrics,
host fingerprint, allowed CPU mask, target load when available, and failure
reason.

Phase 3A recommendations must use bounded wording such as:

> Best observed performance candidate among N tested configurations.

They must not recommend candidates that failed required trials, missed required
metrics, are statistically indistinguishable within the configured noise band
unless marked inconclusive, or differ in model, binary, target,
prompt/generation counts, or protocol.

Phase 3A.1 validation evidence:

```text
Evidence file: docs/armtune/results/PHASE_3A_REAL_SEARCH_VALIDATION.md
Run ID: 20260718_001757_718789
Target type: ssh
Target architecture: aarch64
Target OS: Ubuntu 24.04.4 LTS
Kernel: 6.17.0-1017-aws
CPU model: Neoverse-N1
Logical CPUs: 2
PerfRunbench commit: ee4cbb305d788e8405e65ab839b1848cb743a2eb
llama.cpp commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama-bench SHA-256: 990d4eee8877ed2b6758afd198b61546c87092a606e2ee696d08c34f896b01b8
Model SHA-256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
Search status: completed
Result origin: measured
Parse statuses: parsed=34
Benchmark statuses: passed=34
Failed repetitions: 0
Artifact validation: passed
Validated repetition manifests: 34
Checked artifact records: 442
Quality gate status: not_evaluated
Selected candidate: affinity-spread-0_1
Recommendation status: inconclusive
Noise-band classification: inconclusive
Delta percent: 0.116029
Noise band percent: 5.0
```

This evidence supports a real-target validation claim for repeated benchmark,
staged search, statistics, conservative recommendation, SSH worker,
provenance, parser, and artifact-validation plumbing only. It does not show
final optimization, stable production performance, KleidiAI usage,
quality-gate validation, build-profile comparison, Performix, Performance
Passport generation, or formal competition benchmark results.

## Phase 3A.2 Plan: Search Candidate Controls and Cost Guardrails

Status: implemented. Phase 3A.2 adds explicit controls for small, bounded
real-target smoke profiles without changing the Phase 3A performance-only
claim boundary.

Phase labels are reserved as follows:

```text
Phase 3A.1
Real bounded search validation

Phase 3A.2
Candidate-count and affinity guardrails

Phase 3B
Server inference and deterministic quality validation
```

Allowed scope:

- Add `search.max_candidates` to cap unique tested candidate configurations.
- Keep the baseline candidate even when the cap is one.
- Apply the cap in deterministic stage order: baseline, thread candidates,
  supported batch candidate, affinity candidates.
- Do not count warm-up runs, paired-baseline insertions, confirmation runs, or
  process execution count toward `search.max_candidates`.
- Mark stages that still had raw candidates but could not be evaluated due to
  the cap as `skipped_due_to_candidate_limit`.
- Add `search.affinity_modes` to allowlist generated affinity candidates.
- Reject unknown affinity modes as configuration errors and deduplicate
  repeated modes.
- Avoid invalid affinity masks, including `reserve-one-core` on a single
  visible CPU and `numa-local` on single-node topology.
- Keep default behavior backward compatible when these fields are omitted.
- Expose matching CLI overrides for `autotuneai arm optimize`.
- Persist the final effective config after CLI overrides are applied.
- Add fake-executable tests for config parsing, affinity filtering, and capped
  candidate plans.

Out of scope:

- New real-target benchmark evidence.
- New optimization claims.
- Quality gate.
- Build-profile comparison.
- KleidiAI claims.
- Performix.
- Performance Passport.
- Cloud provisioning or instance control.

New optional search controls:

```text
max_candidates: optional positive integer
affinity_modes: optional list from scheduler-default, compact, spread,
  reserve-one-core, numa-local
```

`scheduler-default` is accepted in `affinity_modes` for configuration symmetry
with benchmark affinity settings, but generated affinity candidates are the
non-default modes. The baseline remains the scheduler-default comparison point
unless the benchmark config explicitly sets another baseline affinity.

This hardening addresses the Phase 3A.1 limitation where the real-target smoke
run requested a very small candidate set but the implementation still planned
five candidates on a 2-vCPU target. It does not retroactively change the
recorded Phase 3A.1 evidence.

## Phase 3B.1 Plan: llama-server Lifecycle and Generation Metrics

Status: implemented as a fake-tested foundation. Phase 3B.1 introduces a
`llama-server` benchmark foundation without changing the Phase 3A
`llama-bench` search engine and without adding quality scoring.

Allowed scope:

- Runtime adapter foundation that keeps existing `llama-bench` behavior
  backward compatible.
- `llama-server` binary inspection with raw `--version` and `--help`
  artifacts, supported flag detection, binary SHA-256, and unknown/unverified
  fields when evidence is absent.
- Versioned worker operation `benchmark.llama_server`.
- Versioned server job specification with executable, model, server,
  workload, and artifact controls.
- Loopback-only server binding; Phase 3B.1 rejects public bind addresses and
  does not open external ports.
- Server process lifecycle: free loopback port selection, argument-list
  subprocess launch, bounded readiness, bounded shutdown, process cleanup, and
  port-release verification.
- Target-local HTTP client that connects only to
  `http://127.0.0.1:<selected-port>`.
- OpenAI-compatible streaming request support using standard-library HTTP
  functionality unless a dependency blocker is documented.
- TTFT measured from request start to the first valid generated content delta.
- End-to-end latency measured from request start to response completion.
- Raw HTTP status, selected response headers, complete streaming event log,
  generated output preservation, and request-level result artifacts.
- A small public synthetic prompt suite covering short deterministic
  completion, structured JSON generation request, and longer summarization.
- Sequential protocol with configurable warm-up requests, measured repetitions,
  request timeout, cooldown, seed, temperature, and predicted tokens.
- Warm-up request artifacts preserved but excluded from measured
  distributions.
- Versioned schemas for inspection, configuration, lifecycle, streaming
  events, request results, metric distributions, and final benchmark results.
- Public CLI fake-server E2E and fake SSH worker protocol tests.
- `quality_gate_status: not_evaluated` for all Phase 3B.1 results.

Out of scope:

- BootTrace semantic or deterministic quality scoring.
- Candidate search through `llama-server`.
- Performance recommendation.
- Real AWS or external SSH execution.
- Cloud provisioning, instance start/stop, or security-group changes.
- Model download.
- Build-profile comparison.
- KleidiAI claims without verified execution evidence.
- Arm Performix.
- Performance Passport generation.
- Formal competition benchmark conclusions.

Implementation validation is local/fake only in this checkpoint. No real AWS,
external SSH, real model, or real `llama-server` performance benchmark was
executed for Phase 3B.1.

Phase 3B.1 artifact layout:

```text
.autotuneai/runs/<run_id>/armtune/server_benchmark/
  server_job.json
  executable_manifest.json
  model_manifest.json
  inspection.json
  server_configuration.json
  lifecycle.json
  benchmark_result.json
  artifact_manifest.json
  raw/
    inspection_version_stdout.txt
    inspection_version_stderr.txt
    inspection_help_stdout.txt
    inspection_help_stderr.txt
    server_stdout.txt
    server_stderr.txt
  requests/
    <prompt-id>/
      warmup/
        0001/
          request.json
          stream.jsonl
          output.txt
          result.json
      measured/
        0001/
        0002/
        0003/
```

Phase 3B.1 CLI summary should report run ID, target type, target
architecture, result origin, server startup/readiness status, request counts,
failed request counts, median TTFT when available, median end-to-end latency
when available, cleanup status, `quality_gate_status: not_evaluated`, and the
artifact directory. Missing metrics must print `Not available`, not zero.

Phase 3B.1 exit codes:

```text
0 = server and required measured requests completed and parsed
1 = server executed but benchmark or required requests failed
2 = invalid configuration, unsupported target, validation failure, or protocol mismatch
3 = internal/runtime error
4 = cleanup failure that may have left a process running
```

## Phase 3B.1-R Validation: Bounded Real Arm64 llama-server Run

Status: completed. A user-provisioned AWS T4g / Graviton2 instance was used
only as an existing Linux Arm64 target; ArmTune did not provision, resize,
start, stop, or otherwise control cloud resources.

Phase 3B.1-R validated the real-target server path:

```text
Windows controller
  -> system OpenSSH
  -> Linux Arm64 target
  -> real llama-server binary
  -> real GGUF model
  -> loopback-only target-local HTTP client
  -> OpenAI-compatible SSE parser
  -> generated-output preservation
  -> cleanup and port-release verification
  -> downloaded artifact digest validation
```

Validation evidence:

```text
Evidence file: docs/armtune/results/PHASE_3B1_REAL_SERVER_VALIDATION.md
Run ID: 20260719_004451_487119
Target type: ssh
Target architecture: aarch64
Target OS: Ubuntu 24.04.4 LTS
Kernel: 6.17.0-1019-aws
CPU model: Neoverse-N1
PerfRunbench commit: 1b70660812cee0da2ddf40eb1a0bc1af4a51d4c9
llama.cpp commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama-server SHA-256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
Model SHA-256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
Result origin: measured
Inspection status: passed
Startup status: ready
Readiness status: ready
Request status: completed
Stream parse status: parsed
Benchmark status: passed
Cleanup status: passed
Quality gate status: not_evaluated
Measured requests: 6
Failed requests: 0
Artifact validation: passed
Checked artifacts: 48
```

This evidence validates the server execution pipeline only. It is not evidence
of optimization, KleidiAI execution, quality preservation, stable production
performance, build-profile improvement, Performix support, Performance
Passport generation, or a formal competition benchmark result.

## Phase 3B.2 Plan: Deterministic BootTrace Quality Gate

Status: planned. Phase 3B.2 will add deterministic quality validation on top of
the Phase 3B.1 server benchmark foundation.

Planned scope:

- Structured BootTrace prompt suite and expected-output rules.
- Deterministic semantic validation that does not require external services.
- Quality eligibility status separate from performance metrics.
- Candidate rejection when performance improves but required quality checks
  fail.

Phase 3B.2 remains out of scope for the Phase 3B.1 implementation.

## Phase 4 Plan: Quality-Aware Server Autotuning MVP

Status: implemented and real-target rejection validation completed. Phase 4
combines deterministic BootTrace quality validation with bounded
`llama-server` candidate search. It remains an MVP focused on a small public
synthetic quality suite, not on final competition benchmarking or optimized Arm
execution claims.

Allowed scope:

- Deterministic local BootTrace quality suite using public synthetic traces.
- Versioned quality-suite, validator-result, aggregate-result, candidate
  quality, and quality-gate schemas.
- Deterministic validators for valid JSON, required fields, exact fields,
  allowed values, required substrings, forbidden substrings, regular
  expressions, numeric ranges, evidence line IDs, required evidence citations,
  invented line ID rejection, and nonempty output.
- `autotuneai arm optimize --config <path>` support for
  `runtime.adapter: llama_server`, `workload.mode: server`, and
  `quality_gate.enabled: true`.
- Reuse of the Phase 3B.1 `benchmark.llama_server` worker path for every
  server candidate, preserving lifecycle, request, raw SSE, generated output,
  and artifact-manifest behavior.
- Reuse of Phase 3A search controls where practical: deterministic candidate
  order, seed, warm-up exclusion, measured repetitions, confirmation runs,
  failure preservation, noise-band classification, `max_candidates`, and
  `affinity_modes`.
- Initial server candidates limited to safe dimensions proven by inspected
  help output: thread count, batch size, ubatch size, and process affinity.
  The Phase 4 MVP implementation tests staged candidates in this order:
  baseline, batch/ubatch parameters, thread count, affinity, then confirmation
  when an eligible selected candidate exists.
- Candidate quality eligibility and conservative recommendation states:
  `recommended`, `inconclusive`, `rejected_quality`,
  `rejected_execution`, `insufficient_evidence`, and
  `no_eligible_candidate`.
- Fake full E2E that demonstrates one faster quality-passing candidate and one
  faster quality-failing candidate that is rejected.
- Bounded real Arm64 validation only after the implementation commit, using an
  existing manually started target and sanitized evidence documentation.

Out of scope:

- Build-profile comparison.
- KleidiAI implementation, verification, or claims.
- llama.cpp source modification.
- Model quantization search.
- Concurrency or load testing.
- Distributed inference.
- Arm Performix.
- Complete Performance Passport generation.
- Final competition benchmark conclusions.

Phase 4 artifact layout:

```text
.autotuneai/runs/<run_id>/armtune/server_optimize/
  optimize_job.json
  effective_config.json
  quality_suite_manifest.json
  quality_suite_snapshot.json
  prompt_suite_snapshot.json
  server_optimize_result.json
  artifact_validation.json
  artifact_manifest.json
  candidates/
    <candidate-id>/
      search/
        0001/
          server_job.json
          benchmark_result.json
          artifact_manifest.json
          raw/...
          requests/...
      confirmation/
        0002/
          server_job.json
          benchmark_result.json
          artifact_manifest.json
          raw/...
          requests/...
```

The controller creates a derived `prompt_suite_snapshot.json` from the quality
suite for local fake/full E2E runs. For SSH targets, the workload
`prompt_suite_path` remains an absolute target path; the operator must upload
the derived prompt suite to that target path before running the bounded real
validation.

Phase 4 recommendation wording must be bounded, for example:

> Best observed eligible candidate under the tested server benchmark and
> deterministic quality protocol.

Phase 4 must not describe any result as the best Arm configuration, globally
optimal, production optimal, stable production performance, KleidiAI evidence,
or a formal competition benchmark result.

Phase 4 real-target validation evidence:

```text
Evidence file: docs/armtune/results/PHASE_4_REAL_QUALITY_AWARE_TUNING.md
Run ID: 20260719_035956_457526
Target type: ssh
Target architecture: aarch64
Target OS: Ubuntu 24.04.4 LTS
Kernel: 6.17.0-1019-aws
CPU model: Neoverse-N1
Logical CPUs: 2
PerfRunbench commit: 33504d731bdb11316d439609cb3bf3145401b9e8
llama.cpp commit: b15ca938ad00aa6b3ee6c2edda7363fd02826b18
llama-server SHA-256: 6faba7459f6d4ed13564bc5ec876e0bde0c0aa538fb371865dd89e03b11b254e
Model SHA-256: 041474553fcabfc2a2d67903f9d2c2e50bd92528e670da4f33b5d0ce6e59fd55
Result origin: measured
Runtime adapter: llama_server
Quality suite ID: synthetic_boottrace_quality_v1
Quality suite digest: 42a784ba1da5c624f14dfb9d2d89917768650c8e60ad0918f98c2e4da96d1169
Required quality cases: 5
Candidate count: 3
Eligible candidate count: 0
Quality gate status: failed
Recommendation state: no_eligible_candidate
Artifact validation: passed
Checked artifacts: 224
Cleanup status: passed
```

This evidence validates real Arm execution, server lifecycle, streaming
metrics, deterministic quality evaluation, quality-aware rejection, cleanup,
and artifact validation. It does not validate a quality-preserving
recommendation because every measured candidate failed the canonical BootTrace
quality gate.

## Phase 5 Plan: Competition Evidence MVP

Status: local MVP implemented; bounded real quality calibration recorded in
`docs/armtune/results/PHASE_5_COMPETITION_EVIDENCE.md`. Phase 5 turns the
Phase 4 quality-rejection result into a competition-evidence
pipeline. It recovers the model/profile quality contract through auditable
diagnostics and optional normalization without weakening canonical BootTrace
expected answers, adds reproducible Arm build-profile evidence, compares only
quality-preserving profiles, and generates a verifiable Performance Passport.

Allowed scope:

- Structured quality failure diagnostics for real and fixture generated
  outputs.
- Optional, versioned, auditable output normalization.
- Strict quality mode where no normalization is permitted.
- Structured-generation capability inspection based on the inspected
  `llama-server` binary and bounded probes, not assumptions.
- A versioned BootTrace prompt contract with rendered prompts and digests.
- `autotuneai arm quality-calibrate --config <path>` for fixed-profile model
  suitability checks before performance search.
- Declarative Arm build-profile plans for `portable-baseline`, `arm-native`,
  and `arm-optimized`.
- Build and runtime provenance sufficient to distinguish requested Arm
  optimization from evidenced Arm optimization.
- Quality-aware build/profile comparison that rejects quality-failing,
  execution-failing, or evidence-incomplete profiles.
- Performance Passport JSON and Markdown generation plus verification.
- Public fake E2Es for unsuitable model calibration, suitable normalized or
  structured-output calibration, build-profile evidence, profile comparison,
  and passport verification.
- Bounded real validation only on an existing manually started Arm target.

Out of scope:

- llama.cpp source-level kernel modification.
- Model fine-tuning.
- Quantization conversion.
- Distributed inference.
- Concurrency or load benchmarking.
- Automatic cloud provisioning or cloud instance control.
- Proprietary BootTrace input.
- Formal production certification.

Phase 5 quality recovery rules:

- The existing Phase 4 real result
  `20260719_035956_457526` is valid evidence that the quality gate rejected
  unsuitable generated output.
- Validators must not be weakened to force that model/profile to pass.
- Normalization may trim whitespace, remove UTF-8 BOMs, normalize line
  endings, and extract exactly one complete Markdown fenced JSON block when
  the selected policy permits it.
- Normalization must not repair malformed JSON, invent missing fields, change
  values, infer evidence IDs, or replace invalid answers with canonical
  answers.
- Every case result must preserve raw output, optional normalized output,
  normalization steps, parser status, validator results, final failure
  categories, raw SHA-256, and normalized SHA-256 when applicable.

Phase 5 build-profile rules:

- `build-plan` produces an auditable plan without executing a build.
- `build-profile` may execute only against an explicitly configured existing
  target and must not provision cloud resources.
- Build execution must use allowlisted source/build paths and argument lists,
  not arbitrary shell command fields.
- `arm-optimized` may report KleidiAI only with build and runtime evidence.
  A requested CMake flag alone is not sufficient.

Phase 5 comparison rules:

- A profile is comparison-eligible only when build evidence, binary/model
  digests, server lifecycle, quality calibration, cleanup, and artifact
  validation pass.
- The same model, prompt suite, normalization policy, generation settings,
  target, and CPU allocation must be used across compared profiles.
- The primary metric is median client-observed TTFT; median end-to-end latency,
  p95 TTFT, p95 end-to-end latency, runtime timing, failure rate, and
  variability are secondary metrics.
- Failed trials are not aggregated as zero.
- Recommendation wording must be bounded:

> Best observed eligible build profile under the tested Arm target, model,
> server workload, and deterministic quality protocol.

Phase 5 Performance Passport rules:

- Generate `performance_passport.json` and `performance_passport.md`.
- Include repository commit, target fingerprint, source commit, build evidence,
  compiler/CMake versions, binary/model digests, quality suite/prompt/
  normalization digests, benchmark protocol, provenance, profile results,
  quality results, rejected results, selected result, statistics,
  artifact-manifest digest, claim eligibility, limitations, and redacted
  reproducibility command shapes.
- Claim eligibility states are `eligible_preliminary`, `ineligible_quality`,
  `ineligible_provenance`, `ineligible_statistics`, `ineligible_cleanup`, and
  `fixture_only`.
- Passport verification must reject schema, digest, artifact-manifest, or claim
  consistency failures.

Phase 5 real-validation boundary:

- First inspect structured-output capability.
- Then run quality calibration for the existing model/profile.
- If the model/profile remains unsuitable, stop performance comparison and
  record ineligible evidence.
- If suitable, build or inspect profiles, calibrate every profile, compare only
  eligible profiles, generate and verify a passport, revalidate artifact
  manifests, and verify cleanup.
- Do not automatically download or substitute a model.

Phase 5 implemented artifact roots:

```text
.autotuneai/runs/<run_id>/armtune/quality_calibration/
  quality_calibration_job.json
  effective_config.json
  quality_suite_manifest.json
  quality_suite_snapshot.json
  normalization_policy.json
  prompt_contract.json
  prompt_suite_snapshot.json
  benchmark/
  quality_calibration_result.json
  artifact_manifest.json
  artifact_validation.json

.autotuneai/runs/<run_id>/armtune/build_profile/
  build_plan.json
  build_profile_result.json
  artifact_manifest.json
  artifact_validation.json

.autotuneai/runs/<run_id>/armtune/profile_comparison/
  profile_comparison_job.json
  profile_comparison_result.json
  artifact_manifest.json
  artifact_validation.json

.autotuneai/runs/<run_id>/armtune/passport/
  performance_passport.json
  performance_passport.md
  performance_passport_verification.json
```

Phase 5 fake E2E coverage validates command wiring, fixture build evidence,
quality-preserving profile comparison, quality rejection, Performance Passport
generation, and digest tamper detection. Fixture measurements remain
`result_origin: test_fixture` and must not be used as Arm performance results.

Phase 5-R run `20260719_132056_332743` followed this boundary on an existing
Linux Arm64 target. The fixed profile completed measured server requests and
artifact validation passed, but strict BootTrace quality calibration failed
with `required_quality_cases_passed: 0/5`, so measured profile comparison and
measured Performance Passport generation were intentionally not executed.

## Phase 6 Plan: Qualified Model And Arm Profile Comparison

Status: partially executed. Phase 6 uses the Phase 5 evidence pipeline without
adding framework functionality. Qwen2.5-7B qualification was executed and did
not pass the public synthetic BootTrace quality gate, so Arm build-profile
comparison, paired measurement, and Performance Passport generation remain
blocked pending a suitable model/profile.

Phase 6A result boundary:

- Qwen3-4B Q4_K_M was evaluated on a manually provided C7g target.
- Default Qwen3 behavior produced `reasoning_content` without final
  `content`, so request parsing failed before semantic validation.
- A private Qwen3 `/no_think` prompt-contract variant produced raw JSON, but
  the semantic BootTrace validators still failed.
- Qwen3-4B is therefore unsuitable for measured profile comparison under the
  tested contracts. Validators and canonical expected answers must not be
  weakened to force it through the gate.

Phase 6B qualification target:

- Model repository: `Qwen/Qwen2.5-7B-Instruct-GGUF`.
- Quantization target: `Q4_K_M`.
- Expected split files:
  `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` and
  `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`.
- Merge target:
  `qwen2.5-7b-instruct-q4_k_m.gguf`.
- Use the unchanged public BootTrace suite, canonical expected answers, and
  validator thresholds.
- Do not carry Qwen3-specific `/think`, `/no_think`, `enable_thinking`, or
  `reasoning_content` assumptions into Qwen2.5 qualification.

Phase 6B target and storage checks:

- Use an existing manually started Linux Arm64 C7g target only.
- Verify Linux `aarch64`, Neoverse-V1, 8 logical CPUs, approximately
  15 GiB usable memory, and `production_benchmark_ready: true`.
- Verify remote PerfRunbench commit
  `ff6a31531db48f423915ae37875afa90c949a0c4`.
- Keep the existing llama.cpp source commit unchanged.
- Require enough free disk for split files, merged GGUF, artifacts, and logs
  before downloading. If disk is insufficient, stop before running
  qualification rather than resizing storage automatically.

Phase 6B model acquisition evidence:

- Record model repository, revision when available, split file names, split
  sizes, split SHA-256 digests, merged file size, merged SHA-256 digest,
  download command shape, merge command shape, and disk space before and after.
- Use `llama-gguf-split --merge` for split GGUF merging. If the utility is
  absent, build only that target from the unchanged llama.cpp source tree.
- Do not commit model files or private paths.

Phase 6B quality gate:

- First run strict output calibration.
- Run the existing versioned single-fence normalization only when the sole
  remaining failure is exactly one auditable Markdown JSON fence.
- Normalization must not repair `wrong_status`, `wrong_root_cause`,
  `wrong_failure_stage`, `missing_required_evidence`, `invented_evidence`, or
  malformed semantic values.
- Qualification success requires `result_origin: measured`, 5/5 request
  success, 5/5 required cases passed, zero critical validator failures, zero
  invented evidence IDs, passed cleanup, passed artifact validation, and
  `model_profile_suitability: suitable`.

Phase 6B Qwen2.5-7B result boundary:

- Qwen2.5-7B Q4_K_M was downloaded as split GGUF files from
  `Qwen/Qwen2.5-7B-Instruct-GGUF`, merged with `llama-gguf-split --merge`,
  and validated by SHA-256.
- The run used an existing C7g / Graviton3 Linux `aarch64` target with
  `production_benchmark_ready: true`.
- Strict calibration produced `result_origin: measured`, 5/5 completed
  requests, and 5/5 raw-valid JSON outputs.
- The required BootTrace quality gate still failed with
  `required_quality_cases_passed: 0/5` and
  `model_profile_suitability: unsuitable`.
- Failure categories included `wrong_root_cause`, `wrong_status`,
  `missing_required_evidence`, `response_format_violation`, and
  `model_capability_failure`; this was not a Markdown-fence-only failure.
- Artifact digest validation passed for both nested benchmark artifacts and
  outer quality-calibration artifacts.
- Phase 6C profile comparison and measured Performance Passport generation
  were intentionally not executed.

Conditional Phase 6C profile comparison:

- Proceed only when Qwen2.5-7B qualification is suitable.
- Generate and inspect build plans for `portable-baseline`, `arm-native`, and
  `arm-optimized`.
- Record source commit, source dirty state, compiler and CMake versions,
  requested definitions, effective CMake cache, compile evidence, binary
  SHA-256, binary size, linked-library evidence, build output, and optimization
  evidence status.
- Do not claim KleidiAI or another optimized path unless build and runtime
  evidence support it.
- Re-run identical quality calibration for each successfully built profile.
- Compare only quality-passing profiles that share the same model digest,
  quality-suite digest, prompt-contract digest, normalization policy,
  generation seed, temperature, context size, output limit, target allocation,
  and CPU affinity policy.
- Use TTFT as the primary metric and end-to-end latency as a secondary metric.
  Retain failed trials separately and do not aggregate failures as zero.
- Recommendation wording remains bounded:

> Best observed eligible build profile under the tested Arm target, model,
> server workload, and deterministic quality protocol.

Phase 6 Passport boundary:

- Generate and verify `performance_passport.json` and
  `performance_passport.md` only when at least one profile is eligible.
- Valid preliminary claim states include `eligible_preliminary`,
  `ineligible_statistics`, `ineligible_provenance`, `ineligible_cleanup`, and
  `ineligible_quality`.
- Do not force `eligible_preliminary`.

Phase 6 result document:

- Record final evidence in
  `docs/armtune/results/PHASE_6_QWEN25_PROFILE_COMPARISON.md`.
- The document must truthfully report either an unsuitable model with
  comparison blocked, or a suitable model with completed profile comparison.
- Do not include private paths, public IP addresses, SSH keys, credentials, or
  model contents.

## Phase 7 Plan: BootTrace Contract Calibration And Conditional Model Ladder

Status: foundation implemented; real model reruns are pending explicit target
availability and cost approval. Phase 7 does not weaken held-out validators or
change canonical answers merely because Qwen3-4B and Qwen2.5-7B failed.

Phase 7 separates four concerns:

- held-out evaluation suite:
  `configs/armtune/quality/boottrace_suite.json`
- public development calibration suite:
  `configs/armtune/quality/boottrace_calibration_suite.json`
- held-out case audit:
  `configs/armtune/quality/boottrace_case_audit.json`
- prompt-contract version selected by
  `quality_gate.prompt_contract_version`

Held-out case audit:

- Each of the five evaluation cases records trace sufficiency, unique expected
  classification, expected status, root-cause taxonomy, failure stage,
  required evidence lines, distractor status, unknown-policy applicability,
  ambiguity, and whether a canonical answer change is required.
- The Phase 7 audit records no canonical answer changes. The
  `boot_config_missing` case notes a low ambiguity between the absent storage
  UUID trace and the legacy canonical summary phrase; the answer remains
  unchanged and the contract clarifies taxonomy instead.
- Canonical answers may change only when the fixture or answer is
  demonstrably incorrect or ambiguous, with written rationale. They must not
  change merely because a model failed.

Development calibration suite:

- The development suite is public and synthetic.
- It is disjoint from the held-out evaluation suite by case ID and evidence
  line IDs.
- It may provide few-shot examples for prompt-contract calibration.
- It must not be reported as final qualification evidence or included in the
  5/5 held-out pass count.

Versioned prompt contracts:

- `current-zero-shot`: preserves the existing evaluation prompts.
- `taxonomy-and-decision-rules`: prepends explicit status decision order,
  unknown policy, and root-cause taxonomy while keeping the same output schema.
- `taxonomy-rules-and-disjoint-few-shot`: adds disjoint development examples
  from the public calibration suite. The five held-out evaluation cases are not
  reused as examples.

Prompt contract artifacts record:

- contract version
- taxonomy version
- rendered prompt text and prompt digest for every held-out case
- expected response schema and digest
- generation settings
- normalization policy
- development example digests when the few-shot contract is used

SSH quality calibration now uploads the controller-rendered
`prompt_suite_snapshot.json` into the versioned remote run directory before
invoking the fixed worker command. This ensures the remote `llama-server`
requests actually use the selected prompt contract. Request JSON artifacts
still preserve the final message content sent to the server.

Phase 7 real-target order:

1. Verify free memory and disk on the existing target.
2. Re-run Qwen2.5-7B Q4_K_M with:
   `current-zero-shot`, `taxonomy-and-decision-rules`, and
   `taxonomy-rules-and-disjoint-few-shot`.
3. Keep model digest, evaluation suite digest, validators, generation seed,
   temperature, context size, output token bound, and normalization policy
   fixed across the three contract variants.
4. Qualify only if the held-out evaluation result has 5/5 required cases, zero
   critical failures, zero invented evidence IDs, zero request failures,
   cleanup passed, artifact validation passed, and
   `model_profile_suitability: suitable`.
5. If Qwen2.5-7B remains unsuitable, conditionally evaluate a stronger model
   ladder after checking memory, disk, license access, repository revision,
   file digests, chat-template behavior, and structured-output support.

Conditional model ladder:

- `Qwen/Qwen2.5-14B-Instruct-GGUF` with a suitable 4-bit quantization.
- `google/gemma-3-12b-it-qat-q4_0-gguf` only after confirming license access.
- `Qwen/Qwen3-8B-GGUF` remains lower priority because prior Qwen3 runs showed
  thinking/final-content integration risk.

Stop conditions before model comparison:

- OOM or memory pressure that would require swap to manufacture success
- unsupported or ambiguous chat template
- parser ambiguity
- cleanup failure
- digest mismatch
- artifact-validation failure
- model/profile suitability other than `suitable`

Profile comparison remains blocked for unsuitable models.
