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

Phase 3A recommendations must use wording such as:

> Best observed performance candidate under the tested llama-bench
> configuration.

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
