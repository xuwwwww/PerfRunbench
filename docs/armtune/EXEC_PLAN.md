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
