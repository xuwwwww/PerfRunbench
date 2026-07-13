# AGENTS.md

## Project identity

This repository is PerfRunbench.

The active major development effort is:

**ArmTune: Architecture-Aware AI Inference Autotuning for Arm Neoverse**

ArmTune is an Arm-first AI inference optimization subsystem built on top of
PerfRunbench's reusable execution, monitoring, comparison, and reporting
infrastructure.

ArmTune is not generic Arm compatibility and must not be presented as a generic
benchmark wrapper that happens to execute on Arm.

## Communication language

- Communicate with the user exclusively in Traditional Chinese used in Taiwan.
- Never respond in Simplified Chinese.
- Technical terms may remain in English when that is clearer.
- Explain plans, risks, design decisions, test failures, and implementation
  progress in Traditional Chinese.
- Ask questions in Traditional Chinese only when a decision materially affects
  architecture, security, cost, data, or backward compatibility.
- Do not ask unnecessary confirmation questions for routine implementation
  details.
- Do not use emoji.

## Repository language

Use English for:

- source code
- identifiers
- comments
- docstrings
- CLI output
- configuration field names
- tests
- commit messages
- public README content
- challenge documentation
- Devpost submission materials
- diagrams and report labels

The user-facing conversation remains in Traditional Chinese.

## Truthfulness and benchmark integrity

- Never invent benchmark results.
- Never invent test execution results.
- Never claim a command was executed unless it was actually executed.
- Never treat fixture, mocked, synthetic, or CI-runner results as production
  Arm performance results.
- Never silently convert parser failures into zero-valued metrics.
- Never claim that Neon, DotProd, I8MM, BF16, SVE, SVE2, SME, SME2, KleidiAI,
  or another optimized path was used solely because the host supports it.
- Host capability and verified workload execution evidence must remain separate.
- Mark examples, placeholders, fixtures, and synthetic outputs explicitly.
- Preserve raw benchmark output and provenance needed to verify conclusions.
- When evidence is incomplete, report the result as unknown or unverified.

## Safety and privacy

- Never commit credentials, API keys, SSH private keys, cloud tokens, model
  weights, private logs, proprietary firmware logs, or internal company data.
- Never use Lenovo, Microsoft, AMD, customer, or NDA-protected material.
- Use only public, licensed, or synthetic data for the reference workload.
- Do not provision cloud resources, start paid instances, resize instances, or
  perform actions that may incur charges without explicit user approval.
- Do not execute destructive git operations.
- Do not rewrite git history.
- Do not force-push.
- Do not delete user-authored files merely to simplify implementation.
- Do not modify machine-wide settings without explicit opt-in.
- Temporary tuning must be restored after success, failure, timeout, SIGINT,
  and SIGTERM.

## Development workflow

For any multi-file or multi-hour task:

1. Read this file.
2. Inspect the relevant repository code and tests.
3. Read `docs/armtune/EXEC_PLAN.md` when it exists.
4. Update the plan before making major architectural changes.
5. Reuse existing PerfRunbench abstractions when they are sound.
6. Do not duplicate execution, monitoring, run storage, or report functionality.
7. Implement in reviewable phases.
8. Run focused tests after each meaningful change.
9. Run the complete relevant suite before declaring a phase complete.
10. Stop at the requested phase boundary and report results in Traditional
   Chinese.

At the end of each phase, report:

- changed files
- completed functionality
- design decisions
- commands actually executed
- test results
- known limitations
- unresolved risks
- the proposed next phase

## Architecture boundaries

PerfRunbench core owns reusable infrastructure such as:

- process execution
- process lifecycle management
- resource monitoring
- CPU affinity
- cgroup/systemd/Docker integration
- paired baseline comparison
- noise-band analysis
- run artifact storage
- generic JSON and HTML report utilities

ArmTune owns:

- controller/Arm-target orchestration
- real Arm64 target validation
- Arm CPU identity and capability detection
- Arm topology-aware search
- Arm-aware llama.cpp build inspection
- verified Arm optimization evidence
- Arm Performix integration
- AI inference quality validation
- Arm Performance Passport generation
- Arm-specific regression checks
- ArmTune challenge documentation and demo

Do not place Arm-specific logic into generic PerfRunbench modules unless a
small generic abstraction is clearly beneficial to existing and future users.

## Controller and target model

The developer controller may be Windows, WSL, Linux, or macOS on x86_64 or
Arm64.

Production benchmark execution must occur on a real Linux Arm64 target.

Supported initial targets:

- local Arm64 Linux
- remote Arm64 Linux over SSH

The controller may perform:

- configuration
- search orchestration
- result comparison
- report generation
- passport comparison
- fixture-based tests

The Arm target performs:

- host inspection
- build inspection or compilation
- inference execution
- CPU affinity
- process monitoring
- profiling
- raw artifact generation

Never present controller-side x86 measurements as Arm results.

## Code quality

- Target Python 3.10 or newer.
- Add type hints to new public APIs.
- Prefer dataclasses and standard-library components unless an external
  dependency provides clear value.
- Avoid shell=True.
- Validate external paths and arguments.
- Use bounded timeouts.
- Ensure child processes are terminated and reaped.
- Use atomic file writes for important manifests.
- Version serialized schemas.
- Produce actionable errors.
- Preserve backward compatibility with existing `autotuneai` commands.
- Keep new modules focused and testable.
- Do not create a heavy web framework for reporting.

## Testing

Tests must not require:

- a real model
- internet access
- root privileges
- an Arm machine
- Arm Performix
- paid cloud resources

Use fixtures and fake executables for unit and integration tests.

Real Arm integration tests must be explicitly marked and skipped by default.

Existing tests must remain green.

Do not weaken or remove existing tests merely to make new code pass.

## Documentation

Maintain these documents when relevant:

- `docs/armtune/EXEC_PLAN.md`
- `docs/armtune/ARCHITECTURE.md`
- `docs/armtune/ARM_SPECIFICITY.md`
- `docs/armtune/BENCHMARK_METHODOLOGY.md`
- `docs/armtune/PERFORMANCE_PASSPORT_SPEC.md`
- `docs/armtune/BEFORE_AFTER.md`
- `docs/armtune/RISK_REGISTER.md`
- `docs/armtune/LIMITATIONS.md`
- `docs/armtune/RESEARCH_NOTES.md`
- `docs/armtune/DEVPOST_DRAFT.md`
- `docs/armtune/DEMO_SCRIPT.md`

Public project documents must clearly distinguish:

- measured results
- examples
- fixtures
- synthetic results
- planned features
- unsupported features
- future work