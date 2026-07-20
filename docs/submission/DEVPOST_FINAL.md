# Devpost Final Copy

Copy these sections into the competition submission form. Keep the wording
conservative: measured, synthetic, fixture, and preliminary evidence are
different categories.

## 1. Project Title

PerfRunbench

## 2. One-Line Tagline

One-command, guarded performance autotuning for Arm inference and PyTorch
training workloads.

## 3. Short Summary

PerfRunbench wraps an existing workload with one command, runs bounded tuning
slices, confirms the best observed eligible configuration statistically, caches
the recommendation, and safely reuses it with environment validation, canary
execution, process-scoped apply, and rollback. ArmTune adds real Linux Arm64
inspection, llama.cpp benchmarking, and Performance Passport evidence.

## 4. Full Project Description

Performance tuning is usually a slow manual loop: change a runtime flag, build
with a different compiler profile, run the workload, copy numbers into a
spreadsheet, and hope the machine state did not drift. The best configuration
depends on the machine, runtime, workload, and measurement protocol. A setting
that is fast on one target can be invalid on another. A naive autotuner can
recommend noise, change training semantics, or leave unsafe system state.

PerfRunbench is a guarded performance autotuning system. A user wraps an
existing workload with one command. PerfRunbench validates workload identity,
runs bounded tuning slices, evaluates declared configuration candidates,
statistically confirms the best observed eligible configuration, caches the
recommendation, and safely reuses it with environment validation, canary
execution, process-scoped apply, and rollback.

The default workflow is:

```bash
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai doctor --config autotune.yaml -- python train.py
autotuneai run --auto --config autotune.yaml -- python train.py
```

On the first run, a cache miss triggers bounded tuning, fresh confirmation,
recommendation generation, canary execution, and then the full workload. On
later runs, PerfRunbench validates the environment and reuses an eligible
cached recommendation without retuning.

ArmTune is the Arm inference subsystem inside PerfRunbench. It inspects Linux
Arm64 targets, validates production-ready Arm provenance, builds and compares
llama.cpp profiles, runs bounded runtime search, and generates a Performance
Passport that records the target, model, build evidence, runtime protocol,
statistics, and recommendation boundary.

For PyTorch training workloads, PerfRunbench uses declared knobs and bounded
training slices. The adapter preserves effective batch semantics, enforces
resource guards, rejects unsafe candidates, and prevents failed trials from
being silently converted into zero-valued metrics.

PerfRunbench is built around one principle: an optimization recommendation is
only useful if it is measured, bounded, reproducible, and safe to apply on the
same kind of environment where it was produced.

## 5. Inspiration

Manual performance tuning is expensive because every result needs context. A
single throughput number is not enough: users need to know what machine was
used, what build produced the binary, what runtime knobs were active, whether
the result cleared a noise band, and whether the recommendation can safely be
reused later.

The project was inspired by Arm performance work where build flags, thread
placement, model format, and target identity all matter. The goal was to turn
that workflow into a repeatable tool instead of a one-off benchmark notebook.

## 6. What It Does

- Initializes a public configuration for supported workload adapters.
- Checks readiness with `autotuneai doctor` without launching the workload.
- Wraps existing commands with `autotuneai run --auto`.
- Runs bounded candidate searches instead of full Cartesian products.
- Preserves declared workload semantics such as effective batch size.
- Confirms selected candidates with repeated measurements.
- Rejects results inside the configured noise band.
- Caches RecommendationBundles bound to workload and environment identity.
- Refuses hard-drifted recommendations on different targets.
- Applies process-scoped recommendations with validation, canary, lease, and
  rollback.
- Generates auditable Performance Passport evidence for Arm inference results.

## 7. How It Works

PerfRunbench separates tuning into a few explicit stages:

1. Workload identity: command, config, identity files, adapter, and metric
   source are fingerprinted.
2. Environment identity: platform, CPU, topology, memory, and runtime evidence
   are captured.
3. Candidate generation: users declare knobs and bounds; PerfRunbench creates a
   small deterministic plan.
4. Bounded slices: short representative slices are run with resource guards and
   timeouts.
5. Metrics parsing: structured metrics are parsed without converting failures
   into zeros.
6. Statistical confirmation: repeated baseline and candidate measurements are
   compared against a noise band.
7. RecommendationBundle: the selected configuration is saved with evidence,
   digests, eligibility state, and apply policy.
8. Reuse: `run --auto` and `run --best` validate the environment, run canaries
   when required, apply only process-scoped settings, and clean up leases.

## 8. How It Was Built

PerfRunbench is implemented in Python with standard library components for the
core workflow. The repository includes:

- CLI commands under `autotuneai`.
- Run directories and artifact manifests.
- Resource and lifecycle management.
- Recommendation cache state handling.
- Process-scoped apply, canary, lease, status, and rollback commands.
- A PyTorch training adapter that keeps PyTorch optional.
- ArmTune modules for Linux Arm64 target inspection, SSH worker protocol,
  llama.cpp profile comparison, runtime search, and Performance Passport
  generation.
- Unit, fixture, installed-wheel, Windows, Linux, and Arm64 CI validation.

## 9. Challenges

- Preventing fixture and CI smoke evidence from being mistaken for real Arm
  performance.
- Keeping host capability separate from verified workload execution evidence.
- Designing cache reuse so that soft memory drift is not confused with hard CPU
  drift.
- Preserving PyTorch effective batch semantics while still allowing runtime
  tuning.
- Running useful low-resource validation on a small Arm target without
  triggering unsafe memory pressure.
- Keeping experimental model-capability work separate from the main guarded
  tuning workflow.

## 10. Accomplishments

- Released `v0.2.0` as a GitHub Release with verified wheel and source
  distribution digests.
- Validated installed-wheel release smoke on Windows, Linux x86_64, and Linux
  Arm64 GitHub Actions runners.
- Validated a real C7g Arm64 llama.cpp build and runtime tuning workflow.
- Validated guarded recommendation apply with canary, lease, and rollback.
- Validated synthetic PyTorch tuning on C7g Arm64, T4g Arm64, and Windows
  x86_64 without mixing their performance claims.
- Added environment-bound cache reuse and hard-drift refusal.
- Kept the public product path focused on benchmark evidence, bounded tuning,
  cache reuse, and guarded apply.

## 11. What Was Learned

The most important lesson was that performance evidence is only useful when it
is tied to provenance. PerfRunbench had to track target identity, source
commit, build evidence, workload identity, runtime protocol, raw outputs,
artifact digests, cache state, and cleanup status before a recommendation could
be treated as reusable.

Another lesson was that a small bounded search can be more useful than a large
uncontrolled search. The tool should produce a defensible recommendation or an
honest inconclusive result, not an impressive number with weak evidence.

## 12. What Is Next

- More workload adapters with the same RecommendationBundle and guard model.
- More Arm target families and repeated validation across additional hardware.
- Richer Performance Passport comparison tools.
- Better video-friendly reporting for tuning runs.
- Optional external workload integration through the generic command adapter.

## 13. Built With

- Python
- GitHub Actions
- llama.cpp
- PyTorch CPU wheels for optional synthetic training validation
- Linux Arm64 targets
- Windows x86_64 validation
- SVG and deterministic local chart generation

## 14. Arm Relevance

PerfRunbench includes ArmTune, an Arm-first inference subsystem validated on
Linux Arm64 targets. The submission includes:

- Arm platform inspection.
- Linux Arm64 production provenance checks.
- llama.cpp build-profile comparison on C7g / Neoverse-V1.
- Bounded runtime search for Arm inference.
- T4g low-resource Arm portability validation.
- Environment-bound recommendation refusal across incompatible targets.
- Performance Passport evidence for preliminary Arm inference results.

The tool does not claim global Arm optimality. It reports the best observed
eligible configuration under the tested Arm target, model, inference workload,
and measurement protocol.

## 15. Links

- Repository: `https://github.com/xuwwwww/PerfRunbench`
- GitHub Release: `https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.2.0`
- Related project: `https://github.com/xuwwwww/BootTraceBench`
- Demo video: `TBD`

## 16. Testing And Evidence

Public evidence is recorded in the repository:

- `docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md`
- `docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md`
- `docs/results/PHASE_14B_REAL_ARM_PYTORCH_VALIDATION.md`
- `docs/results/PHASE_15A_T4G_LOW_RESOURCE_VALIDATION.md`
- `docs/results/PHASE_15B_T4G_REAL_PYTORCH_VALIDATION.md`
- `docs/results/PHASE_15C_WINDOWS_X86_PYTORCH_VALIDATION.md`
- `docs/results/VALIDATION_MATRIX.md`
- `docs/releases/v0.2.0.md`

The GitHub Release assets were downloaded and their SHA-256 digests matched
the final local build artifacts:

```text
perfrunbench-0.2.0-py3-none-any.whl
d1f4b01bc4251c13700f36deeae5deef982c778c33b2c8408a2883dc22e18e21

perfrunbench-0.2.0.tar.gz
da4e14021ef8daa8f214c575827fe9a07050685a89bb54f23746892ad7d0ee37
```

## 17. Limitations

- The measured Arm inference result is preliminary and limited to one
  target/model/source/protocol set.
- Synthetic PyTorch training results validate tuning mechanics and guard
  behavior; they are not production training claims.
- C7g, T4g, and Windows results are not directly comparable.
- PyTorch is optional and is not required for the dependency-free release smoke.
- PerfRunbench does not automatically understand every arbitrary training
  script. Users declare knobs, metrics, identity files, and bounded slice
  controls.
- No PyPI package is published for this submission.
- Optional experimental workloads are outside the default PerfRunbench product
  path unless they are wrapped as generic command workloads.

## 18. Credits

PerfRunbench was built as an individual systems-performance project focused on
reproducible benchmarking, guarded autotuning, and Arm inference evidence.
It uses public tools and public or synthetic workloads only. The submission
does not include private keys, cloud credentials, model weights, or proprietary
logs.

## Shortened Version For Strict Character Limits

PerfRunbench is a guarded performance autotuning system. A user wraps an
existing workload with one command, then PerfRunbench runs bounded tuning
slices, confirms the best observed eligible configuration statistically,
caches the recommendation, and safely reuses it with environment validation,
canary execution, process-scoped apply, and rollback.

ArmTune is the Arm inference subsystem inside PerfRunbench. It inspects Linux
Arm64 targets, compares llama.cpp build profiles, runs bounded runtime search,
and generates Performance Passport evidence. The v0.2.0 submission also
validates the golden-path CLI and synthetic PyTorch training adapter across
Linux Arm64 and Windows x86_64.

The project focuses on defensible recommendations rather than single
unqualified speedup claims. Measurements remain protocol-specific; fixture,
synthetic, and real target evidence are clearly labeled.
