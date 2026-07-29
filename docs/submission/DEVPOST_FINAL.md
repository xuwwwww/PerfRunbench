# Devpost Final Copy

This is the final English copy for the Arm AI Optimization Challenge. Claims
must stay within the evidence boundaries in `PROJECT_FACTS.md`.

## Project Title

PerfRunbench

## Tagline

Guarded autotuning for repeatable AI workloads on Arm.

## Short Description

PerfRunbench turns performance tuning into a bounded, reproducible workflow:
inspect the target, establish a baseline, search build and runtime candidates,
confirm the best observed eligible result statistically, and reuse it through
environment validation, canary execution, process-scoped apply, lease, and
rollback. ArmTune provides the Linux Arm64 inspection, llama.cpp tuning, and
Performance Passport evidence path.

## Full Description

### The Problem

The fastest benchmark run is not automatically the safest recommendation.
Performance depends on the target, workload, runtime, concurrency, topology,
and measurement protocol. A single result can be noise. A configuration may
stop being valid when the CPU or environment changes. An unsafe tool can leave
processes, leases, ports, or machine state behind.

PerfRunbench treats tuning as an evidence problem rather than a one-number
leaderboard. It wraps an existing workload, generates a deterministic bounded
candidate plan, excludes warm-up runs, measures repeated trials, compares
paired baselines, applies a configured noise band, confirms the selected
candidate, and records the evidence required to reuse that recommendation.

### What PerfRunbench Does

The primary workflow is:

```text
inspect target
-> validate workload and environment
-> establish baseline
-> search a bounded candidate plan
-> confirm the best observed eligible candidate
-> generate a Performance Passport
-> validate, canary, apply, lease, and roll back safely
```

The simplest command-workload entry point is config-free:

```bash
autotuneai run --auto --preset quick -- python app.py
```

`quick`, `standard`, and `deep` presets impose hard 5, 20, and 60 minute
budgets with bounded candidate counts. If a deadline expires, PerfRunbench
writes a valid partial report, performs cleanup, and does not cache ineligible
evidence.

ArmTune is the Arm inference subsystem. It validates Linux Arm64 production
targets, inspects workload binaries, builds architecture-aware profiles, runs
staged runtime search, performs paired confirmation, and generates a
Performance Passport with provenance, digests, statistics, cleanup state, and
recommendation boundaries.

The serving workflow adds built-in traffic profiles, deterministic load
generation, remote-client validation, topology-aware recommendation classes,
and `run --best` reuse without search.

### Guarded Recommendation Reuse

A cached recommendation is bound to workload identity and environment
fingerprints. Hard drift, such as a different CPU class or logical CPU count,
causes refusal rather than silent reuse. Eligible recommendations can be
validated, canary-tested, applied only to the target process, leased, and
rolled back. `run --best` never retunes.

Failed trials are never converted into zero-valued metrics. Results inside the
configured noise band remain inconclusive. Unsupported optimizations remain
unsupported or unverified instead of being promoted from a requested flag.
Kernel and system tuning are advanced opt-in experiments, not default behavior.

### Real Arm Result

The headline result comes from a real C7g Arm64 server and T4g Arm64 remote
client using a direct private network path, a pinned llama-server build,
Qwen2.5-7B Instruct Q4_K_M, closed-loop concurrency 4, and three declared
workloads.

The validated `slots4-cont-on` candidate was compared against the full-capacity
J1 eight-CPU baseline with deterministic randomized block order and fresh
server restarts:

| Workload | Baseline output req/s | Candidate output req/s | Improvement | Baseline p95 TTFT | Candidate p95 TTFT | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Interactive short | 17.032 | 31.355 | 84.092% | 3055.682 ms | 271.088 ms | 91.128% |
| Decode heavy | 17.046 | 31.790 | 86.490% | 10126.795 ms | 202.424 ms | 98.001% |
| Prefill heavy | 16.563 | 28.054 | 69.375% | 3967.424 ms | 341.693 ms | 91.388% |

Across the complete remote validation there were 608 measured requests, zero
retries, zero request failures, and zero quality-guard failures. Both hosts
were restored and artifact manifests were independently revalidated.

These numbers are protocol-specific. They are not a universal speedup claim
and do not apply to other hardware, models, runtimes, concurrency levels, or
network paths.

A later, separate C7g.4xlarge loopback experiment tested realistic
larger-target capacity. Under a separate pp512/tg128 protocol, prefill scaled
from 6.761 to 73.669 tokens/s and decode scaled from 2.759 to 30.089 tokens/s
when moving from 1 to 15 threads, approximately 10.9x in both cases.

Under the full-capacity loopback concurrency-8 paired protocol,
`slots4-cont-on` improved eligible `decode-heavy` output throughput from
25.779 to 46.488 tokens/s (+80.331%) and `interactive-short` from 25.600 to
43.712 tokens/s (+70.749%). The same candidate regressed `prefill-heavy` by
21.025%, so that workload was excluded. Eight-slot long-prefill requests that
exceeded per-slot context were preserved as HTTP 400 failures rather than
converted into metrics.

This larger-target result is loopback evidence, not remote-network evidence,
and is not a direct C7g.2xlarge versus C7g.4xlarge comparison. Hardware
scaling and runtime tuning remain separate protocols.

A newer same-protocol experiment repeated one pinned Qwen2.5-7B artifact,
llama.cpp runtime bundle, CPU-allocation rule, and concurrency-4 loopback
protocol on C7g.xlarge, C7g.2xlarge, and C7g.4xlarge. Four-slot continuous
batching improved formally confirmed decode throughput from 6.237 to 12.136,
13.626 to 25.876, and 25.475 to 46.186 tokens/s respectively. The within-target
deltas were 94.585%, 89.901%, and 81.296%.

The result remained workload-aware: four-slot prefill-heavy screening regressed
by 38.977%, 38.039%, and 35.310% and was rejected on all three sizes. Across
the three targets, the protocol recorded 1,047 measured HTTP requests, 30 fresh
server starts, zero request or quality failures, complete cleanup, and 514
successful controller-side artifact digest checks after download.

These within-target tuning percentages are not added to hardware scaling and
are not remote-network evidence.

Separate earlier protocols established a C7g llama.cpp build-profile decode
median of 10.825 tokens/s for `portable-baseline` and 17.530 tokens/s for
`arm-native`; a bounded runtime search measured 13.630 to 17.480 tokens/s
(28.246515% observed improvement); and a secondary llama-server comparison
measured 97.260 to 62.826 ms median TTFT. These improvements are intentionally
not combined.

### Compatibility Versus Performance

The published wheel is validated on Windows x86-64, Ubuntu x86-64, and Ubuntu
Arm64 GitHub-hosted runners. Those jobs prove installation and behavior, not
performance. Fixture outputs remain labeled `result_origin: test_fixture`,
including when they execute on native Arm64 CI.

Public synthetic PyTorch workloads validate semantics preservation,
low-resource guards, cache reuse, and hard-drift refusal on Arm and Windows.
They are not production training performance claims.

A separate clean measured T4g.small protocol used CPU-only PyTorch 2.13.0,
fixed effective batch size 32, 5,000-step slices, and 12 paired confirmations.
It measured a baseline median of 14,430.040 samples/s and a selected median of
33,274.800 samples/s. The ratio-of-medians improvement was 130.593958%, with
3.360982% selected variability. Four stalled DataLoader-worker trials remained
explicit failures. The result discloses that the burstable target ran with
zero CPU credit balance and T4g Unlimited surplus capacity. It is not combined
with C7g or llama.cpp results and is not a production training claim.

### Why It Matters

PerfRunbench makes performance work reviewable. Instead of handing a user a
single tuned command, it preserves:

- target and workload identity;
- source, binary, model, and protocol digests;
- raw outputs and artifact manifests;
- baseline and candidate distributions;
- failures, variability, and noise-band decisions;
- cleanup, canary, lease, and rollback state; and
- the exact boundary where a recommendation remains valid.

The result is a recommendation that can be audited, refused when the
environment changes, and reused safely when the evidence still matches.

## Inspiration

Arm systems performance work exposed a recurring problem: useful tuning
results require far more context than a throughput number. CPU identity,
binary provenance, build flags, placement, workload shape, and measurement
order all affect the result. PerfRunbench turns that manual engineering loop
into a reusable product.

## How It Was Built

PerfRunbench is implemented in Python 3.10+ and uses standard-library
components for its core orchestration. It includes:

- process execution, monitoring, affinity, and lifecycle cleanup;
- versioned JSON schemas and atomic manifest writes;
- local and system-OpenSSH Arm target control;
- workload adapters for generic commands, llama.cpp, serving, and optional
  PyTorch training;
- deterministic bounded search and paired statistical confirmation;
- environment-bound RecommendationBundles;
- Performance Passport generation and verification; and
- fixture, unit, fake integration, installed-wheel, Windows, Linux, and Arm64
  CI coverage.

The real serving result used AWS Graviton C7g and T4g instances, a pinned
llama-server build, and a model whose digest was fixed before comparison.

## Challenges

- Keeping fixture and CI output separate from measured Arm performance.
- Separating host capability from verified workload execution evidence.
- Preventing environment drift from silently reusing an invalid result.
- Preserving workload semantics while still exposing tunable knobs.
- Measuring a remote serving path without turning client or network behavior
  into a false server optimization.
- Restoring processes, ports, leases, and reversible settings after success,
  failure, timeout, or interruption.

## Accomplishments

- Published PerfRunbench v0.3.1 under the MIT License.
- Validated installed-wheel behavior on Windows x86-64, Linux x86-64, and
  Linux Arm64.
- Built config-free bounded tuning with hard quick, standard, and deep budgets.
- Generated valid partial reports when a tuning deadline expires.
- Added portable `platform validate` readiness checks.
- Validated real C7g build and runtime tuning with Performance Passport
  verification.
- Validated process-scoped recommendation apply, canary, lease, and rollback.
- Validated a real C7g server plus T4g remote-client serving comparison with
  608 measured requests and zero failures.
- Refused cross-target recommendation reuse under hard environment drift.
- Kept unsupported compiler, kernel, 64K-page, and network experiments from
  becoming fabricated performance recommendations.
- Demonstrated 1-to-15-thread Arm inference scaling and a workload-scoped
  concurrency-8 serving recommendation on C7g.4xlarge.

## What Was Learned

Reproducibility and restraint are features. A useful autotuner must be able to
say "inconclusive", preserve failures, and refuse an old recommendation. Small
deterministic searches with strong evidence can be more useful than large
uncontrolled sweeps.

The remote serving result also showed why workload shape matters. The same
validated candidate improved three declared concurrency-4 workloads, but the
claim still belongs to that exact target, model, topology, and protocol.

The larger-target result made that boundary more visible: four serving slots
helped interactive and decode-heavy concurrency-8 traffic, but harmed
prefill-heavy traffic. A useful recommendation therefore identifies eligible
workloads instead of declaring one profile globally best.

## Significant Updates Since June 10, 2026

The repository existed before the competition submission period. Significant
work completed during the period includes:

- Linux Arm64 local and SSH target inspection;
- a versioned remote worker and artifact protocol;
- llama.cpp build-profile and runtime autotuning;
- statistical confirmation and Performance Passport evidence;
- guarded recommendation validation and apply;
- PyTorch training and constrained-target validation;
- workload-aware llama-server serving autotuning;
- real dual-host C7g/T4g remote validation;
- config-free tuning presets and timeout-safe partial reports;
- portable platform validation; and
- cross-platform installed-wheel release validation.

Git history and the phase evidence documents preserve the implementation and
validation chronology.

## Built With

- Python
- Linux Arm64
- AWS Graviton C7g and T4g
- llama.cpp / llama-server
- PyTorch CPU wheels for optional synthetic training validation
- OpenSSH
- GitHub Actions
- JSON, Markdown, HTML, and SVG report artifacts

## Links

- Repository: `https://github.com/xuwwwww/PerfRunbench`
- GitHub Release: `https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1`
- Release validation:
  `https://github.com/xuwwwww/PerfRunbench/actions/runs/30234620475`
- Real remote-serving evidence:
  `https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md`
- Larger C7g capacity evidence:
  `https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G4XLARGE_REALISTIC_ARM_INFERENCE_VALIDATION.md`

## Limitations

- The headline result covers one C7g server, one T4g client, one model, one
  pinned runtime, concurrency 4, and three declared workloads.
- The same-protocol size-scaling result covers C7g.xlarge, C7g.2xlarge, and
  C7g.4xlarge with one loopback client CPU, one pinned model/runtime identity,
  and a declared concurrency-4 protocol.
- A separate larger-target result covers one C7g.4xlarge and declared c4/c8
  protocols. Its percentages are not combined with the size-scaling result.
- Broader Arm hardware, model, and concurrency validation remains future work.
- Synthetic training evidence validates mechanics and guards, not production
  training speed.
- Compatibility CI is not measured performance.
- PerfRunbench is distributed through GitHub Releases, not PyPI.
- BootTrace remains optional experimental capability work and does not
  participate in default candidate eligibility.
- The optional demo video is not included.

## Credits

PerfRunbench is an individual systems-performance engineering project. It uses
public tools, public or synthetic workloads, and sanitized evidence. The
submission contains no credentials, model weights, private host details, or
proprietary logs.

## Screenshot Captions

1. `PerfRunbench turns bounded search, statistical confirmation, and guarded reuse into one workflow.`
2. `On the tested C7g server, T4g remote client, pinned model/runtime, and concurrency-4 protocols, output throughput improved by 69.375%-86.490%.`
3. `The same scoped remote validation reduced p95 TTFT by 91.128%-98.001% across three workloads, with 608 measured requests and zero failures.`
4. `An actual public fixture run demonstrates config-free run --auto; fixture output is not performance evidence.`
5. `Hard drift, noise-band, cleanup, and no-evidence guards prevent unsafe recommendations.`
6. `Real Arm64 performance evidence is separated from installed-wheel compatibility validation on Windows, Linux x86-64, and Linux Arm64.`
7. `On one T4g.small and one public synthetic CPU PyTorch protocol, 12 paired confirmations measured 14,430 to 33,275 samples/s while preserving failures, CPU-credit state, and guard evidence.`
8. `On one C7g.4xlarge and a pinned Qwen2.5-7B/llama.cpp loopback protocol, fifteen-thread throughput scaled about 10.9x over one thread; eligible serving workloads improved while prefill regression and context failures were rejected.`
9. `With one pinned model and llama.cpp runtime, measured pp512/tg128 throughput scales across C7g.xlarge, C7g.2xlarge, and C7g.4xlarge while the largest thread point reserves one CPU for the client.`
10. `Across three tested C7g sizes, four-slot continuous batching improved eligible interactive and decode serving under the same concurrency-4 loopback protocol; prefill-heavy traffic was rejected separately.`
