# PerfRunbench Competition Submission Facts

This file is the claim source of truth for the Arm AI Optimization Challenge
submission. Use these facts consistently across Devpost copy, screenshots,
slides, release notes, and any later demo.

## Project Identity

- Project title: `PerfRunbench`
- Submission version: `0.3.1`
- Challenge track: `Cloud AI`
- Product definition: guarded performance autotuning for repeatable workloads
- Repository: `https://github.com/xuwwwww/PerfRunbench`
- Release: `https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1`
- License: MIT
- Submission language: English

ArmTune is the Arm inference subsystem inside PerfRunbench. llama.cpp is the
first inference adapter, not the product identity. BootTrace is preserved only
as an optional experimental capability workload and is not part of the default
tuning or recommendation path.

## One-Line Description

Guarded autotuning for repeatable AI workloads on Arm, with bounded search,
statistical confirmation, environment-bound reuse, and auditable evidence.

## Primary Flow

```text
inspect target
-> validate workload and environment
-> establish baseline
-> search a bounded candidate plan
-> confirm the best observed eligible candidate
-> generate a Performance Passport
-> validate, canary, apply, lease, and roll back safely
```

The config-free public entry point is:

```bash
autotuneai run --auto --preset quick -- python app.py
```

The serving path uses:

```bash
autotuneai serve workload show interactive-short --json
autotuneai serve recommend \
  --workload interactive-short \
  --concurrency 4 \
  --recommendation-class maximum_throughput_remote
```

## Release And Compatibility Evidence

- Public version: `0.3.1`
- GitHub Actions release validation run:
  `https://github.com/xuwwwww/PerfRunbench/actions/runs/30234620475`
- Validated installed-wheel runners:
  - `windows-latest`
  - `ubuntu-latest`
  - `ubuntu-24.04-arm`
- Release workflow head: `5ccab60b0db61d7d8473e2254eccd2c2457c5714`

Installed-wheel CI proves packaging and compatibility behavior. It is not
measured performance evidence. Fixture values remain `result_origin:
test_fixture`, including on a native Arm64 CI runner.

## Release Artifacts

```text
perfrunbench-0.3.1-py3-none-any.whl
fe7a598305bfcc141ec9b20531f1083ad7254694d58d1a676cd124be55eaf33f

perfrunbench-0.3.1.tar.gz
83bf2f936c1e63f3c76768adc40d1e4ade550eb8373bf3026028714833609d8d
```

These SHA-256 digests were independently verified against downloaded GitHub
Release assets.

## Headline Measured Arm Serving Evidence

The Phase 21D2B result is a real, protocol-specific remote-serving comparison:

- Server: AWS C7g `c7g.2xlarge`, Linux Arm64, Neoverse-V1
- Remote client: AWS T4g `t4g.small`, Linux Arm64, Neoverse-N1
- Model: Qwen2.5-7B Instruct Q4_K_M
- Runtime: pinned llama-server build
- Traffic: closed-loop, concurrency 4
- Network path: direct private target-to-target path
- Candidate: `slots4-cont-on`
- Measured requests: 608
- Retries: 0
- Request failures: 0
- Quality failures: 0

| Workload | J1 baseline output req/s | Candidate output req/s | Throughput improvement | Baseline p95 TTFT | Candidate p95 TTFT | p95 TTFT reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `interactive-short` | `17.032` | `31.355` | `84.092%` | `3055.682 ms` | `271.088 ms` | `91.128%` |
| `decode-heavy` | `17.046` | `31.790` | `86.490%` | `10126.795 ms` | `202.424 ms` | `98.001%` |
| `prefill-heavy` | `16.563` | `28.054` | `69.375%` | `3967.424 ms` | `341.693 ms` | `91.388%` |

Approved headline wording:

```text
On the tested C7g Arm64 server, T4g Arm64 remote-client path, Qwen2.5-7B
Q4_K_M model, closed-loop concurrency-4 workloads, and pinned measurement
protocol, the validated slots4-cont-on candidate improved output throughput
by 69.375%-86.490% and reduced p95 TTFT by 91.128%-98.001%.
```

This is not a universal speedup claim and must not be generalized to other
models, hardware, concurrency levels, runtimes, or network paths.

## Larger C7g Capacity Evidence

A separate C7g.4xlarge loopback experiment used 16 logical Neoverse-V1 CPUs,
one reserved local client CPU, 15 server CPUs, the same model digest, and a
fresh pinned llama.cpp binary digest.

The separate pp512/tg128 protocol scaled from 1 to 15 threads:

```text
prefill: 6.761 -> 73.669 tokens/s, 10.897x
decode:  2.759 -> 30.089 tokens/s, 10.907x
```

Full-capacity concurrency-8 paired confirmation measured:

| Workload | Baseline output tok/s | `slots4-cont-on` output tok/s | Delta | Eligibility |
| --- | ---: | ---: | ---: | --- |
| `decode-heavy` | 25.779 | 46.488 | +80.331% | eligible |
| `interactive-short` | 25.600 | 43.712 | +70.749% | eligible |
| `prefill-heavy` | 19.734 | 15.585 | -21.025% | rejected |

The run recorded 769 measured HTTP requests, 17 fresh server starts, passed
cleanup, and 278 independently revalidated downloaded artifacts. It also
preserved HTTP 400 context-limit failures for the eight-slot long-prefill
candidate.

This result is loopback evidence, not remote-network evidence. It is not a
direct C7g.2xlarge versus C7g.4xlarge comparison. Hardware-scaling and
runtime-tuning percentages remain separate.

## Same-Protocol C7g Size-Scaling Evidence

A newer experiment used one pinned Qwen2.5-7B Q4_K_M artifact, llama.cpp
runtime bundle, binary identity, client/server CPU-allocation rule, and
concurrency-4 loopback protocol across `c7g.xlarge`, `c7g.2xlarge`, and
`c7g.4xlarge`.

| Target | Server threads | Decode baseline | Decode selected | Within-target delta |
| --- | ---: | ---: | ---: | ---: |
| `c7g.xlarge` | 3 | 6.237 tok/s | 12.136 tok/s | +94.585% |
| `c7g.2xlarge` | 7 | 13.626 tok/s | 25.876 tok/s | +89.901% |
| `c7g.4xlarge` | 15 | 25.475 tok/s | 46.186 tok/s | +81.296% |

The formally confirmed interactive-short deltas were +84.866%, +81.040%, and
+73.324%. Four-slot prefill-heavy screening regressed by -38.977%, -38.039%,
and -35.310%, so that workload was excluded on all three sizes.

The three runs recorded 1,047 measured HTTP requests, 30 fresh server starts,
zero request or quality failures, complete cleanup, and 514 successful
controller-side digest checks after download.

Approved wording:

```text
Under the tested C7g size, pinned model and llama.cpp runtime, single-host
concurrency-4 protocol, and declared guards, four-slot continuous batching was
the best observed eligible configuration for interactive-short and
decode-heavy serving.
```

Do not combine the within-target tuning deltas with hardware-size scaling.
These are loopback results, not remote-network evidence.

## Separate Measured Protocols

Do not add, multiply, average, or otherwise combine these results with the
remote-serving result.

### C7g llama.cpp Build Profiles

| Profile | Decode median |
| --- | ---: |
| `portable-baseline` | `10.825 tokens/s` |
| `arm-native` | `17.530 tokens/s` |
| `arm-optimized` | `17.380 tokens/s` |

### C7g llama.cpp Runtime Search

| Item | Decode median |
| --- | ---: |
| baseline | `13.630 tokens/s` |
| selected | `17.480 tokens/s` |
| observed relative improvement | `28.246515%` |

### C7g llama-server Secondary Measurement

| Profile | TTFT median |
| --- | ---: |
| `portable-baseline` | `97.260 ms` |
| `arm-native` | `62.826 ms` |

### Synthetic Training Guard Validation

Synthetic PyTorch workloads validated tuning mechanics and guard behavior on
C7g Arm64, T4g Arm64, and Windows x86-64. Those values are not production
training performance claims and must not be compared across targets.

A clean measured T4g.small visual-evidence protocol added stronger
distribution and guard evidence:

```text
baseline median: 14,430.039978 samples/s
selected median: 33,274.800277 samples/s
ratio-of-medians improvement: 130.593958%
median paired improvement: 128.172176%
paired confirmations: 12
selected variability: 3.360982%
failed DataLoader-worker trials preserved: 4
```

This claim is limited to one T4g.small, CPU-only PyTorch 2.13.0, the public
synthetic workload, fixed effective batch size 32, and the recorded slice and
measurement protocol. CPU credit balance was zero throughout the clean run;
the run used T4g Unlimited surplus capacity, which is disclosed in the
evidence. It is not combined with C7g or llama.cpp results.

## Safety And Guard Evidence

- Hard environment drift prevents recommendation reuse.
- `run --best` never retunes.
- Cache insertion requires eligible evidence.
- Failed trials are not converted to zero.
- Differences inside the configured noise band remain inconclusive.
- Apply is process-scoped and covered by validation, canary, lease, cleanup,
  and rollback.
- System and kernel experiments are advanced opt-in.
- Unsupported or unevidenced mechanisms are recorded without manufacturing a
  recommendation.
- Network tuning was not executed when prerequisite bottleneck evidence was
  absent.

## Significant Updates During The Submission Period

The challenge submission period began on June 10, 2026. The repository existed
before that date. Significant work added during the submission period includes:

- ArmTune Linux Arm64 local and SSH target inspection
- versioned worker protocol and artifact validation
- llama.cpp build-profile and bounded runtime autotuning
- repeated paired comparison and noise-band classification
- Performance Passport generation and verification
- guarded recommendation validation, canary, lease, apply, and rollback
- PyTorch training adapter and low-resource Arm validation
- workload-aware llama-server serving autotuning
- real C7g server plus T4g remote-client validation
- larger C7g thread scaling and workload-aware loopback serving validation
- config-free `run --auto` with quick, standard, and deep hard budgets
- valid partial reports on timeout
- portable `platform validate`
- installed-wheel validation on Windows, Linux x86-64, and Linux Arm64

The repository history and evidence documents preserve these changes.

## Evidence Documents

- `docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md`
- `docs/results/C7G4XLARGE_REALISTIC_ARM_INFERENCE_VALIDATION.md`
- `docs/results/PHASE_21D3_SERVING_AUTOTUNING_PRODUCTIZATION.md`
- `docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md`
- `docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md`
- `docs/results/PHASE_14B_REAL_ARM_PYTORCH_VALIDATION.md`
- `docs/results/PHASE_15A_T4G_LOW_RESOURCE_VALIDATION.md`
- `docs/results/PHASE_15B_T4G_REAL_PYTORCH_VALIDATION.md`
- `docs/results/PHASE_15C_WINDOWS_X86_PYTORCH_VALIDATION.md`
- `docs/results/T4G_TRAINING_VISUAL_EVIDENCE.md`
- `docs/results/VALIDATED_PERFORMANCE_SUMMARY.md`
- `docs/results/VALIDATION_MATRIX.md`
- `docs/releases/v0.3.1.md`

## Required Claim Boundaries

Do not:

- claim global optimality;
- combine improvements from different protocols;
- compare `samples/s` with `tokens/s` or requests/s;
- rank C7g, T4g, and Windows using unrelated measurements;
- describe synthetic or fixture output as production performance;
- claim compiler, 64K-page, kernel, or network tuning improved performance;
- claim KleidiAI without verified build and runtime evidence;
- present CI compatibility as measured Arm performance;
- expose private hosts, keys, model paths, or artifact locations.

Use this recommendation wording:

```text
Best observed eligible configuration under the tested target, model, workload,
runtime, concurrency, topology, and measurement protocol.
```

## Current Limitations

- Headline serving evidence covers one C7g server, one T4g client, one model,
  one pinned runtime, concurrency 4, and three declared workloads.
- C7g and T4g are AWS Graviton targets; broader Arm server validation remains
  future work.
- PyTorch evidence uses public synthetic CPU workloads.
- The project is published through GitHub Releases, not PyPI.
- A demo video is optional and is not included in this submission package.
