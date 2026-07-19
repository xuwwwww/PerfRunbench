# ArmTune Limitations

## Current Phase 0 Limitations

- No ArmTune production code exists yet.
- No `autotuneai arm` commands exist yet.
- No local or SSH Arm target has been inspected by ArmTune yet.
- No llama.cpp binary has been inspected by ArmTune yet.
- No ArmTune benchmark has been executed.
- No measured Arm performance result exists in this repository.
- No Arm Performance Passport has been generated.
- No Performix integration exists.

## MVP Out of Scope

- Automatic cloud instance provisioning.
- Kubernetes orchestration.
- Hosted SaaS dashboards.
- User accounts.
- Custom Neon or SVE assembly.
- A new inference engine.
- Model training.
- Automatic quantization.
- ONNX Runtime, ExecuTorch, or vLLM adapters before the llama.cpp vertical
  slice is stable.
- Power measurement without reliable hardware support.
- Cost estimates without verified current pricing.
- x86-vs-Arm comparisons as the primary product result.

## Result Claim Limitations

Existing PerfRunbench benchmark results are useful prior work but are not
ArmTune production Arm results. They must not be used as evidence that an Arm
target, Arm optimized path, KleidiAI, SVE, I8MM, BF16, SME, or Performix was
used.

Any future result missing raw artifacts, provenance, or comparability checks
should be labeled unverified.

## Phase 2A Limitations

- Phase 2A is a minimal llama-bench execution slice, not an optimization
  system.
- Phase 2A runs one fixed configured invocation only.
- Phase 2A does not implement candidate search, warmup logic, paired
  baselines, confirmation runs, or noise-band decisions.
- Phase 2A does not implement a quality gate.
- Phase 2A does not verify KleidiAI or any other optimized Arm path.
- Phase 2A parser support is limited to explicit llama-bench table/text
  strategies covered by fixtures. Unknown formats are reported as
  `unsupported_format` rather than converted into metrics.
- Phase 2A fake executable tests are `test_fixture` results, not measured Arm
  results.
- Real production Arm benchmark validation remains out of scope for Phase 2A.

## Phase 2B Limitations

- GitHub native Arm64 CI proves packaging, CLI, host inspection, focused tests,
  and fixture benchmark plumbing on an Arm64 Linux runner only.
- The completed GitHub run
  `https://github.com/xuwwwww/PerfRunbench/actions/runs/29270842566` validated
  commit `2074dfac80e699ba22178774b28c66e79afae475` on Linux `aarch64`
  kernel `6.17.0-1018-azure`.
- Downloaded artifact validation passed for artifact
  `armtune-arm64-smoke-artifacts`; 13 artifacts were checked and
  `result_origin` remained `test_fixture`.
- GitHub native Arm64 CI does not prove real model performance, inference
  optimization, KleidiAI usage, Performix support, or competition benchmark
  results.
- Workflow timing is not benchmark data.
- Fixture values such as `321.5 tokens/s` prefill and `87.25 tokens/s` decode
  are fake workload constants and must not be reported as Arm performance.
- Fake llama-bench output remains `test_fixture` on real Arm hardware.
- Phase 2B does not use AWS, external SSH servers, cloud provisioning, real
  models, `llama-server`, time to first token, candidate search, quality gates,
  build profiles, Performix, or Performance Passport generation.
- Real-Arm smoke scaffolding is skipped by default and requires explicit
  user-supplied binary and model paths plus acknowledgement.
- Phase 2C must provide a real model, real target, and explicit measured-data
  handling before any preliminary measured result is run.

## Phase 2C Limitations

- Phase 2C produced one preliminary measured baseline on a user-provisioned
  burstable AWS T4g instance. It validates the real SSH Arm64 execution
  pipeline, not stable production performance.
- The Phase 2C evidence is recorded in
  `docs/armtune/results/PHASE_2C_AWS_GRAVITON2.md`.
- The run `20260716_231915_147773` used a real `llama-bench` binary and a real
  GGUF model on Linux `aarch64`, with `result_origin: measured`.
- Parsed preliminary throughput was `43.8 tokens/s` prefill and
  `30.13 tokens/s` decode. These values are pipeline-validation measurements,
  not final competition benchmark results.
- Artifact digest validation passed after download, with 13 artifacts checked.
- Phase 2C does not prove optimization, KleidiAI execution, build-profile
  improvement, thread-search improvement, quality preservation, stable
  production performance, Performix support, Performance Passport generation,
  or formal benchmark conclusions.
- Phase 2C did not run candidate search, repeated confirmation, paired
  baselines, quality gates, `llama-server`, streaming, TTFT, or automated cloud
  instance control.

## Phase 3A Limitations

- Phase 3A is limited to repeated `llama-bench` measurement and
  performance-only candidate search.
- Phase 3A recommendations use `quality_gate_status: not_evaluated` and do not
  prove semantic quality preservation.
- The Phase 3A.1 evidence is recorded in
  `docs/armtune/results/PHASE_3A_REAL_SEARCH_VALIDATION.md`.
- The run `20260718_001757_718789` validated repeated benchmark, staged search,
  statistics, conservative recommendation, SSH worker, provenance, parser, and
  artifact-validation plumbing against a real `llama-bench` workload on Linux
  `aarch64`.
- The run completed 34 measured repetitions with 0 failed trials and passed
  artifact digest validation for 34 repetition manifests covering 442 artifact
  records.
- The selected observed candidate was `affinity-spread-0_1`, but the
  recommendation was `inconclusive` because the primary-metric delta
  `0.116029%` was inside the configured 5% noise band.
- Phase 3A.1 used a user-provisioned burstable AWS Graviton2 development
  instance. The results are pipeline-validation measurements, not stable
  production performance or formal competition benchmark results.
- The Phase 3A.1 validation run predates Phase 3A.2 candidate controls. On the
  2-vCPU validation target, that historical run planned five candidates rather
  than the desired three-or-fewer smoke profile.
- Phase 3A.2 adds `search.max_candidates` and `search.affinity_modes` for
  bounded smoke profiles, but no new real-target benchmark evidence has been
  collected with those controls yet.
- Phase 3A.2 `max_candidates` limits unique tested candidate configurations,
  not warm-up runs, paired-baseline insertions, confirmation runs, or process
  execution count.
- Phase 3A does not implement `llama-server`, HTTP, streaming, TTFT, BootTrace,
  build-profile comparison, KleidiAI claims, Performix, Performance Passport
  generation, cloud provisioning, or automatic instance control.
- Phase 3A defaults are intended to make the search protocol testable and
  auditable, not to establish final stable production performance.

## Phase 3B Planned Limitations

- Phase 3B is reserved for `llama-server` lifecycle management,
  local-on-target HTTP measurement, streaming TTFT, end-to-end latency,
  structured BootTrace output, and deterministic quality validation.
- Phase 3B will be required before ArmTune can reject a faster candidate because
  real generation output is invalid, malformed, or semantically unacceptable.
- Phase 3A.2 candidate-count and affinity guardrails must not be described as
  Phase 3B server-inference or quality-gate validation.

## Phase 3B.1 Limitations

- Phase 3B.1 implements a `llama-server` benchmark foundation.
- Phase 3B.1-R adds one bounded real-target server validation run on a
  user-provisioned AWS T4g / Graviton2 Linux Arm64 development instance.
- The Phase 3B.1-R evidence is recorded in
  `docs/armtune/results/PHASE_3B1_REAL_SERVER_VALIDATION.md`.
- The run `20260719_004451_487119` validated server lifecycle, target-local
  loopback HTTP measurement, OpenAI-compatible SSE parsing, generated-output
  preservation, cleanup, and artifact digest validation against a real
  `llama-server` binary and a real GGUF model on Linux `aarch64`, with
  `result_origin: measured`.
- The run completed 6 measured requests with 0 failed requests and passed
  artifact digest validation for 48 artifact records.
- The observed median TTFT was `41.105 ms`, and the observed median
  end-to-end latency was `762.659 ms`. These are bounded pipeline-validation
  measurements from a burstable development instance, not final competition
  benchmark results.
- Phase 3B.1 does not download models, open public server ports, provision
  cloud resources, or change EC2 security groups.
- Phase 3B.1 request timing is target-local, but fake-server timings are
  fixture data and must not be reported as Arm performance.
- Phase 3B.1 records `quality_gate_status: not_evaluated`; successful nonempty
  generated output is not a semantic quality pass.
- Phase 3B.1 does not add BootTrace validation, deterministic semantic scoring,
  `llama-server` candidate search, performance recommendation, build-profile
  comparison, KleidiAI verification, Performix, or Performance Passport
  generation.

## Phase 4 Limitations

- Phase 4 implements a deterministic public synthetic BootTrace quality suite
  and quality-aware `llama-server` candidate search.
- Local fake E2E evidence validates controller orchestration, server lifecycle,
  stream parsing, quality rejection, conservative recommendation state, and
  artifact digest validation without Arm hardware, SSH, network, model weights,
  root privilege, or cloud credentials.
- Fake-server quality and timing values are fixtures and must not be reported
  as Arm performance.
- Phase 4-R completed a bounded real-target run on a user-provisioned AWS
  T4g / Graviton2 Linux Arm64 development target. The run validated real
  server execution, streaming latency measurement, deterministic quality
  evaluation, quality-aware rejection, cleanup, and artifact digest validation.
- Phase 4-R did not produce an eligible candidate. All three measured
  candidates were rejected by the quality gate, so Phase 4 has not yet shown a
  quality-preserving measured recommendation on real Arm hardware.
- The Phase 4-R generated outputs were preserved as artifacts and showed
  systematic strict-output failures, including Markdown-fenced JSON and
  failure cases mislabeled as `success`. The validators were not weakened to
  force a pass.
- Phase 4 uses a public synthetic BootTrace suite. It does not validate private
  firmware logs, production customer logs, or NDA-protected data.
- Phase 4 does not compare build profiles, verify KleidiAI, modify llama.cpp,
  search quantization, run concurrency/load tests, integrate Performix,
  generate a complete Performance Passport, or establish final competition
  benchmark conclusions.

## Phase 5 Limitations

- Phase 5 implements the local competition-evidence MVP: quality diagnostics,
  optional fenced-JSON normalization, structured capability reporting,
  BootTrace prompt-contract digests, quality calibration, declarative build
  profiles, profile comparison, and Performance Passport generation and
  verification.
- Phase 5 local fake E2E evidence remains `result_origin: test_fixture`; its
  fixture timings and fixture build artifacts must not be reported as Arm
  performance or as llama.cpp optimization results.
- Phase 5 does not automatically build llama.cpp from source on the target.
  `build-plan` is non-executing, and `build-profile` currently records and
  inspects existing build artifacts rather than running arbitrary build
  commands.
- `arm-optimized` remains `requested_but_unverified` unless compile, link,
  runtime, or binary evidence supports an optimized path. A requested
  `LLAMA_KLEIDIAI=ON` setting alone is not evidence that KleidiAI executed.
- Phase 5-R recorded a bounded measured quality-calibration run on a
  user-provisioned AWS T4g / Graviton2 Linux Arm64 development target. The
  evidence is recorded in
  `docs/armtune/results/PHASE_5_COMPETITION_EVIDENCE.md`.
- Phase 5-R validated real SSH `llama-server` execution, strict BootTrace
  quality calibration, cleanup, structured capability inspection, and artifact
  digest validation, but the model/profile remained `unsuitable` with
  `quality_gate_status: failed`.
- Phase 5-R therefore did not run measured profile comparison and did not
  produce an eligible preliminary Performance Passport for a real Arm target.
- Phase 5 does not add Performix profiling, concurrency/load testing, model
  download or substitution, cloud provisioning, automatic EC2 control,
  quality-suite expansion beyond the public synthetic BootTrace suite, or final
  competition benchmark conclusions.

## Phase 6 Planned Limitations

- Phase 6 uses the existing Phase 5 pipeline and should not add framework
  features merely to make one model pass.
- Phase 6 model qualification is model/profile-specific. An unsuitable model
  is not evidence that the ArmTune pipeline failed, and a suitable model is
  not evidence of global model quality.
- Qwen3-4B was unsuitable under the tested Phase 6A contracts, including a
  private `/no_think` prompt-contract variant. Validators and canonical
  expected answers must not be weakened in response.
- Qwen2.5-7B Q4_K_M was unsuitable under the tested strict public BootTrace
  quality contract even though every request completed and every output was
  raw-valid JSON. Semantic failures such as wrong root cause, wrong status,
  missing required evidence, and invalid numeric values were not normalized
  away.
- A future model/profile qualification must pass before build-profile
  comparison, performance recommendation, or measured Performance Passport
  generation.
- C7g and T4g measurements are not comparable. Any C7g result needs its own
  baseline and must not be compared to earlier T4g evidence.
- Phase 6 does not automatically provision, resize, start, stop, or alter
  cloud resources beyond explicitly approved operations.
- Phase 6 does not validate private firmware logs, customer data,
  source-level llama.cpp optimizations, Performix evidence, stable production
  SLOs, or final competition benchmark conclusions.

## Phase 7 Planned Limitations

- Phase 7 prompt-contract calibration does not weaken the held-out BootTrace
  evaluation validators or canonical answers.
- The development calibration suite is not final qualification evidence and
  must not be mixed into the held-out 5/5 pass count.
- Versioned prompt contracts can reduce prompt ambiguity, but they do not prove
  model capability until a real model/profile passes the held-out evaluation
  suite with artifact validation and cleanup evidence.
- SSH quality calibration uploads only the rendered prompt suite into the
  versioned remote run directory. It does not upload model weights,
  credentials, private logs, or arbitrary workload files.
- Qwen2.5-7B was re-run against the three Phase 7 contract variants on an
  explicitly approved C7g target. The winning development-suite contract did
  not qualify on the held-out suite; Qwen2.5-7B remained `unsuitable`.
- Qwen2.5-14B Q4_K_M passed the development sanity suite under the same frozen
  contract, but it also remained `unsuitable` on the held-out suite with only
  2/5 required cases passed.
- Gemma 3 12B QAT Q4_0 was not evaluated because GGUF file access was gated
  for the available target environment. This is an access limitation, not a
  quality result.
- Build-profile comparison, Performance Passport generation, Performix
  evidence, KleidiAI claims, and final competition conclusions remain blocked
  until a model/profile is `suitable`.

## Phase 8 Planned Limitations

- Phase 8 preflight did not start AWS and did not execute a new model
  qualification run.
- The Qwen2.5-7B and Qwen2.5-14B held-out failure matrix is diagnostic
  evidence only. It must not be used to weaken validators, rewrite canonical
  answers, or tune prompts against the held-out cases.
- Both Qwen2.5 models failed the same three held-out cases despite 5/5 request
  completion, 5/5 raw-valid JSON output, zero invented evidence, cleanup
  success, and artifact-validation success. This leaves model/profile
  suitability blocked.
- Gemma 3 12B QAT Q4_0 remains unevaluated. The current controller
  environment could see repository metadata but received HTTP `401` when
  resolving the GGUF file, and no Hugging Face token was present.
- AWS lifecycle remains user-controlled for Phase 8. ArmTune must not
  automatically start, stop, provision, resize, or replace cloud resources.
- Runtime tuning, build-profile comparison, Performance Passport generation,
  KleidiAI claims, Performix evidence, and final competition conclusions remain
  blocked until an unchanged held-out qualification run reaches `suitable`.
