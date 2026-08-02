# PerfRunbench ArmTune Demo Script

Target duration: 3 minutes 15 seconds. The product story is ArmTune. Generic
PerfRunbench capabilities are supporting infrastructure, not the opening topic.

Do not access AWS during recording. Use the local guided CLI for live UX and
the committed measured evidence for performance claims. Label the two evidence
types on screen.

## 0:00-0:20 - Problem And Product

Screen: challenge deck cover, then the repository README.

Narration:

> AI inference settings that are faster for one Arm workload can be slower for
> another. ArmTune inspects the target, runs bounded measurements, confirms a
> candidate statistically, and keeps the recommendation tied to its evidence.

## 0:20-0:50 - Guided ArmTune Entry

Screen: a clean local terminal.

```powershell
autotuneai guide
autotuneai init --adapter llama-bench --target ssh --output .autotuneai/demo-armtune --preset quick
autotuneai guide --project .autotuneai/demo-armtune
```

Narration:

> The guided entry creates private starter configs with strict SSH checking,
> raw artifact preservation, a four-candidate cost cap, confirmation runs, and
> a five-percent noise band. Guide reports exactly what remains before target
> access. This live run creates configuration only; it is not an Arm
> performance measurement.

Show:

- `result_origin: unverified` in a starter file;
- `readiness: needs_configuration`;
- the five workflow steps;
- no network or workload process.

## 0:50-1:15 - Terms Before Numbers

Screen:

```powershell
autotuneai explain interactive-short
autotuneai explain decode-heavy
autotuneai explain prefill-heavy
```

Narration:

> Interactive-short represents a normal assistant turn. Decode-heavy uses a
> short prompt and long output. Prefill-heavy uses a long prompt and short
> output. TTFT measures responsiveness before the first token. ArmTune keeps
> these traffic shapes separate because one setting is not universally best.

## 1:15-1:40 - Target And Evidence Boundary

Screen: prepared output or README commands.

```powershell
autotuneai doctor --target-config <private-target.json> --require-production --summary
autotuneai inference inspect-host --target-config <private-target.json> --summary
```

Narration:

> The controller can be Windows or x86, but production measurements require a
> real Linux Arm64 target. Host features remain separate from proof that the
> workload actually used an optimized path.

Do not expose the real host, identity-file path, model path, or SSH command.

## 1:40-2:15 - Measured Before And After

Screen: C7g cross-size thread-scaling and serving before/after charts.

Narration:

> Under one pinned Qwen2.5-7B and llama.cpp protocol, decode serving improved
> from 6.237 to 12.136 tokens per second on C7g.xlarge, 13.626 to 25.876 on
> C7g.2xlarge, and 25.475 to 46.186 on C7g.4xlarge. These are within-target
> runtime-tuning deltas, not a universal Arm speedup and not hardware scaling
> percentages added together.

On-screen label:

```text
PREVIOUSLY MEASURED REAL ARM EVIDENCE
same model, runtime, CPU allocation rule, concurrency, and protocol
```

Keep the earlier Phase 9 build/runtime protocol visible as a separate evidence
card, not as part of the serving percentage:

```text
Phase 9 build profile: portable 10.825 tok/s
                       arm-native 17.530 tok/s
                       arm-optimized 17.380 tok/s
Phase 9 runtime search: 13.630 tok/s -> 17.480 tok/s
Observed within-protocol improvement: 28.246515%
Secondary llama-server TTFT: 97.260 ms -> 62.826 ms
Passport: eligible_preliminary | verification passed
guarded apply validation: passed
```

These build, runtime-search, secondary-server, and C7g serving results use
different protocols and must not be combined into one improvement claim.

## 2:15-2:40 - The Guard Matters

Screen: workload guard chart.

Narration:

> Four-slot continuous batching passed paired confirmation for interactive and
> decode traffic. The same family regressed prefill-heavy screening by 35.3 to
> 39.0 percent and worsened TTFT, so ArmTune rejected it for that workload. The
> product result is not just a bigger number; it is a scoped recommendation and
> preserved negative evidence.

## 2:40-3:00 - Passport And Safe Reuse

Screen: architecture flow and Performance Passport evidence.

Narration:

> A Performance Passport binds the selected result to target, model, binary,
> protocol, distributions, guards, and artifact digests. Environment drift,
> parser failure, cleanup failure, or a result inside the noise band blocks
> reuse. Apply remains process-scoped, validated, canaried, leased, and
> rollback-capable.

## 3:00-3:15 - Close

Screen: GitHub repository and final slide.

Narration:

> ArmTune makes Arm AI inference tuning repeatable: inspect, measure, search,
> confirm, passport, and reuse only when the evidence still matches.

## Recording Integrity Checklist

- Label live starter/fixture output separately from measured Arm evidence.
- Do not show AWS Console, public or private IP addresses, PEM paths, tokens,
  private configs, model paths, or raw private artifacts.
- Do not combine percentages from different protocols.
- Do not claim global optimality, generic Arm acceleration, or KleidiAI without
  sufficient build and runtime evidence.
- Keep the repository and release URL visible in the closing frame.
