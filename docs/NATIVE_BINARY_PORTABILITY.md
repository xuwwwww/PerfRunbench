# Native Binary Portability

Native CPU builds can improve a tested workload, but they also narrow where the
binary may safely run. PerfRunbench keeps native-binary compatibility separate
from workload identity and recommendation cache identity.

## Scope States

```text
portable
architecture_bound
feature_bound
cpu_model_bound
exact_target_only
```

The portability state is evidence, not a marketing claim.

## Required Evidence

A CPU-specific binary recommendation retains:

- source commit
- compiler identity
- effective CPU flags
- detected CPU architecture
- required CPU features
- CPU model or family policy
- binary SHA-256
- build target fingerprint
- verified target fingerprints
- startup-smoke requirement

Unavailable CPU information remains `unknown` or `Not available`; it is not
filled by guessing from an instance name.

## Reuse Guard

Before a cached native binary can launch on a current target, PerfRunbench
checks:

```text
architecture
required CPU feature set
CPU model policy
binary SHA-256
verified target fingerprint when exact-target scoped
startup smoke when policy requires it
```

Hard drift examples:

- x86_64 target for an aarch64 binary
- required feature missing
- Neoverse-V1 target for a Neoverse-N1 CPU-model-bound binary
- binary digest mismatch
- malformed required-feature evidence
- startup smoke failure or illegal instruction

Portable binaries retain broader compatibility, but still keep source,
compiler, and digest evidence.

## Relationship To RecommendationBundle

RecommendationBundle validation still checks workload and environment identity.
Native binary portability is an additional guard for compiled artifacts.
Passing one does not imply passing the other.

The default safe behavior is refusal before workload launch when compatibility
cannot be proven.

## Phase 20 C7g Guard Outcome

Phase 20 built CPU-aware `llama.cpp` binaries on a C7g / Neoverse-V1 target.
The `explicit-neoverse-v1` build was eligible for comparison on that target,
while KleidiAI-requested builds were rejected from recommendation eligibility
because runtime backend evidence was not proven.

This preserves the portability boundary:

```text
CPU-specific binary evidence is target/protocol evidence.
It is not portable to Neoverse-N1, x86_64, or another target by name.
Requested optimized backend flags do not create portability or speedup claims.
```

Any future reuse of a Phase 20 binary must validate architecture, CPU feature
requirements, CPU model policy, binary SHA-256, source commit, and startup
smoke before workload launch.

