# Research direction: cost-aware, allocation-aware autotuning

Status: design proposal, not an implemented or experimentally validated decision policy.

## Research question

Under which workload, CPU allocation and competition conditions can a low-cost
capacity rule achieve a sufficiently good configuration, and when can bounded
search repay its execution and validation cost?

## Decision choices

1. **Reuse** a previously eligible configuration only when hard identity
   matches and a contention-sensitive validation policy permits reuse.
2. **Rule**: use a runtime-specific conservative rule (initially both llama.cpp
   thread pools equal to CPUs permitted to an exclusive LLM set).
3. **Probe**: measure rule versus an alternative with short, matched, randomized
   pilot blocks and verify output correctness.
4. **Search**: expand to a budgeted candidate space only if evidence suggests
   additional improvement can plausibly repay the added cost.
5. **Abstain**: reject an unsafe or insufficiently validated recommendation;
   retain a separately verified baseline, without an improvement claim.

## Cost model

Let H be the expected number of comparable future executions, C the incremental
elapsed search+validation cost, T_r the measured rule runtime, and T_s the
searched-profile runtime. Search pays back in elapsed time when

    H * (T_r - T_s) > C

after accounting for workload-specific measurement and reuse costs. Use
confidence bounds on T_r-T_s, not only a point estimate. If H is unknown, report
the break-even horizon rather than assuming unlimited reuse. Treat energy,
money, and worker interference as separate objectives unless measured and
explicitly weighted.

A pilot should expose uncertainty; when it cannot rule out meaningful benefit,
report uncertainty and optionally request a larger budget. A failed safety or
correctness check blocks recommendation regardless of payback.

## Planned ablations

- Full search versus capacity rule versus cost-aware decision policy.
- Exclusive 8-core split versus shared placement under at least three worker
  occupancy levels, with identical LLM CPU mask where possible.
- Cold fixed-input 64/512-token inference and corrected indexed file tasks.
- Reuse with unchanged environment, modest memory fluctuations, changed CPU
  mask, and changed competitor load.
- End-to-end wall cost including search, monitoring, canaries and launches.
- Outcomes: end-to-end runtime, recommendation quality/regret versus
  offline best-tested setting, throughput and latency, total tuning cost,
  decision error rate, retuning frequency and correctness.

## Implementation sequence

1. Repair caller-mask restoration and baseline-selected execution behavior.
2. Reduce accidental canaries caused by negligible available-memory drift.
3. Add explicit decision records with a reason, cost estimate and threshold.
4. Establish a no-decision fallback and audit every shortcut.
5. Run randomized confirmatory campaigns. Retain zero-gain and failure outcomes.

## Methodological constraints

The 495.1-second initial auto invocation versus a capacity rule is evidence
that the existing search can be expensive. It does **not** demonstrate that a
proposed decision policy is better. Compare policies on matched independent
test blocks and report search-selection bias and workload reuse assumptions.
The known caller-affinity fix is currently implemented by temporary
caller-affinity changes during spawn; concurrent parent-thread launches need
separate safety testing or a child-only affinity executor.
