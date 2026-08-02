# PerfRunbench Arm AI Challenge Deck

This is a standalone Slidev project for the Arm AI Optimization Challenge,
Cloud AI category.

## Run

```bash
npm install
npm run render:fixture
npm run render:armtune-ux
npm run dev
```

Open the URL printed by Slidev. The default slide canvas is 16:9.

## Validate

```bash
npm run build
npm run qa
npm run export
```

The source deck is `slides.md`; shared presentation styling is in `style.css`.
Asset provenance and claim boundaries are tracked in `ASSETS_NEEDED.md`.

## Claim Boundary

The headline serving result is limited to the preserved Phase 21D2B protocol:

```text
AWS C7g Arm64 server
AWS T4g Arm64 remote client
Qwen2.5-7B Q4_K_M
pinned llama-server binary
concurrency 4
direct private network path
```

Slides 6 through 8 add a same-identity, same-protocol C7g size-scaling
validation:

- `llama-bench` pp512/tg128 scaling on xlarge, 2xlarge, and 4xlarge
- one client CPU reserved on every size
- loopback serving at concurrency 4
- workload-scoped four-slot continuous-batching eligibility
- explicit rejection of prefill-heavy regressions

Hardware-scaling values, loopback tuning percentages, and the Phase 21D2B
remote-client percentages remain separate and are not added or multiplied.

Compatibility CI is shown separately and is not performance evidence.
