# PerfRunbench Arm AI Challenge Deck

This is a standalone Slidev project for the Arm AI Optimization Challenge,
Cloud AI category.

## Run

```bash
npm install
npm run render:fixture
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

Slides 7 and 8 add a separate measured C7g.4xlarge protocol:

- `llama-bench` pp512/tg128 scaling from 1 to 15 threads
- 15-server-CPU loopback serving at concurrency 8
- workload-scoped `slots4-cont-on` eligibility
- explicit rejection of regressed and context-limited workloads

The larger-target percentages are not added to the remote-client result and
are not presented as a direct cross-instance comparison.

Compatibility CI is shown separately and is not performance evidence.
