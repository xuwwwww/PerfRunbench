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

Compatibility CI is shown separately and is not performance evidence.
