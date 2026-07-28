# T4g Training Visual Demo

## Purpose

This recording path demonstrates how PerfRunbench discovers, validates,
visualizes, and safely reuses one measured training recommendation. The
workload is the public synthetic CPU PyTorch example. It is not a production
training performance claim.

Use the checked-in clean evidence for the final charts:

```text
docs/results/T4G_TRAINING_VISUAL_EVIDENCE.md
docs/results/t4g-training-visual/training_tuning_report.html
docs/results/t4g-training-visual/report_overview_1920x1080.png
```

## Recording Setup

- Record at 1920x1080.
- Use a terminal font of at least 18 px.
- Hide hostnames, IP addresses, key paths, account identifiers, and private
  artifact roots.
- Keep the prompt at the repository root.
- Label condensed measurement footage as `10-minute measured run, condensed`.
- Do not imply that the checked-in result was produced during a short fixture
  clip.

## Commands

Inspect the bounded public config:

```bash
autotuneai doctor \
  --config configs/experiments/t4g_training_visualization.yaml \
  -- .venv/bin/python \
  examples/pytorch_synthetic_training/train_synthetic.py \
  --seed 0
```

Start a real measured tuning run:

```bash
autotuneai tune \
  --config configs/experiments/t4g_training_visualization.yaml \
  -- .venv/bin/python \
  examples/pytorch_synthetic_training/train_synthetic.py \
  --seed 0
```

Generate the report from preserved artifacts:

```bash
autotuneai report-training \
  --input .autotuneai/runs/<run-id>/pytorch_training/tuning_result.json \
  --output-dir results/reports/t4g-training \
  --cloudwatch cloudwatch-normalized.json
```

Demonstrate no-retune reuse:

```bash
autotuneai run --auto \
  --config configs/experiments/t4g_training_visualization.yaml \
  -- .venv/bin/python \
  examples/pytorch_synthetic_training/train_synthetic.py \
  --seed 0

autotuneai run --best \
  --config configs/experiments/t4g_training_visualization.yaml \
  -- .venv/bin/python \
  examples/pytorch_synthetic_training/train_synthetic.py \
  --seed 0
```

The expected reuse evidence is `cache_hit`, `newly_tuned=false`, process-scoped
apply, canary where policy requires it, and rollback after execution.

## Three-Minute Storyboard

### 00:00-00:20 - Problem

Show the PerfRunbench primary flow.

Narration:

> A benchmark number is not enough. PerfRunbench binds a result to its target,
> workload, protocol, and evidence, then refuses unsafe reuse.

### 00:20-00:45 - Bounded Plan

Show the config and doctor output.

Call out:

- effective batch fixed at 32;
- five bounded candidates;
- 5,000 measured steps per trial;
- 12 paired confirmations;
- five-percent noise band.

### 00:45-01:10 - Measured Run

Show the start and final JSON summary, with the middle condensed.

Narration:

> The real T4g run completed in about ten minutes. One DataLoader candidate
> stalled and remained a failure. PerfRunbench did not convert it to zero.

### 01:10-01:45 - Before And After

Show:

```text
docs/results/t4g-training-visual/charts/paired_before_after.png
docs/results/t4g-training-visual/charts/throughput_distribution.png
```

Narration:

> Across 12 fresh pairs, the baseline median was 14,430 samples per second and
> the selected median was 33,275. The ratio-of-medians improvement was 130.6
> percent, and the median paired improvement was 128.2 percent.

### 01:45-02:15 - Guard Evidence

Show:

```text
docs/results/t4g-training-visual/charts/guard_summary.png
docs/results/t4g-training-visual/charts/cpu_credit_timeline.png
```

Narration:

> Eligibility required enough samples, consistent paired direction, an effect
> outside the noise band, bounded variability, cleanup, and measured
> provenance. The CPU-credit chart also discloses that this burstable target
> ran in Unlimited surplus state.

### 02:15-02:40 - Safe Reuse

Show the preserved summaries for two cache-hit `run --auto` calls and one
`run --best`.

Narration:

> Reuse validated the environment, did not retune, applied only to the process,
> and rolled back. A deliberately exhausted budget returned inconclusive and
> never entered the cache.

### 02:40-03:00 - Claim Boundary

Show the 1920x1080 report overview.

Narration:

> This is one T4g.small and one synthetic CPU PyTorch protocol. It is not a
> universal PyTorch speedup and it is not combined with the separate C7g
> llama.cpp evidence.

## Recording Safety

Before publishing, inspect every frame for:

- AWS addresses or resource IDs;
- SSH identity paths or fingerprints;
- local absolute paths;
- account identifiers;
- model paths or model files;
- private artifact locations; and
- unlabeled fixture or synthetic values.
