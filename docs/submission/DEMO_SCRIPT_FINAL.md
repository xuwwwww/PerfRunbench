# PerfRunbench Demo Script

Target duration: 2 minutes 30 seconds to 3 minutes 30 seconds.

Do not run AWS live during recording. Use local fixture execution, saved public
evidence documents, release assets, and generated submission visuals. Clearly
distinguish live fixture execution from previously measured real evidence.

## Timeline

### 00:00-00:15 - Problem Statement

Screen: a simple terminal or editor view showing a training command and a few
manual tuning notes.

Narration:

> Performance tuning is usually a manual loop: change a flag, run a benchmark,
> copy the number, and hope the machine did not drift. The best setting depends
> on the target, runtime, workload, and measurement protocol. PerfRunbench turns
> that loop into a guarded workflow.

Transition: open the repository README.

### 00:15-00:35 - Product Flow

Screen: repository README followed by
`docs/submission/assets/perfrunbench_architecture.png`.

Narration:

> PerfRunbench wraps an existing workload with one command. It checks identity,
> runs bounded tuning slices, confirms the best observed eligible
> configuration, caches the recommendation, and safely reuses it with
> validation, canary execution, process-scoped apply, and rollback.

Command shown, not necessarily typed:

```powershell
autotuneai init --adapter pytorch-training --output autotune.yaml
autotuneai doctor --config autotune.yaml -- python train.py
autotuneai run --auto --config autotune.yaml -- python train.py
```

### 00:35-00:55 - Doctor

Screen: local demo terminal.

Command:

```powershell
autotuneai doctor --config autotune.yaml -- python train.py
```

Narration:

> Doctor checks the declared workload, metric source, identity files, and
> runtime readiness without launching the workload or changing system state.

Show:

- config loaded;
- workload identity accepted;
- no tuning launched.

### 00:55-01:25 - First Auto Run

Screen: local fixture command from `docs/submission/DEMO_COMMANDS.md`.

Command:

```powershell
autotuneai run --auto --config autotune.yaml -- python train.py
```

Narration:

> On the first run, the cache misses. PerfRunbench runs a bounded candidate
> plan, rejects failed candidates, confirms the selected result, creates a
> RecommendationBundle, runs a canary, and then launches the full workload.

Show:

- cache miss;
- bounded tuning;
- selected candidate;
- confirmation passed;
- recommendation cached.

### 01:25-01:45 - Cache Reuse

Screen: same terminal, repeat command.

Command:

```powershell
autotuneai run --auto --config autotune.yaml -- python train.py
```

Narration:

> The second run validates the same workload and environment, gets an exact
> cache hit, avoids retuning, and reuses the verified configuration.

Show:

- cache hit;
- no retuning;
- canary when policy requires it;
- full workload launch.

### 01:45-02:05 - Environment Refusal

Screen: `docs/results/VALIDATION_MATRIX.md` or a prepared terminal clip showing
hard-drift refusal.

Narration:

> Recommendations are not portable guesses. A C7g Arm recommendation is refused
> on a different CPU class or Windows machine because the environment
> fingerprint no longer matches.

Show:

- hard drift;
- no workload launch through `run --best`;
- recommendation remains bound to the measured target.

### 02:05-02:30 - Results And Validation Matrix

Screen:

- `docs/submission/assets/c7g_llama_runtime_before_after.png`
- `docs/submission/assets/c7g_pytorch_before_after.png`
- `docs/submission/assets/multitarget_validation_summary.png`
- `docs/results/c7g4xlarge-realistic/arm_inference_report_overview.png`
- `docs/results/c7g4xlarge-realistic/charts/serving_before_after.svg`
- `docs/results/c7g4xlarge-realistic/charts/guard_summary.svg`
- `docs/results/t4g-training-visual/charts/paired_before_after.png`
- `docs/results/t4g-training-visual/charts/guard_summary.png`

Narration:

> The real evidence is protocol-specific. C7g llama.cpp runtime search measured
> 13.630 to 17.480 tokens per second under one inference protocol. Synthetic
> PyTorch validations exercise the guard and cache workflow; a separate clean
> T4g protocol measured 12 paired confirmations and preserved four stalled
> trials. On a separate larger C7g.4xlarge loopback protocol, fifteen-thread
> llama-bench scaling reached about 10.9 times the one-thread throughput. At
> concurrency eight, slots4-cont-on improved eligible interactive and decode
> throughput by 70.749 to 80.331 percent, while prefill regressed and was
> rejected. These separate protocols are not combined.

### 02:30-02:50 - Passport And Guarded Apply

Screen:

- `docs/armtune/results/PHASE_10_GUARDED_APPLY_VALIDATION.md`
- `docs/releases/v0.3.1.md`

Narration:

> The selected Arm inference result produced an eligible preliminary
> Performance Passport. Guarded apply validation passed with environment
> validation, dry run, canary, lease, rollback, artifact revalidation, and
> cleanup evidence.

### 02:50-03:10 - Closing

Screen: GitHub Release page and repository quick start.

Narration:

> PerfRunbench is a guarded autotuning workflow for users who need more than a
> benchmark number. It records what was tested, rejects incompatible reuse, and
> gives a safe path from measurement to recommendation.

End with:

- repository URL;
- release URL;
- note that optional external workloads use the generic command adapter.

## Fallback Recording Plan

If the live fixture demo is too slow or terminal output is hard to read:

1. Record short terminal clips for doctor, first `run --auto`, second
   `run --auto`, and `run --best`.
2. Use static screenshots for release, validation matrix, charts, and
   Performance Passport evidence.
3. Add captions that label each clip as `live fixture execution` or
   `previously measured real evidence`.
4. Do not include AWS console pages, IP addresses, PEM paths, private cache
   paths, or model file paths.

The larger-target saved-evidence segment can be recorded independently by
following `docs/demo/C7G4XLARGE_VISUAL_EVIDENCE_DEMO.md`.
