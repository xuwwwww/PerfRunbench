# C7g.4xlarge Visual Evidence Demo

This recording flow presents the saved C7g.4xlarge measured evidence without
starting AWS or exposing private target details. It must not be presented as a
live cloud benchmark.

## Recording Boundary

- live content: local CLI help and the checked-in HTML report;
- saved measured evidence: C7g.4xlarge charts, tables, guards, and Passport;
- topology: single-host loopback with one client CPU and fifteen server CPUs;
- model/runtime: one pinned Qwen2.5-7B Q4_K_M artifact and llama.cpp build;
- claim boundary: hardware scaling and runtime tuning are separate protocols;
- AWS console, target addresses, SSH material, model paths, and private
  artifacts must remain off screen.

## Start The Local Evidence Viewer

From the repository root:

```powershell
py -3.11 -m http.server 8765
```

Open:

```text
http://localhost:8765/docs/results/c7g4xlarge-realistic/arm_inference_report.html
```

No model, AWS resource, or benchmark process is required.

## Suggested 75-Second Segment

### 00:00-00:12 - Evidence Boundary

Show the top of the report.

Narration:

> This is preserved measured evidence from one C7g.4xlarge, one pinned model,
> one pinned llama.cpp build, and a declared loopback protocol. The report
> keeps hardware scaling separate from runtime tuning.

Show:

- `result_origin: measured`;
- `769` measured requests;
- four candidates;
- two rejected slices;
- `eligible preliminary` Passport status.

### 00:12-00:28 - Thread Scaling

Scroll to **llama-bench thread scaling**.

Narration:

> Under the same pp512 and tg128 protocol, throughput scaled by about 10.9
> times from one to fifteen threads. Efficiency tapered to roughly 73 percent,
> which is consistent with growing synchronization, cache, and memory-traffic
> pressure, but the run did not collect memory-controller counters, so that
> mechanism remains an inference.

Do not call this a universal Arm speedup.

### 00:28-00:48 - Serving Before And After

Scroll to **Serving before and after**.

Narration:

> At full fifteen-CPU server capacity and concurrency eight, four slots with
> continuous batching improved decode-heavy throughput by 80.331 percent and
> interactive-short throughput by 70.749 percent. Fresh server restarts,
> randomized paired blocks, and bootstrap confirmation were used.

Keep the 7-CPU capacity-matched row and 15-CPU full-capacity rows visibly
separate.

### 00:48-01:05 - Guarded Rejection

Scroll to **Screening guards**.

Narration:

> The recommendation is workload-aware. Prefill-heavy throughput regressed by
> 21.025 percent, and the eight-slot long-prefill slice exceeded the per-slot
> context and returned HTTP 400. PerfRunbench preserved those failures and
> excluded prefill from the recommendation.

Show the rejected-slice count and context-limited status. Do not omit the
negative result.

### 01:05-01:15 - Verification

Show the artifact validation and claim boundary sections.

Narration:

> The final archive was downloaded and all 278 manifest entries were
> independently revalidated. Cleanup and port release passed. The result is
> scoped to this target, model, build, workload, and protocol.

## Optional CLI Regeneration Shot

Show the public command with placeholders only:

```powershell
autotuneai report-arm-inference `
  --input "<measured-summary.json>" `
  --output-dir "<report-directory>"
```

Do not type a private source path during recording.

## Final Recording Checks

- browser address bar shows only `localhost`;
- no terminal history exposes AWS, SSH, or model paths;
- the voice-over calls the result measured but protocol-scoped;
- the `10.9x` scaling and `70.749%` to `80.331%` runtime improvements are not
  added or multiplied;
- the `-21.025%` prefill result and HTTP 400 context failure remain visible;
- the video labels this as saved measured evidence, not a live AWS run.
