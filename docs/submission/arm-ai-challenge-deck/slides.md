---
theme: default
title: PerfRunbench — Guarded Autotuning for Repeatable AI Workloads
info: |
  Arm AI Optimization Challenge
  Cloud AI category
class: deck-cover
transition: fade-out
drawings:
  persist: false
mdc: true
---

<div class="hero-grid">
  <div class="hero-copy">
    <div class="eyebrow light">ARM AI OPTIMIZATION CHALLENGE · CLOUD AI</div>
    <h1>PerfRunbench</h1>
    <p class="hero-subtitle">Guarded autotuning for repeatable AI workloads</p>
    <p class="hero-definition">Bounded search, statistical confirmation, and safe recommendation reuse.</p>
  </div>
  <div class="hero-image">
    <img src="/assets/server-cpu-hero.png" alt="Unbranded editorial server CPU and motherboard visual" />
    <div class="hero-image-label">REAL ARM64 DATA · GENERIC HARDWARE VISUAL</div>
  </div>
</div>

<div class="hero-metrics">
  <div class="hero-metric">
    <strong>+86.5%</strong>
    <span>output throughput</span>
  </div>
  <div class="hero-metric">
    <strong>−98.0%</strong>
    <span>p95 TTFT</span>
  </div>
  <div class="hero-metric reliability">
    <strong>608</strong>
    <span>measured requests · 0 failures</span>
  </div>
</div>

<div class="hero-scope">C7g Arm64 server · T4g Arm64 remote client · Qwen2.5-7B Q4_K_M · concurrency 4 · direct private path</div>

<!--
[Sources]
- PerfRunbench Phase 21D2B evidence: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md
- Hero image: original unbranded image generated for this deck; no third-party logo or product image.
-->

---
layout: default
class: slide-problem
---

<div class="eyebrow">01 · THE ACTUAL PROBLEM</div>
<h2>The fastest run is not automatically the safest recommendation.</h2>

<div class="problem-layout">
  <div class="problem-thesis">
    <p>The best setting depends on the target, workload, runtime, and measurement protocol.</p>
    <div class="thesis-lines">
      <span>The fastest run may be noise.</span>
      <span>The environment may drift.</span>
      <span>The apply step may be unsafe.</span>
    </div>
  </div>

  <div class="evidence-stack">
    <div class="evidence-excerpt">
      <div class="excerpt-head"><span>RUNTIME SEARCH</span><span>2% noise band</span></div>
      <div class="excerpt-value">17.495 <i>single point</i></div>
      <div class="excerpt-value selected">17.480 <i>staged eligible candidate</i></div>
      <div class="excerpt-note">Fastest observed point ≠ globally optimal recommendation</div>
    </div>
    <div class="evidence-excerpt">
      <div class="excerpt-head"><span>ENVIRONMENT VALIDATION</span><span>hard drift</span></div>
      <div class="drift-line"><b>C7g / Neoverse-V1</b><span>recommendation bound</span></div>
      <div class="drift-arrow">↓ different CPU class</div>
      <div class="drift-line refused"><b>T4g / Neoverse-N1</b><span>reuse refused</span></div>
    </div>
    <div class="evidence-excerpt">
      <div class="excerpt-head"><span>SYSTEM EXPERIMENTS</span><span>guarded</span></div>
      <div class="mono-line">no-mmap: +3.014% &lt; 5% guard</div>
      <div class="mono-line">network tuning: not executed</div>
      <div class="excerpt-note">No eligible recommendation was manufactured.</div>
    </div>
  </div>
</div>

<!--
[Sources]
- Runtime search evidence: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/armtune/results/PHASE_9_FIRST_REAL_ARM_TUNING.md
- Drift and system guard summary: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/VALIDATED_PERFORMANCE_SUMMARY.md
-->

---
layout: default
class: slide-interaction
---

<div class="eyebrow">02 · PRODUCT INTERACTION</div>
<h2>One command starts a bounded, process-scoped tuning run.</h2>

<div class="interaction-grid">
  <div class="command-column">
    <div class="command-label">CONFIG-FREE ENTRY</div>
    <div class="command-block"><span class="shell-prompt">$</span><span>autotuneai run --auto --preset quick -- python app.py</span></div>
    <div class="preset-line">
      <span><b>quick</b> 5 min / 3 candidates</span>
      <span><b>standard</b> 20 min / 8 candidates</span>
      <span><b>deep</b> 60 min / 16 candidates</span>
    </div>
    <div class="interaction-points">
      <div><strong>01</strong><span>No configuration file</span></div>
      <div><strong>02</strong><span>Hard execution budget</span></div>
      <div><strong>03</strong><span>No machine-wide state change</span></div>
    </div>
  </div>

  <div class="cli-capture">
    <img src="/assets/cli-run-auto-fixture.png" alt="Actual public fixture run-auto output showing an inconclusive result and cleanup" />
    <div class="cli-capture-label">
      <strong>ACTUAL PUBLIC FIXTURE RUN</strong>
      <span>UX and guard evidence only · not Arm performance</span>
    </div>
  </div>
</div>

<div class="scope-note">When the hard budget expires, PerfRunbench still writes a valid partial report and runs cleanup; invalid evidence is not cached.</div>

<!--
[Sources]
- Public CLI and preset semantics: https://github.com/xuwwwww/PerfRunbench/blob/main/README.md
- Release v0.3.1 notes: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/releases/v0.3.1.md
-->

---
layout: default
class: slide-pipeline
---

<div class="eyebrow">03 · GUARDED PIPELINE</div>
<h2>Every recommendation carries the evidence needed to reuse it safely.</h2>

<div class="pipeline">
  <div class="pipeline-rail"></div>
  <div class="pipeline-stage">
    <span class="stage-index">01</span>
    <strong>Fingerprint</strong>
    <p>Command, config, target, runtime</p>
  </div>
  <div class="pipeline-stage">
    <span class="stage-index">02</span>
    <strong>Search</strong>
    <p>Deterministic plan, hard budget</p>
  </div>
  <div class="pipeline-stage">
    <span class="stage-index">03</span>
    <strong>Repeat</strong>
    <p>Warm-up exclusion, paired runs</p>
  </div>
  <div class="pipeline-stage">
    <span class="stage-index">04</span>
    <strong>Confirm</strong>
    <p>Noise band, failures, sample guard</p>
  </div>
  <div class="pipeline-stage">
    <span class="stage-index">05</span>
    <strong>Passport</strong>
    <p>Digests, provenance, statistics</p>
  </div>
  <div class="pipeline-stage selected-stage">
    <span class="stage-index">06</span>
    <strong>Guarded reuse</strong>
    <p>Validate, canary, lease, rollback</p>
  </div>
</div>

<div class="pipeline-bottom">
  <div class="pipeline-quote">Measure once. Reuse only when identity still matches.</div>
  <div class="pipeline-rejections">
    <span>parser failure</span>
    <span>inside noise band</span>
    <span>hard drift</span>
    <span>cleanup failure</span>
  </div>
</div>

<!--
[Sources]
- PerfRunbench architecture and recommendation safety: https://github.com/xuwwwww/PerfRunbench/blob/main/README.md
- Performance Passport specification: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/armtune/PERFORMANCE_PASSPORT_SPEC.md
-->

---
layout: default
class: slide-throughput
---

<div class="eyebrow">04 · REMOTE SERVING THROUGHPUT</div>
<div class="chart-title-row">
  <h2>69.4–86.5% more output throughput.</h2>
  <div class="chart-legend"><span class="legend-base"></span> baseline <span class="legend-selected"></span> selected</div>
</div>

<div class="dumbbell-chart">
  <div class="axis">
    <span>0</span><span>10</span><span>20</span><span>30</span><span>35 tok/s</span>
  </div>
  <div class="dumbbell-row" style="--base:48.66%; --selected:89.59%;">
    <div class="row-label"><strong>Interactive-short</strong><span>+84.1%</span></div>
    <div class="dumbbell-plot">
      <div class="gridlines"></div>
      <div class="dumbbell-link"></div>
      <div class="plot-point base-point"><span>17.032</span></div>
      <div class="plot-point selected-point"><span>31.355</span></div>
    </div>
  </div>
  <div class="dumbbell-row" style="--base:48.70%; --selected:90.83%;">
    <div class="row-label"><strong>Decode-heavy</strong><span>+86.5%</span></div>
    <div class="dumbbell-plot">
      <div class="gridlines"></div>
      <div class="dumbbell-link"></div>
      <div class="plot-point base-point"><span>17.046</span></div>
      <div class="plot-point selected-point"><span>31.790</span></div>
    </div>
  </div>
  <div class="dumbbell-row" style="--base:47.32%; --selected:80.15%;">
    <div class="row-label"><strong>Prefill-heavy</strong><span>+69.4%</span></div>
    <div class="dumbbell-plot">
      <div class="gridlines"></div>
      <div class="dumbbell-link"></div>
      <div class="plot-point base-point"><span>16.563</span></div>
      <div class="plot-point selected-point"><span>28.054</span></div>
    </div>
  </div>
</div>

<div class="chart-scope">
  <strong>Selected runtime</strong>
  <span>4 parallel slots · continuous batching on · ctx 2048 · batch 512 · ubatch 128</span>
  <span>C7g server · T4g client · direct private path · concurrency 4</span>
</div>

<!--
[Sources]
- Exact remote serving results and protocol: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md
-->

---
layout: default
class: slide-latency
---

<div class="eyebrow">05 · LATENCY + RELIABILITY</div>
<h2>p95 time-to-first-token fell by 91.1–98.0%.</h2>

<div class="latency-layout">
  <div class="latency-chart">
    <div class="latency-row" style="--candidate:8.87%;">
      <div class="latency-label">Interactive-short</div>
      <div class="latency-bars">
        <div class="latency-bar baseline-bar"><span>3056 ms baseline</span></div>
        <div class="latency-bar candidate-bar"><span>271 ms selected</span></div>
      </div>
      <div class="reduction">−91.1%</div>
    </div>
    <div class="latency-row" style="--candidate:2.00%;">
      <div class="latency-label">Decode-heavy</div>
      <div class="latency-bars">
        <div class="latency-bar baseline-bar"><span>10127 ms baseline</span></div>
        <div class="latency-bar candidate-bar"><span>202 ms selected</span></div>
      </div>
      <div class="reduction">−98.0%</div>
    </div>
    <div class="latency-row" style="--candidate:8.61%;">
      <div class="latency-label">Prefill-heavy</div>
      <div class="latency-bars">
        <div class="latency-bar baseline-bar"><span>3967 ms baseline</span></div>
        <div class="latency-bar candidate-bar"><span>342 ms selected</span></div>
      </div>
      <div class="reduction">−91.4%</div>
    </div>
  </div>

  <div class="reliability-column">
    <div class="reliability-primary"><strong>608</strong><span>measured requests</span></div>
    <div class="reliability-row"><strong>0</strong><span>retries</span></div>
    <div class="reliability-row"><strong>0</strong><span>request failures</span></div>
    <div class="reliability-row"><strong>0</strong><span>quality failures</span></div>
    <div class="reliability-pass">Artifacts validated · both hosts restored</div>
  </div>
</div>

<div class="scope-note">Client-observed TTFT · fresh server restart between blocks · deterministic randomized block order · same concurrency and workload protocol per comparison.</div>

<!--
[Sources]
- Phase 21D2B protocol, reliability, artifact validation, and cleanup: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md
-->

---
layout: default
class: slide-no
---

<div class="eyebrow">06 · STATISTICAL GUARD</div>
<h2>A trustworthy tuner must be able to say no.</h2>

<div class="no-layout">
  <div class="no-statement">
    <div class="not-equal">Measured <span>≠</span> eligible</div>
    <p>PerfRunbench recommends only candidates that clear evidence, provenance, correctness, and cleanup gates.</p>
    <div class="no-verdict">NO ELIGIBLE RECOMMENDATION</div>
  </div>

  <div class="rejected-list">
    <div class="rejected-row">
      <strong>Later compiler / backend rerun</strong>
      <span>inside 5% noise band</span>
    </div>
    <div class="rejected-row">
      <strong>no-mmap</strong>
      <span>+3.014% · below guard</span>
    </div>
    <div class="rejected-row">
      <strong>4K / 64K kernel profiles</strong>
      <span>no qualified improvement</span>
    </div>
    <div class="rejected-row">
      <strong>Network tuning</strong>
      <span>not executed · no bottleneck evidence</span>
    </div>
  </div>
</div>

<div class="guard-foot">Negative evidence is preserved. Unsupported, regressed, or inconclusive candidates remain visible instead of becoming a speedup claim.</div>

<!--
[Sources]
- Validated negative evidence: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/VALIDATED_PERFORMANCE_SUMMARY.md
- Network tuning gate: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md
-->

---
layout: default
class: slide-platforms
---

<div class="eyebrow">07 · ARM RELEVANCE + PORTABILITY</div>
<h2>Performance evidence and compatibility validation are separate claims.</h2>

<div class="platform-split">
  <div class="platform-half measured-half">
    <div class="platform-kicker">MEASURED ON ARM64</div>
    <div class="platform-main">AWS C7g</div>
    <div class="platform-sub">Arm Neoverse-V1 server</div>
    <div class="path-line"><span>C7g server</span><i>direct private path</i><span>T4g client</span></div>
    <ul>
      <li>Qwen2.5-7B Q4_K_M</li>
      <li>pinned llama-server binary</li>
      <li>608 remote measured requests</li>
    </ul>
  </div>

  <div class="platform-half compatibility-half">
    <div class="platform-kicker">COMPATIBILITY VALIDATED</div>
    <div class="compat-list">
      <div><strong>Ubuntu</strong><span>Arm64</span><b>PASS</b></div>
      <div><strong>Ubuntu</strong><span>x86-64</span><b>PASS</b></div>
      <div><strong>Windows</strong><span>x86-64</span><b>PASS</b></div>
    </div>
    <p>Installed-wheel CI validates packaging and behavior. It is not a performance benchmark.</p>
  </div>
</div>

<div class="ci-strip">
  <img src="/assets/github-actions-release-validation.png" alt="GitHub Actions release validation jobs" />
  <div>
    <strong>Release validation · all three runners succeeded</strong>
    <span>GitHub Actions run 30234620475 · commit 5ccab60</span>
  </div>
</div>

<!--
[Sources]
- Cross-platform CI run: https://github.com/xuwwwww/PerfRunbench/actions/runs/30234620475
- Validation matrix: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/VALIDATION_MATRIX.md
- Arm remote performance evidence: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md
-->

---
layout: default
class: slide-close
---

<div class="close-grid">
  <div class="close-copy">
    <div class="eyebrow light">PERFRUNBENCH · v0.3.1</div>
    <h2>Measure once.<br/>Reject false gains.<br/><span>Reuse safely.</span></h2>
    <p>Best observed eligible configuration under the tested target, model, workload, runtime, concurrency, topology, and measurement protocol.</p>
    <div class="close-links">
      <div><span>REPOSITORY</span><b>github.com/xuwwwww/PerfRunbench</b></div>
      <div><span>RELEASE</span><b>github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1</b></div>
    </div>
    <div class="author-line">@xuwwwww · Arm AI Optimization Challenge · Cloud AI</div>
  </div>

  <div class="close-assets">
    <div class="release-shot">
      <img src="/assets/github-release-v0.3.1.png" alt="PerfRunbench v0.3.1 GitHub Release" />
      <span>Published GitHub Release · verified wheel + sdist digests</span>
    </div>
    <div class="qr-row">
      <div class="qr-block repository-qr">
        <img src="/assets/repository-qr.svg" alt="QR code for PerfRunbench repository" />
        <span>OPEN SOURCE · MIT LICENSE</span>
      </div>
    </div>
  </div>
</div>

<!--
[Sources]
- Repository: https://github.com/xuwwwww/PerfRunbench
- GitHub Release v0.3.1: https://github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1
- Release assets and digests verified through the GitHub Releases API on 2026-07-28.
-->
