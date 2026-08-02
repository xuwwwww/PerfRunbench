---
theme: default
title: PerfRunbench ArmTune - Guarded Arm AI Inference Autotuning
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
    <div class="eyebrow light">ARM AI OPTIMIZATION CHALLENGE | CLOUD AI</div>
    <h1>ArmTune</h1>
    <p class="hero-subtitle">Guarded Arm AI inference autotuning</p>
    <p class="hero-definition">PerfRunbench target inspection, bounded search, statistical confirmation, and auditable recommendation reuse.</p>
  </div>
  <div class="hero-image">
    <img src="/assets/server-cpu-hero.png" alt="Unbranded editorial server CPU and motherboard visual" />
    <div class="hero-image-label">REAL ARM64 DATA | GENERIC HARDWARE VISUAL</div>
  </div>
</div>

<div class="hero-metrics">
  <div class="hero-metric"><strong>3 sizes</strong><span>C7g measured</span></div>
  <div class="hero-metric"><strong>1,047</strong><span>HTTP measurements</span></div>
  <div class="hero-metric reliability"><strong>514 / 0</strong><span>artifact checks / failures</span></div>
</div>

<div class="hero-scope">C7g.xlarge to C7g.4xlarge | Qwen2.5-7B Q4_K_M | pinned llama.cpp runtime | guarded concurrency 4</div>

<!--
[Sources]
- Same-protocol C7g size-scaling evidence: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G_SIZE_SCALING_VALIDATION.md
- Hero image: original unbranded image generated for this deck; no third-party logo or product image.
-->

---
layout: default
class: slide-problem
---

<div class="eyebrow">01 | THE ACTUAL PROBLEM</div>
<h2>The fastest run is not automatically the safest recommendation.</h2>

<div class="problem-layout">
  <div class="problem-thesis">
    <p>The best setting depends on the target, workload, runtime, and measurement protocol.</p>
    <div class="thesis-lines">
      <span>The fastest point may be noise.</span>
      <span>The environment may drift.</span>
      <span>One workload may improve while another regresses.</span>
    </div>
  </div>
  <div class="evidence-stack">
    <div class="evidence-excerpt">
      <div class="excerpt-head"><span>RUNTIME SEARCH</span><span>5% noise guard</span></div>
      <div class="excerpt-value">candidate <i>observed</i></div>
      <div class="excerpt-value selected">paired confirmation <i>eligible</i></div>
      <div class="excerpt-note">Fastest observed point is not a global optimum.</div>
    </div>
    <div class="evidence-excerpt">
      <div class="excerpt-head"><span>WORKLOAD GUARD</span><span>same profile</span></div>
      <div class="drift-line"><b>interactive + decode</b><span>eligible</span></div>
      <div class="drift-arrow">different traffic shape</div>
      <div class="drift-line refused"><b>prefill-heavy</b><span>rejected</span></div>
    </div>
    <div class="evidence-excerpt">
      <div class="excerpt-head"><span>ENVIRONMENT VALIDATION</span><span>hard drift</span></div>
      <div class="mono-line">C7g recommendation on T4g: refused</div>
      <div class="mono-line">failed runs remain failures</div>
      <div class="excerpt-note">No recommendation is manufactured.</div>
    </div>
  </div>
</div>

<!--
[Sources]
- Guard and negative-evidence summary: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/VALIDATED_PERFORMANCE_SUMMARY.md
- Same-protocol workload guard: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G_SIZE_SCALING_VALIDATION.md
-->

---
layout: default
class: slide-interaction
---

<div class="eyebrow">02 | GUIDED ARM EXPERIENCE</div>
<h2>Start with the workflow, not a wall of research configuration.</h2>

<div class="interaction-grid">
  <div class="command-column">
    <div class="command-label">ARMTUNE STARTER</div>
    <div class="command-block"><span class="shell-prompt">$</span><span>autotuneai arm init --preset quick</span></div>
    <div class="command-block"><span class="shell-prompt">$</span><span>autotuneai arm guide</span></div>
    <div class="command-block"><span class="shell-prompt">$</span><span>autotuneai arm explain decode-heavy</span></div>
    <div class="preset-line">
      <span><b>quick</b> 4 candidates</span>
      <span><b>standard</b> 8 candidates</span>
      <span><b>deep</b> 12 candidates</span>
    </div>
    <div class="interaction-points">
      <div><strong>01</strong><span>Versioned private starter</span></div>
      <div><strong>02</strong><span>Readiness and exact next steps</span></div>
      <div><strong>03</strong><span>Plain-language workload help</span></div>
    </div>
  </div>
  <div class="cli-capture">
    <img src="/assets/armtune-guided-cli.png" alt="Actual ArmTune init, guide, and explain CLI output" />
    <div class="cli-capture-label">
      <strong>ACTUAL LOCAL GUIDED UX</strong>
      <span>configuration evidence only | no target access</span>
    </div>
  </div>
</div>

<div class="scope-note">Generated configs are strict-SSH by default, preserve raw output, cap search cost, and remain result_origin: unverified until the private target is configured.</div>

<!--
[Sources]
- Guided ArmTune UX: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/armtune/USER_GUIDE.md
- ArmTune glossary: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/armtune/GLOSSARY.md
-->

---
layout: default
class: slide-pipeline
---

<div class="eyebrow">03 | GUARDED PIPELINE</div>
<h2>Every recommendation carries the evidence needed to reuse it safely.</h2>

<div class="pipeline">
  <div class="pipeline-rail"></div>
  <div class="pipeline-stage"><span class="stage-index">01</span><strong>Fingerprint</strong><p>Command, target, runtime</p></div>
  <div class="pipeline-stage"><span class="stage-index">02</span><strong>Search</strong><p>Deterministic plan, hard budget</p></div>
  <div class="pipeline-stage"><span class="stage-index">03</span><strong>Repeat</strong><p>Warm-up exclusion, paired runs</p></div>
  <div class="pipeline-stage"><span class="stage-index">04</span><strong>Confirm</strong><p>Noise band, failures, samples</p></div>
  <div class="pipeline-stage"><span class="stage-index">05</span><strong>Passport</strong><p>Digests, provenance, statistics</p></div>
  <div class="pipeline-stage selected-stage"><span class="stage-index">06</span><strong>Guarded reuse</strong><p>Validate, canary, lease, rollback</p></div>
</div>

<div class="pipeline-bottom">
  <div class="pipeline-quote">Measure once. Reuse only when identity still matches.</div>
  <div class="pipeline-rejections"><span>parser failure</span><span>inside noise band</span><span>hard drift</span><span>cleanup failure</span></div>
</div>

<!--
[Sources]
- PerfRunbench architecture and recommendation safety: https://github.com/xuwwwww/PerfRunbench/blob/main/README.md
- Performance Passport specification: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/armtune/PERFORMANCE_PASSPORT_SPEC.md
-->

---
layout: default
class: slide-concepts
---

<div class="eyebrow">04 | HOW TO READ THE WORKLOAD</div>
<h2>Prompt processing and token generation stress different bottlenecks.</h2>

<div class="inference-flow">
  <div class="flow-stage prefill-stage"><span>INPUT</span><strong>Prefill</strong><p>Read the prompt before the first output token</p><b>prefill tok/s + TTFT</b></div>
  <div class="flow-arrow"><span>first token</span><b>→</b></div>
  <div class="flow-stage decode-stage"><span>OUTPUT</span><strong>Decode</strong><p>Generate response tokens one at a time</p><b>output tok/s + E2E</b></div>
</div>

<div class="concept-cards">
  <div><span>INTERACTIVE-SHORT</span><strong>short prompt → short answer</strong><p>Responsiveness and p95 TTFT dominate.</p></div>
  <div><span>DECODE-HEAVY</span><strong>short prompt → long answer</strong><p>Sustained generation throughput dominates.</p></div>
  <div><span>PREFILL-HEAVY</span><strong>long prompt → short answer</strong><p>Prompt ingestion, memory traffic, and queueing dominate.</p></div>
</div>

<div class="scope-note">A serving profile can pass one traffic shape and fail another. ArmTune evaluates eligibility per workload instead of creating one universal score.</div>

<!--
[Sources]
- Plain-language definitions: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/armtune/GLOSSARY.md
- Workload protocol definitions: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/SERVING_AUTOTUNING.md
-->

---
layout: default
class: slide-scale
---

<div class="eyebrow">05 | SAME-PROTOCOL HARDWARE SCALING</div>
<h2>One pinned workload scales from 3 to 15 Arm server threads.</h2>

<div class="scale-layout">
  <div class="scale-chart">
    <img src="/assets/c7g-size-thread-scaling.svg" alt="Measured llama-bench prefill and decode throughput across three C7g sizes" />
  </div>
  <div class="scale-facts">
    <div class="scale-fact"><span>C7G.XLARGE | 3 THREADS</span><strong>16.444 / 7.101</strong><b>prefill / decode tok/s</b></div>
    <div class="scale-fact"><span>C7G.2XLARGE | 7 THREADS</span><strong>38.993 / 15.541</strong><b>prefill / decode tok/s</b></div>
    <div class="scale-fact"><span>C7G.4XLARGE | 15 THREADS</span><strong>73.598 / 29.871</strong><b>prefill / decode tok/s</b></div>
  </div>
</div>

<div class="scope-note">pp512 and tg128 are separate llama-bench tests | one warm-up plus five measured repetitions per point | one client CPU reserved | absolute scaling is not added to runtime-tuning percentages.</div>

<!--
[Sources]
- Exact cross-size protocol and results: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G_SIZE_SCALING_VALIDATION.md
-->

---
layout: default
class: slide-size-serving
---

<div class="eyebrow">06 | BEFORE / AFTER ACROSS THREE SIZES</div>
<h2>Four-slot continuous batching improved interactive and decode serving.</h2>

<div class="size-serving-layout">
  <div class="size-serving-chart">
    <img src="/assets/c7g-size-serving-before-after.svg" alt="Baseline and tuned serving throughput across three C7g sizes" />
  </div>
  <div class="size-serving-facts">
    <div><span>XLARGE DECODE</span><strong>6.237 to 12.136</strong><b>+94.6%</b></div>
    <div><span>2XLARGE DECODE</span><strong>13.626 to 25.876</strong><b>+89.9%</b></div>
    <div><span>4XLARGE DECODE</span><strong>25.475 to 46.186</strong><b>+81.3%</b></div>
  </div>
</div>

<div class="scope-note">Formal paired confirmation | three randomized pairs | fresh server restart between blocks | 20 measured requests per workload and side | 10,000-sample bootstrap intervals exclude zero.</div>

<!--
[Sources]
- Same-protocol paired confirmation: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G_SIZE_SCALING_VALIDATION.md
-->

---
layout: default
class: slide-workload-guard
---

<div class="eyebrow">07 | WORKLOAD-AWARE GUARD</div>
<h2>The tuner refuses a profile when the traffic shape changes.</h2>

<div class="workload-guard-layout">
  <div class="workload-guard-chart">
    <img src="/assets/c7g-size-workload-guard.svg" alt="Workload-aware screening matrix across three C7g sizes" />
  </div>
  <div class="workload-guard-facts">
    <div class="guard-result pass"><strong>6 / 6</strong><span>formal interactive + decode comparisons eligible</span></div>
    <div class="guard-result reject"><strong>3 / 3</strong><span>four-slot prefill-heavy slices rejected</span></div>
    <div class="context-guard"><span>WHY REJECTED</span><strong>-35.3% to -39.0%</strong><p>Throughput regressed and p95 TTFT worsened. No global recommendation was created.</p></div>
  </div>
</div>

<div class="scope-note">The selected family is eligible only for interactive-short and decode-heavy under this model, target size, concurrency, and protocol.</div>

<!--
[Sources]
- Screening guard matrix and rejection reasons: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G_SIZE_SCALING_VALIDATION.md
-->

---
layout: default
class: slide-throughput
---

<div class="eyebrow">08 | REAL REMOTE SERVING</div>
<div class="chart-title-row">
  <h2>69.4% to 86.5% more remote output throughput.</h2>
  <div class="chart-legend"><span class="legend-base"></span> baseline <span class="legend-selected"></span> selected</div>
</div>

<div class="dumbbell-chart">
  <div class="axis"><span>0</span><span>10</span><span>20</span><span>30</span><span>35 tok/s</span></div>
  <div class="dumbbell-row" style="--base:48.66%; --selected:89.59%;">
    <div class="row-label"><strong>Interactive-short</strong><span>+84.1%</span></div>
    <div class="dumbbell-plot"><div class="gridlines"></div><div class="dumbbell-link"></div><div class="plot-point base-point"><span>17.032</span></div><div class="plot-point selected-point"><span>31.355</span></div></div>
  </div>
  <div class="dumbbell-row" style="--base:48.70%; --selected:90.83%;">
    <div class="row-label"><strong>Decode-heavy</strong><span>+86.5%</span></div>
    <div class="dumbbell-plot"><div class="gridlines"></div><div class="dumbbell-link"></div><div class="plot-point base-point"><span>17.046</span></div><div class="plot-point selected-point"><span>31.790</span></div></div>
  </div>
  <div class="dumbbell-row" style="--base:47.32%; --selected:80.15%;">
    <div class="row-label"><strong>Prefill-heavy</strong><span>+69.4%</span></div>
    <div class="dumbbell-plot"><div class="gridlines"></div><div class="dumbbell-link"></div><div class="plot-point base-point"><span>16.563</span></div><div class="plot-point selected-point"><span>28.054</span></div></div>
  </div>
</div>

<div class="chart-scope">
  <strong>Validated remote topology</strong>
  <span>4 parallel slots | continuous batching | C7g server | T4g client</span>
  <span>direct private path | concurrency 4 | 608 measured requests | 0 failures</span>
</div>

<!--
[Sources]
- Exact remote serving results and protocol: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/PHASE_21D2B_C7G_T4G_REMOTE_SERVING_VALIDATION.md
-->

---
layout: default
class: slide-no
---

<div class="eyebrow">09 | STATISTICAL GUARD</div>
<h2>A trustworthy tuner must be able to say no.</h2>

<div class="no-layout">
  <div class="no-statement">
    <div class="not-equal">Measured <span>is not</span> eligible</div>
    <p>PerfRunbench recommends only candidates that clear evidence, provenance, correctness, and cleanup gates.</p>
    <div class="no-verdict">NO ELIGIBLE RECOMMENDATION</div>
  </div>
  <div class="rejected-list">
    <div class="rejected-row"><strong>Prefill-heavy serving</strong><span>regressed across all three C7g sizes</span></div>
    <div class="rejected-row"><strong>Later compiler / backend rerun</strong><span>inside 5% noise band</span></div>
    <div class="rejected-row"><strong>no-mmap</strong><span>+3.014% | below guard</span></div>
    <div class="rejected-row"><strong>Network tuning</strong><span>not executed | no bottleneck evidence</span></div>
  </div>
</div>

<div class="guard-foot">Negative evidence is preserved. Unsupported, regressed, or inconclusive candidates remain visible instead of becoming a speedup claim.</div>

<!--
[Sources]
- Validated positive and negative evidence: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/VALIDATED_PERFORMANCE_SUMMARY.md
- Cross-size prefill rejection: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G_SIZE_SCALING_VALIDATION.md
-->

---
layout: default
class: slide-platforms
---

<div class="eyebrow">10 | EVIDENCE, NOT JUST NUMBERS</div>
<h2>Performance results remain bound to identity, protocol, and cleanup.</h2>

<div class="platform-split">
  <div class="platform-half measured-half">
    <div class="platform-kicker">MEASURED ON ARM64</div>
    <div class="platform-main">3 C7g sizes</div>
    <div class="platform-sub">Arm Neoverse-V1 / Graviton3</div>
    <ul><li>same model and binary digests</li><li>1,047 measured HTTP requests</li><li>30 fresh server starts</li></ul>
  </div>
  <div class="platform-half compatibility-half">
    <div class="platform-kicker">INDEPENDENT VERIFICATION</div>
    <div class="compat-list">
      <div><strong>Artifacts</strong><span>514 checked</span><b>PASS</b></div>
      <div><strong>HTTP</strong><span>0 failures</span><b>PASS</b></div>
      <div><strong>Cleanup</strong><span>all targets</span><b>PASS</b></div>
    </div>
    <p>Performance Passport status is eligible preliminary. The result is not a global optimum.</p>
  </div>
</div>

<div class="ci-strip">
  <img src="/assets/github-actions-release-validation.png" alt="GitHub Actions release validation jobs" />
  <div><strong>Compatibility stays separate from performance</strong><span>Windows x86-64, Ubuntu x86-64, and Ubuntu Arm64 installed-wheel validation</span></div>
</div>

<!--
[Sources]
- Cross-size artifact validation and cleanup: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/C7G_SIZE_SCALING_VALIDATION.md
- Cross-platform validation matrix: https://github.com/xuwwwww/PerfRunbench/blob/main/docs/results/VALIDATION_MATRIX.md
-->

---
layout: default
class: slide-close
---

<div class="close-grid">
  <div class="close-copy">
    <div class="eyebrow light">PERFRUNBENCH | v0.3.1</div>
    <h2>Measure once.<br/>Reject false gains.<br/><span>Reuse safely.</span></h2>
    <p>Best observed eligible configuration under the tested target, model, workload, runtime, concurrency, topology, and measurement protocol.</p>
    <div class="close-links">
      <div><span>REPOSITORY</span><b>github.com/xuwwwww/PerfRunbench</b></div>
      <div><span>RELEASE</span><b>github.com/xuwwwww/PerfRunbench/releases/tag/v0.3.1</b></div>
    </div>
    <div class="author-line">@xuwwwww | Arm AI Optimization Challenge | Cloud AI</div>
  </div>
  <div class="close-assets">
    <div class="release-shot">
      <img src="/assets/github-release-v0.3.1.png" alt="PerfRunbench v0.3.1 GitHub Release" />
      <span>Published GitHub Release | verified wheel and sdist digests</span>
    </div>
    <div class="qr-row">
      <div class="qr-block repository-qr">
        <img src="/assets/repository-qr.svg" alt="QR code for PerfRunbench repository" />
        <span>OPEN SOURCE | MIT LICENSE</span>
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
