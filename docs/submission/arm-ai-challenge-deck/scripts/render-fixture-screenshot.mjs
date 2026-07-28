import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-chromium";

const evidenceRoot = path.resolve("..", "evidence");
const outputPath = path.resolve("public", "assets", "cli-run-auto-fixture.png");
const raw = JSON.parse(
  await fs.readFile(path.join(evidenceRoot, "public_fixture_run_auto.json"), "utf8")
);
const summary = JSON.parse(
  await fs.readFile(
    path.join(evidenceRoot, "public_fixture_run_auto_summary.json"),
    "utf8"
  )
);

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

const terminalOutput = JSON.stringify(raw, null, 2);
const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; width: 100%; height: 100%; background: #eef0eb; }
  body {
    display: grid;
    place-items: center;
    color: #17261d;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main {
    width: 1600px;
    height: 900px;
    padding: 70px 78px;
    background:
      linear-gradient(90deg, rgba(47, 91, 64, 0.06) 1px, transparent 1px),
      linear-gradient(rgba(47, 91, 64, 0.06) 1px, transparent 1px),
      #eef0eb;
    background-size: 40px 40px;
  }
  header {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-bottom: 28px;
  }
  .kicker {
    color: #ba3b2f;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 2px;
  }
  h1 { margin: 8px 0 0; font-size: 46px; line-height: 1; letter-spacing: 0; }
  .fixture-badge {
    padding: 12px 16px;
    border: 2px solid #ba3b2f;
    color: #87291f;
    background: #fff7f4;
    font-size: 15px;
    font-weight: 850;
    letter-spacing: 0.5px;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1.65fr) minmax(360px, 0.75fr);
    gap: 24px;
  }
  .terminal {
    min-height: 620px;
    overflow: hidden;
    border: 1px solid #1e3025;
    border-radius: 6px;
    background: #101913;
    box-shadow: 0 18px 50px rgba(20, 35, 26, 0.18);
  }
  .terminal-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 44px;
    padding: 0 16px;
    border-bottom: 1px solid #2c4133;
    background: #17261d;
  }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #7f9887; }
  .dot:first-child { background: #ba3b2f; }
  .terminal-title { margin-left: 8px; color: #9eb1a4; font-size: 13px; }
  .command {
    margin: 0;
    padding: 28px 30px 16px;
    color: #f0f4f1;
    font: 700 21px/1.45 "Cascadia Mono", Consolas, monospace;
    white-space: pre-wrap;
  }
  .prompt { color: #d25b4e; }
  pre {
    margin: 0;
    padding: 0 30px 30px;
    color: #b8c8bd;
    font: 17px/1.55 "Cascadia Mono", Consolas, monospace;
    white-space: pre-wrap;
  }
  .side { display: grid; grid-template-rows: repeat(4, 1fr); gap: 12px; }
  .fact {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 18px 20px;
    border-left: 4px solid #2f5b40;
    background: rgba(255, 255, 255, 0.78);
  }
  .fact.alert { border-left-color: #ba3b2f; }
  .fact span {
    color: #66746a;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .fact strong { margin-top: 8px; font-size: 25px; line-height: 1.05; }
  .fact small { margin-top: 7px; color: #5f6d63; font-size: 13px; line-height: 1.35; }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="kicker">ACTUAL CONFIG-FREE EXECUTION</div>
      <h1>PerfRunbench kept an inconclusive run honest.</h1>
    </div>
    <div class="fixture-badge">PUBLIC FIXTURE DEMO · NOT PERFORMANCE EVIDENCE</div>
  </header>
  <section class="layout">
    <div class="terminal">
      <div class="terminal-bar">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
        <span class="terminal-title">isolated Python 3.11 environment</span>
      </div>
      <div class="command"><span class="prompt">$</span> ${escapeHtml(summary.command)}</div>
      <pre>${escapeHtml(terminalOutput)}</pre>
    </div>
    <aside class="side">
      <div class="fact">
        <span>Bounded plan</span>
        <strong>quick preset</strong>
        <small>5 minute hard cap · 3 unique search candidates</small>
      </div>
      <div class="fact alert">
        <span>Decision</span>
        <strong>inconclusive</strong>
        <small>within the 2% noise band · no recommendation cached</small>
      </div>
      <div class="fact">
        <span>Execution</span>
        <strong>0 failed trials</strong>
        <small>baseline still executed · CLI exit code 0</small>
      </div>
      <div class="fact">
        <span>Cleanup audit</span>
        <strong>passed</strong>
        <small>0 residual fixture processes · 0 active lease/lock files</small>
      </div>
    </aside>
  </section>
</main>
</body>
</html>`;

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({
    viewport: { width: 1600, height: 900 },
    deviceScaleFactor: 1,
  });
  await page.setContent(html, { waitUntil: "load" });
  await page.screenshot({ path: outputPath, fullPage: false });
} finally {
  await browser.close();
}

console.log(
  JSON.stringify(
    {
      output: outputPath,
      evidence_class: summary.evidence_class,
      performance_evidence: summary.performance_evidence,
      auto_status: raw.auto_status,
    },
    null,
    2
  )
);
