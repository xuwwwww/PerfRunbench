import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright-chromium";

const repoRoot = path.resolve("..", "..", "..");
const project = await fs.mkdtemp(path.join(os.tmpdir(), "armtune-guided-"));
const outputPath = path.resolve("public", "assets", "armtune-guided-cli.png");

const run = (args) => {
  const result = spawnSync("py", ["-3.11", "-m", "autotune.cli", ...args], {
    cwd: repoRoot,
    encoding: "utf8",
    shell: false,
  });
  if (result.status !== 0) {
    throw new Error(`command failed (${result.status}): ${result.stderr || result.stdout}`);
  }
  return result.stdout.trim().replaceAll(project, ".autotuneai/demo-armtune");
};

let initOutput;
let guideOutput;
let explainOutput;
try {
  initOutput = run(["arm", "init", "--output", project, "--preset", "quick"]);
  guideOutput = run(["arm", "guide", "--project", project]);
  explainOutput = run(["arm", "explain", "decode-heavy"]);
} finally {
  await fs.rm(project, { recursive: true, force: true });
}

const compactGuide = guideOutput
  .split(/\r?\n/)
  .filter((line) =>
    line.startsWith("readiness:") ||
    line.startsWith("placeholder_count:") ||
    /^  [1-5]\./.test(line) ||
    line.includes("production performance requires")
  )
  .join("\n");

const compactInit = initOutput
  .split(/\r?\n/)
  .filter((line) =>
    line.startsWith("ArmTune") ||
    line.startsWith("project_directory:") ||
    line.startsWith("preset:")
  )
  .join("\n");

const terminalOutput = `$ autotuneai arm init --preset quick\n${compactInit}\n\n$ autotuneai arm guide\n${compactGuide}\n\n$ autotuneai arm explain decode-heavy\n${explainOutput}`;

const escapeHtml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; width: 100%; height: 100%; background: #eef0eb; }
  body { display: grid; place-items: center; font-family: Inter, "Segoe UI", sans-serif; color: #17261d; }
  main { width: 1500px; height: 900px; padding: 58px 64px; }
  header { display: flex; align-items: flex-end; justify-content: space-between; margin-bottom: 24px; }
  .kicker { color: #ba3b2f; font-size: 17px; font-weight: 800; letter-spacing: 2px; }
  h1 { margin: 8px 0 0; font-size: 43px; letter-spacing: 0; }
  .badge { padding: 11px 15px; border: 2px solid #2f5b40; background: #fff; color: #2f5b40; font-size: 14px; font-weight: 800; }
  .layout { display: grid; grid-template-columns: minmax(0, 1.7fr) 360px; gap: 24px; }
  .terminal { height: 675px; overflow: hidden; border: 1px solid #203428; border-radius: 5px; background: #101913; box-shadow: 0 18px 45px rgba(20,35,26,.18); }
  .bar { height: 43px; padding: 13px 18px; border-bottom: 1px solid #2c4133; background: #17261d; color: #9eb1a4; font: 13px Consolas, monospace; }
  pre { margin: 0; padding: 22px 26px; color: #c5d3ca; white-space: pre-wrap; font: 15px/1.42 "Cascadia Mono", Consolas, monospace; }
  .side { display: grid; gap: 12px; grid-template-rows: repeat(4, 1fr); }
  .fact { display: flex; flex-direction: column; justify-content: center; padding: 17px 19px; border-left: 4px solid #2f5b40; background: rgba(255,255,255,.86); }
  .fact.alert { border-left-color: #ba3b2f; }
  .fact span { color: #66746a; font-size: 11px; font-weight: 800; letter-spacing: 1px; }
  .fact strong { margin-top: 7px; font-size: 22px; line-height: 1.08; }
  .fact small { margin-top: 6px; color: #5f6d63; font-size: 12px; line-height: 1.35; }
</style>
</head>
<body>
<main>
  <header>
    <div><div class="kicker">ACTUAL SANITIZED LOCAL CLI</div><h1>ArmTune begins with a bounded workflow.</h1></div>
    <div class="badge">NO TARGET ACCESS</div>
  </header>
  <section class="layout">
    <div class="terminal"><div class="bar">isolated local project | paths sanitized</div><pre>${escapeHtml(terminalOutput)}</pre></div>
    <aside class="side">
      <div class="fact"><span>STARTER</span><strong>4 versioned files</strong><small>private placeholders, no credentials</small></div>
      <div class="fact alert"><span>READINESS</span><strong>needs configuration</strong><small>target access remains blocked</small></div>
      <div class="fact"><span>QUICK PRESET</span><strong>4 candidates max</strong><small>1 warm-up, 3 confirmations, 5% noise band</small></div>
      <div class="fact"><span>EXPLAIN</span><strong>terms at the CLI</strong><small>workload shape, metric, and guard meaning</small></div>
    </aside>
  </section>
</main>
</body>
</html>`;

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
  await page.setContent(html, { waitUntil: "load" });
  await page.screenshot({ path: outputPath, fullPage: false });
} finally {
  await browser.close();
}

console.log(JSON.stringify({ output: outputPath, source: "actual_sanitized_local_cli", target_access: false }, null, 2));
