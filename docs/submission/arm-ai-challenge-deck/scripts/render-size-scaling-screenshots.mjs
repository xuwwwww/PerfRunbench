import { chromium } from "playwright-chromium";
import { mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const repoRoot = path.resolve(import.meta.dirname, "..", "..", "..", "..");
const resultDir = path.join(repoRoot, "docs", "results", "c7g-size-scaling");
const reportPath = path.join(resultDir, "c7g_size_scaling_report.html");
const outputDir = path.join(resultDir, "screenshots");
const installedChrome = process.env.PROGRAMFILES
  ? path.join(
      process.env.PROGRAMFILES,
      "Google",
      "Chrome",
      "Application",
      "chrome.exe",
    )
  : undefined;
const executablePath =
  process.env.CHROME_PATH ??
  (installedChrome && existsSync(installedChrome) ? installedChrome : undefined);

await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch({
  ...(executablePath ? { executablePath } : {}),
  headless: true,
});
try {
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
  });
  await page.goto(pathToFileURL(reportPath).href, { waitUntil: "load" });
  await page.screenshot({
    path: path.join(outputDir, "report-overview-1600x1000.png"),
  });

  await page.setViewportSize({ width: 1600, height: 900 });
  for (const name of [
    "thread_scaling_across_sizes",
    "serving_before_after_by_size",
    "absolute_throughput_by_size",
    "ttft_before_after_by_size",
    "workload_guard_matrix",
    "parallel_efficiency",
  ]) {
    const chart = path.join(resultDir, "charts", `${name}.svg`);
    await page.goto(pathToFileURL(chart).href, { waitUntil: "load" });
    await page.screenshot({
      path: path.join(outputDir, `${name}-1600x900.png`),
    });
  }
} finally {
  await browser.close();
}
