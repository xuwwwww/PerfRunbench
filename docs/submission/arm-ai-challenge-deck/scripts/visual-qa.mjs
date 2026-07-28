import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright-chromium";

const baseUrl = process.env.SLIDEV_URL || "http://localhost:3037";
const outputRoot = path.resolve(".qa");
const viewports = [
  { name: "1366x768", width: 1366, height: 768 },
  { name: "1920x1080", width: 1920, height: 1080 },
];

await fs.mkdir(outputRoot, { recursive: true });

const browser = await chromium.launch({ headless: true });
const report = [];

try {
  for (const viewport of viewports) {
    const context = await browser.newContext({
      viewport: { width: viewport.width, height: viewport.height },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    const viewportDir = path.join(outputRoot, viewport.name);
    await fs.mkdir(viewportDir, { recursive: true });

    for (let slide = 1; slide <= 11; slide += 1) {
      await page.goto(`${baseUrl}/${slide}`, { waitUntil: "networkidle" });
      await page.waitForTimeout(300);

      const inspection = await page.evaluate(() => {
        const layout = document.querySelector(".slidev-layout");
        if (!layout) {
          return { error: "slidev-layout not found" };
        }

        const bounds = layout.getBoundingClientRect();
        const overflow = [];
        const candidates = layout.querySelectorAll(
          "h1, h2, p, strong, span, img, pre, .scope-note, .hero-scope"
        );

        for (const element of candidates) {
          const rect = element.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0) continue;
          if (
            rect.left < bounds.left - 1 ||
            rect.right > bounds.right + 1 ||
            rect.top < bounds.top - 1 ||
            rect.bottom > bounds.bottom + 1
          ) {
            overflow.push({
              tag: element.tagName,
              className: element.className?.toString() || "",
              text: element.textContent?.trim().slice(0, 90) || "",
              rect: {
                left: Math.round(rect.left),
                right: Math.round(rect.right),
                top: Math.round(rect.top),
                bottom: Math.round(rect.bottom),
              },
              bounds: {
                left: Math.round(bounds.left),
                right: Math.round(bounds.right),
                top: Math.round(bounds.top),
                bottom: Math.round(bounds.bottom),
              },
            });
          }
        }

        return {
          overflow,
          bodyScrollWidth: document.body.scrollWidth,
          bodyScrollHeight: document.body.scrollHeight,
          viewportWidth: window.innerWidth,
          viewportHeight: window.innerHeight,
        };
      });

      await page.screenshot({
        path: path.join(viewportDir, `slide-${slide}.png`),
        fullPage: false,
      });

      report.push({ viewport: viewport.name, slide, ...inspection });
    }

    await context.close();
  }
} finally {
  await browser.close();
}

await fs.writeFile(
  path.join(outputRoot, "qa-report.json"),
  `${JSON.stringify(report, null, 2)}\n`,
  "utf8"
);

const failures = report.filter(
  (item) => item.error || (item.overflow && item.overflow.length > 0)
);

console.log(
  JSON.stringify(
    {
      slides_checked: report.length,
      overflow_failures: failures.length,
      report: path.join(outputRoot, "qa-report.json"),
    },
    null,
    2
  )
);

if (failures.length > 0) {
  process.exitCode = 1;
}
