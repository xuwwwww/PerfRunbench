import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { chromium } from "playwright-chromium";

const source = await fs.readFile(path.resolve("slides.md"), "utf8");
const slideCount = (source.match(/^---\s*$/gm) || []).length / 2;
if (!Number.isInteger(slideCount) || slideCount < 1) {
  throw new Error(`unable to determine slide count from slides.md: ${slideCount}`);
}

const localServer = process.env.SLIDEV_URL ? null : await startStaticServer(path.resolve("dist"));
const baseUrl = process.env.SLIDEV_URL || localServer.url;
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

    for (let slide = 1; slide <= slideCount; slide += 1) {
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
  if (localServer) {
    await new Promise((resolve, reject) =>
      localServer.server.close((error) => (error ? reject(error) : resolve()))
    );
  }
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

async function startStaticServer(root) {
  const server = http.createServer(async (request, response) => {
    try {
      const requestUrl = new URL(request.url || "/", "http://localhost");
      const relative = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, "");
      let candidate = path.resolve(root, relative || "index.html");
      if (!candidate.startsWith(`${root}${path.sep}`) && candidate !== path.join(root, "index.html")) {
        response.writeHead(403).end("forbidden");
        return;
      }
      try {
        const info = await fs.stat(candidate);
        if (info.isDirectory()) candidate = path.join(candidate, "index.html");
      } catch {
        candidate = path.join(root, "index.html");
      }
      const body = await fs.readFile(candidate);
      response.writeHead(200, { "content-type": contentType(candidate) });
      response.end(body);
    } catch (error) {
      response.writeHead(500).end(String(error));
    }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "localhost", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("static server did not expose a TCP port");
  return { server, url: `http://localhost:${address.port}` };
}

function contentType(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  return {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
  }[extension] || "application/octet-stream";
}
