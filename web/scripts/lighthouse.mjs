// Lighthouse scores of the built site (mobile preset, simulated throttling), headless Chrome from
// puppeteer's cache in scratch. Reports go to the directory given (never the home).
//   node scripts/lighthouse.mjs <outDir> [routes…]
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer";
import lighthouse from "lighthouse";
import * as chromeLauncher from "chrome-launcher";
import { serve } from "./serve.mjs";
const [out, ...routes] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const port = 4800 + Math.floor(Math.random() * 200);
const server = await serve(path.resolve(process.env.CN_WEB_OUT), port);
const chrome = await chromeLauncher.launch({ chromePath: await puppeteer.executablePath(), chromeFlags: ["--headless=new", "--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const rows = [];
for (const r of routes.length ? routes : ["/"]) {
  const res = await lighthouse(`http://127.0.0.1:${port}${r}`, { port: chrome.port, output: "html", logLevel: "error", onlyCategories: ["performance", "accessibility", "best-practices", "seo"] });
  const c = res.lhr.categories;
  const s = (k) => Math.round(c[k].score * 100);
  const a = res.lhr.audits;
  rows.push({ route: r, performance: s("performance"), accessibility: s("accessibility"), bestPractices: s("best-practices"), seo: s("seo"),
    lcp: a["largest-contentful-paint"].displayValue, cls: a["cumulative-layout-shift"].displayValue, tbt: a["total-blocking-time"].displayValue });
  fs.writeFileSync(path.join(out, (r === "/" ? "home" : r.replace(/\//g, "_").replace(/^_|_$/g, "")) + ".html"), res.report);
  const failed = Object.values(a).filter((x) => x.score !== null && x.score < 0.9 && x.scoreDisplayMode === "binary").map((x) => x.id);
  if (process.env.LH_DEBUG) {
    console.log(r, "benchmarkIndex", res.lhr.environment.benchmarkIndex);
    for (const id of ["first-contentful-paint", "largest-contentful-paint-element", "bootup-time", "mainthread-work-breakdown", "render-blocking-resources", "total-byte-weight"])
      console.log(" ", id, a[id]?.displayValue ?? "", JSON.stringify((a[id]?.details?.items || []).slice(0, 6)).slice(0, 700));
  }
  if (failed.length) console.log(`${r}: failing audits: ${failed.join(", ")}`);
}
console.table(rows);
fs.writeFileSync(path.join(out, "scores.json"), JSON.stringify(rows, null, 2));
await chrome.kill();
server.close();
