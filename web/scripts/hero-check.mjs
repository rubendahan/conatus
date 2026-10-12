// Behaviour of the home-page hero in headless Chrome (SwiftShader), on a compute node:
//   node scripts/hero-check.mjs <outDir>
// Three loads of the built site (CN_WEB_OUT): (1) "/" as a reader without a hardware GPU sees it
// (software renderer refused: the poster runs through the exact Stuart family), (2) "/?sim=soft"
// (software accepted: the solver must start, in a worker where available), (3) "/" with
// prefers-reduced-motion. For each: long tasks (> 50 ms) on the main thread and their sum of
// excess over 50 ms (the quantity Lighthouse calls total blocking time, here unthrottled), page
// errors, whether the live canvas took over, and a screenshot after 9 s. Writes hero.json.
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer";
import { serve } from "./serve.mjs";
const out = process.argv[2];
fs.mkdirSync(out, { recursive: true });
const port = 4600 + Math.floor(Math.random() * 300);
const server = await serve(path.resolve(process.env.CN_WEB_OUT), port);
const browser = await puppeteer.launch({ headless: true, args: ["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const cases = [["poster", "/", false], ["solver", "/?sim=soft", false], ["reduced", "/", true]];
const res = [];
for (const [name, route, reduced] of cases) {
  const page = await browser.newPage();
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1, isMobile: true, hasTouch: true });
  if (reduced) await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: "reduce" }]);
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.evaluateOnNewDocument(() => {
    window.__long = [];
    new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__long.push(Math.round(e.duration)); }).observe({ type: "longtask", buffered: true });
  });
  await page.goto(`http://127.0.0.1:${port}${route}`, { waitUntil: "load" });
  await new Promise((r) => setTimeout(r, 9000));
  const r = await page.evaluate(() => {
    const c = document.querySelector(".hero-sim canvas");
    return { long: window.__long, live: c ? getComputedStyle(c).opacity === "1" : null, controls: !!document.querySelector(".cn-fluid-ui"),
      // a canvas handed to a worker (transferControlToOffscreen) throws on getContext
      where: c ? (() => { try { c.getContext("webgl2"); return "main thread (or none)"; } catch { return "worker"; } })() : null };
  });
  await page.screenshot({ path: path.join(out, `hero-${name}.png`) });
  const blocking = r.long.reduce((s, d) => s + Math.max(0, d - 50), 0);
  res.push({ case: name, route, reduced, longTasks: r.long, blockingMs: blocking, live: r.live, controls: r.controls, solverThread: r.where, errors });
  await page.close();
}
console.table(res.map(({ errors, longTasks, ...x }) => ({ ...x, longTasks: longTasks.join(" "), errors: errors.length })));
for (const x of res) for (const e of x.errors) console.log(`${x.case}: ${e}`);
fs.writeFileSync(path.join(out, "hero.json"), JSON.stringify(res, null, 2));
await browser.close();
server.close();
