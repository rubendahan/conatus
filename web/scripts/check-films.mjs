// Check that the Gallery films play in a browser (headless Chrome via puppeteer), on the built site.
//   node scripts/check-films.mjs <outFile.json>      (CN_WEB_OUT: the built site; BASE as built)
// Checks: (1) with motion allowed, the first film plays (currentTime advances) once in view, a film far
// below the fold has no source attached and is paused, and the first film pauses when scrolled away;
// (2) the lightbox opens on the first card and its film plays; (3) under prefers-reduced-motion no film
// plays, each shows its poster and a Play button, and Play starts it. Exit 1 on any failure.
import fs from "node:fs";
import path from "node:path";
import puppeteer from "puppeteer";
import { serve } from "./serve.mjs";

const out = process.argv[2];
const base = (process.env.BASE || "/").replace(/\/?$/, "/");
const port = 4800 + Math.floor(Math.random() * 500);
const server = await serve(path.resolve(process.env.CN_WEB_OUT), port);
const browser = await puppeteer.launch({ headless: true, args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required"] });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const report = { checks: [] };
const ok = (name, pass, detail = {}) => { report.checks.push({ name, pass: !!pass, ...detail }); console.log(`${pass ? "PASS" : "FAIL"} ${name} ${JSON.stringify(detail)}`); };
const state = (page, sel) => page.$$eval(sel, (vs) => vs.map((v) => ({
  t: v.currentTime, paused: v.paused, sources: v.querySelectorAll("source").length, src: v.currentSrc.split("/").pop(),
  poster: v.poster.split("/").pop(), ready: v.readyState })));

async function open(reduced) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  await page.emulateMediaFeatures([{ name: "prefers-reduced-motion", value: reduced ? "reduce" : "no-preference" }]);
  await page.goto(`http://127.0.0.1:${port}${base}gallery/`, { waitUntil: "networkidle0" });
  await sleep(500);
  return page;
}

try {
  // (1) motion allowed
  let page = await open(false);
  await page.evaluate(() => document.querySelector(".gplate video")?.scrollIntoView({ block: "center" }));
  await sleep(4000);
  let s = await state(page, ".gplate video");
  ok("films in the grid", s.length > 0, { n: s.length });
  ok("first film plays once visible", s[0] && !s[0].paused && s[0].t > 0.5, s[0]);
  const last = s[s.length - 1];
  ok("a film far below the fold is not loaded", last && last.sources === 0 && last.paused, last);
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await sleep(1500);
  s = await state(page, ".gplate video");
  ok("the first film pauses off-screen", s[0].paused, s[0]);
  // (2) lightbox
  await page.evaluate(() => window.scrollTo(0, 0));
  await sleep(500);
  await page.click(".gplate .gplate-img");
  await sleep(4000);
  const lb = await state(page, "dialog.lb video");
  ok("lightbox film plays", lb[0] && !lb[0].paused && lb[0].t > 0.5, lb[0]);
  const info = await page.$eval("dialog.lb .lb-info", (e) => e.textContent);
  ok("lightbox names the time span, the playback and the measured values",
     /Simulated time/.test(info) && /Measured in this run/.test(info) && /SHA-256/.test(info), {});
  ok("lightbox shows the colour bar", !!(await page.$("dialog.lb .lb-ramp")), {});
  await page.close();
  // (3) reduced motion
  page = await open(true);
  await page.evaluate(() => document.querySelector(".gplate video")?.scrollIntoView({ block: "center" }));
  await sleep(2500);
  s = await state(page, ".gplate video");
  ok("reduced motion: no film plays", s.every((v) => v.paused && v.t === 0), { playing: s.filter((v) => !v.paused).length });
  ok("reduced motion: posters shown", s.every((v) => v.poster.endsWith(".webp")), {});
  const btn = await page.$eval(".gplate .film-btn", (b) => b.textContent.trim());
  ok("reduced motion: a Play button", btn === "Play", { btn });
  await page.click(".gplate .film-btn");
  await sleep(3000);
  s = await state(page, ".gplate video");
  ok("reduced motion: Play starts the film", !s[0].paused && s[0].t > 0.5, s[0]);
  await page.close();
} finally {
  await browser.close();
  server.close();
}
report.pass = report.checks.every((c) => c.pass);
fs.writeFileSync(out, JSON.stringify(report, null, 1));
console.log(report.pass ? "films: PASS" : "films: FAIL");
process.exit(report.pass ? 0 : 1);
