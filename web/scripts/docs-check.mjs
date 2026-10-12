// Runtime checks of the built site and docs that a static scan cannot make:
//   node scripts/docs-check.mjs [route …]      CN_WEB_OUT = the built site (web/build.sh --with-docs)
// For each docs page (default: the docs index and the getting-started tutorial), in headless Chrome:
//  * every network request goes to the page's own origin (docs/site/check_origins.py reads the files;
//    this watches what MathJax, the fonts and the theme really fetch);
//  * every formula is typeset and carries its hidden MathML copy for screen readers
//    (mjx-assistive-mml with a <math> element: MathJax's speech pass is off, mathjax_config.js);
// and the old addresses (routes.ts redirects) land on their new page in the server build.
// Prints one line per check; exit 1 if any fails.
import path from "node:path";
import puppeteer from "puppeteer";
import { serve } from "./serve.mjs";

const routes = process.argv.slice(2);
const pages = routes.length ? routes : ["/docs/index.html", "/docs/tutorials/00_getting_started.html"];
const redirects = [["/tutorial/", "/tutorials/getting-started/"], ["/lessons/", "/tutorials/"]];
const port = 4800 + Math.floor(Math.random() * 500);
const server = await serve(path.resolve(process.env.CN_WEB_OUT), port);
const origin = `http://127.0.0.1:${port}`;
const browser = await puppeteer.launch({ headless: true, args: ["--no-sandbox"] });
let fail = 0;
const say = (ok, msg) => { if (!ok) fail++; console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); };

for (const r of pages) {
  const page = await browser.newPage();
  const foreign = [];
  page.on("request", (q) => { const u = q.url(); if (!u.startsWith(origin) && !u.startsWith("data:") && !u.startsWith("blob:")) foreign.push(u); });
  await page.goto(origin + r, { waitUntil: "networkidle0", timeout: 120000 });
  // MathJax typesets one formula per idle task: wait until every .math holds its output
  const m = await page.waitForFunction(() => {
    const items = document.querySelectorAll(".math");
    const done = [...items].filter((el) => el.querySelector("mjx-container")).length;
    return done === items.length ? { items: items.length } : false;
  }, { timeout: 120000, polling: 500 }).then((h) => h.jsonValue()).catch(() => null);
  const a11y = await page.evaluate(() => {
    const cs = [...document.querySelectorAll("mjx-container")];
    const withMml = cs.filter((c) => c.querySelector("mjx-assistive-mml math")).length;
    const hidden = cs.filter((c) => { const a = c.querySelector("mjx-assistive-mml"); return a && getComputedStyle(a).position === "absolute"; }).length;
    return { containers: cs.length, withMml, hiddenCopies: hidden };
  });
  say(foreign.length === 0, `${r}: ${foreign.length} request(s) to another origin${foreign.length ? ": " + [...new Set(foreign)].slice(0, 5).join(", ") : ""}`);
  if (a11y.containers || m?.items) {
    say(m !== null, `${r}: ${m ? m.items : "not all"} formula(s) typeset`);
    say(a11y.containers > 0 && a11y.withMml === a11y.containers && a11y.hiddenCopies === a11y.containers,
      `${r}: assistive MathML on ${a11y.withMml}/${a11y.containers} formulas (visually hidden: ${a11y.hiddenCopies})`);
  } else {
    console.log(`     ${r}: no formula on this page`);
  }
  await page.close();
}
if (!process.env.HASH) {
  for (const [from, to] of redirects) {
    const page = await browser.newPage();
    await page.goto(origin + from + "#x", { waitUntil: "networkidle0" });
    const at = new URL(page.url());
    say(at.pathname.endsWith(to) && at.hash === "#x", `${from} -> ${at.pathname}${at.hash} (expected ${to}#x)`);
    await page.close();
  }
}
await browser.close();
server.close();
process.exit(fail ? 1 : 0);
