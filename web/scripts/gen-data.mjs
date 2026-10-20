// web/scripts/gen-data.mjs — the website's data, read from the repository at build time.
//
//   node scripts/gen-data.mjs        (run by web/build.sh before the typecheck)
//
// Writes src/generated/{ledger,perf,tutorials,citation,stats}.json (gitignored). Nothing measured is
// typed by hand in the site: every number below is read from
//   paper/conatus.tex (Part I only, before \ifprivate): the Verification and Gallery longtables and the
//                      Performance section, converted from LaTeX to Unicode;
//   paper/refs.bib     (\cite{key} → "Author Year");
//   paper/generated/counts.tex (numbers of tests, runs, decisions, errors, scorecard);
//   results/perf/scorecard_*.json (most recent), results/INDEX.md, results/runs/<run_id>/*.json|log;
//   examples/README.md (gallery scripts), docs/site/tutorials/ (index + _generated/status.md);
//   CITATION.cff.
// Rules: a run_id is attached only when the paper cites it, when the scorecard (or results/INDEX.md)
// links the metric of the row to it, or when it is the run whose JSON the number is read from; a run_id
// that has no directory in results/runs/ is dropped with a warning. When a JSON value and the paper
// disagree at the paper's printed precision, the paper (validated text) wins and the disagreement is
// recorded in perf.json → divergences and printed. A table or sentence that is no longer found makes
// the script exit with status 1 rather than publish an empty or partial ledger.
import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ROOT = path.resolve(WEB, "..");
const OUT = path.join(WEB, "src", "generated");
const sources = new Set();
const warnings = [];
const divergences = [];

function fail(msg) { console.error(`gen-data: ERROR: ${msg}`); process.exit(1); }
function warn(msg) { warnings.push(msg); console.warn(`gen-data: warning: ${msg}`); }
function read(rel, { optional = false } = {}) {
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p)) { if (optional) return null; fail(`missing ${rel}`); }
  sources.add(rel);
  return fs.readFileSync(p, "utf8");
}
function readJson(rel) { return JSON.parse(read(rel)); }
/** JSON lines of a log / .out file (non-JSON lines ignored), each tagged with its line number. */
function readJsonLines(rel) {
  return read(rel).split("\n").map((l, i) => [l, i + 1]).filter(([l]) => l.startsWith("{"))
    .map(([l, n]) => { try { return { ...JSON.parse(l), __line: n }; } catch { return null; } }).filter(Boolean);
}
const lineOf = (text, index) => text.slice(0, index).split("\n").length;

// ---------------------------------------------------------------------------------------------
// Bibliography: \cite{key} → "Author Year" (1 author), "A & B Year" (2), "A et al. Year" (3+).
// ---------------------------------------------------------------------------------------------
const ACCENTS = { "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", c: "\u0327", v: "\u030C", u: "\u0306", "=": "\u0304" };
function detexName(s) {
  return s.replace(/\\([`'^"~=]|[cvu](?=[{ ]))\s*\{?\\?([A-Za-z])\}?/g, (_, a, ch) => ch + ACCENTS[a])
    .replace(/[{}]/g, "").replace(/\\&/g, "&").replace(/--/g, "–").normalize("NFC").trim();
}
function parseBib(text) {
  const out = {};
  const re = /@(\w+)\s*\{\s*([^,\s]+)\s*,/g;
  let m;
  while ((m = re.exec(text))) {
    // Fields up to the matching closing brace of the entry.
    let depth = 1, i = re.lastIndex;
    for (; i < text.length && depth > 0; i++) { if (text[i] === "{") depth++; else if (text[i] === "}") depth--; }
    const body = text.slice(re.lastIndex, i - 1);
    const field = (name) => {
      const fm = new RegExp(`\\b${name}\\s*=\\s*\\{`, "i").exec(body);
      if (!fm) { const q = new RegExp(`\\b${name}\\s*=\\s*(\\d+)`, "i").exec(body); return q ? q[1] : null; }
      let d = 1, j = fm.index + fm[0].length;
      const start = j;
      for (; j < body.length && d > 0; j++) { if (body[j] === "{") d++; else if (body[j] === "}") d--; }
      return body.slice(start, j - 1);
    };
    const authors = (field("author") || "").split(/\s+and\s+/).map((a) => {
      a = detexName(a);
      return a.includes(",") ? a.split(",")[0].trim() : a.split(/\s+/).pop();
    }).filter(Boolean);
    const year = field("year");
    const label = authors.length === 0 ? m[2] : authors.length === 1 ? authors[0]
      : authors.length === 2 ? `${authors[0]} & ${authors[1]}` : `${authors[0]} et al.`;
    out[m[2]] = { key: m[2], label: year ? `${label} ${year}` : label, title: detexName(field("title") || ""), year };
  }
  return out;
}
// The validation data (ledger, performance, counts) is read from the paper and the run archive of
// the development repository. A tree without them (the public repository) keeps the validation
// data shipped in src/generated/ and regenerates the rest (external review 0.1.0, finding 71).
const PRIVATE_INPUTS = fs.existsSync(path.join(ROOT, "paper", "conatus.tex"));
const BIB = PRIVATE_INPUTS ? parseBib(read("paper/refs.bib")) : {};

// ---------------------------------------------------------------------------------------------
// LaTeX → Unicode text. Superscripts made of digits and signs become Unicode (10⁻¹³, j²); other
// sub/superscripts are kept as a light markup `_{…}` / `^{…}` in the "rich" strings (rendered as
// <sub>/<sup> by the site) and flattened to "Re_c" in the plain strings.
// ---------------------------------------------------------------------------------------------
const GREEK = { alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ϵ", varepsilon: "ε", zeta: "ζ", eta: "η", theta: "θ", vartheta: "ϑ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ", sigma: "σ", tau: "τ", upsilon: "υ", phi: "ϕ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω", Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Xi: "Ξ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω" };
const SYMBOLS = { times: "×", cdot: "·", pm: "±", mp: "∓", le: "≤", leq: "≤", ge: "≥", geq: "≥", neq: "≠", ne: "≠", approx: "≈", sim: "∼", to: "→", rightarrow: "→", leftarrow: "←", infty: "∞", partial: "∂", nabla: "∇", ell: "ℓ", langle: "⟨", rangle: "⟩", dots: "…", ldots: "…", cdots: "⋯", quad: " ", qquad: "  ", mid: "|", lvert: "|", rvert: "|", vert: "|", star: "⋆", dagger: "†", circ: "∘", in: "∈", propto: "∝", ll: "≪", gg: "≫", S: "§", AA: "Å", colon: ":", prime: "′", sum: "∑", prod: "∏", int: "∫" };
const OPNAMES = new Set(["max", "min", "log", "ln", "exp", "sin", "cos", "tan", "tanh", "sinh", "cosh", "det", "sup", "inf", "arg", "Re", "Im"]);
const RELATIONS = new Set(["=", "<", ">", "≤", "≥", "≈", "→", "≠", "∼", "∝", "≪", "≫"]);
const SUP = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻", "−": "⁻", "+": "⁺", "*": "*", "′": "′", "†": "†" };

function texToRich(src, cites = []) {
  let i = 0;
  const n = src.length;
  function readGroup() { // after '{'
    let d = 1, s = i;
    for (; i < n && d > 0; i++) { if (src[i] === "\\") { i++; continue; } if (src[i] === "{") d++; else if (src[i] === "}") d--; }
    return src.slice(s, i - 1);
  }
  function readArg(math) { // one argument: {group}, \command or one char
    while (i < n && src[i] === " ") i++;
    if (src[i] === "{") { i++; return { raw: readGroup(), math }; }
    if (src[i] === "\\") { const s = i; i++; if (/[A-Za-z]/.test(src[i])) { while (i < n && /[A-Za-z]/.test(src[i])) i++; } else i++; return { raw: src.slice(s, i), math }; }
    return { raw: src[i++], math };
  }
  const sub = (x) => `_{${x}}`;
  const sup = (x) => ([...x].every((c) => SUP[c]) ? [...x].map((c) => SUP[c]).join("") : `^{${x}}`);
  function run(math, stopAtClose = false) {
    let out = "";
    while (i < n) {
      const c = src[i];
      if (c === "}" && stopAtClose) { i++; return out; }
      if (c === "$") { i++; math = !math; continue; }
      if (c === "\\") {
        i++;
        if (!/[A-Za-z]/.test(src[i])) {
          const s = src[i++];
          if (s === "," || s === ";" || s === ":") out += math ? "\u2009" : "\u2009";
          else if (s === " ") out += " ";
          else if (s === "!") { /* negative space */ }
          else if (s === "\\") out += " ";
          else if ("%&_{}#$".includes(s)) out += s;
          else if (ACCENTS[s]) { const a = readArg(math); out += detexName(`\\${s}{${a.raw}}`); }
          else out += s;
          continue;
        }
        let name = "";
        while (i < n && /[A-Za-z]/.test(src[i])) name += src[i++];
        if (GREEK[name]) { out += GREEK[name]; continue; }
        if (SYMBOLS[name] !== undefined) {
          const sym = SYMBOLS[name];
          out += math && RELATIONS.has(sym) ? ` ${sym} ` : sym;
          continue;
        }
        if (OPNAMES.has(name)) { out += name; continue; }
        if (name === "left" || name === "right" || name === "big" || name === "Big" || name === "bigl" || name === "bigr" || name === "displaystyle" || name === "small" || name === "noindent") continue;
        if (name === "cite") {
          let opt = "";
          if (src[i] === "[") { const e = src.indexOf("]", i); opt = src.slice(i + 1, e); i = e + 1; }
          const a = readArg(false);
          const keys = a.raw.split(",").map((k) => k.trim()).filter(Boolean);
          cites.push(...keys);
          out += keys.map((k) => {
            if (!BIB[k]) { warn(`\\cite{${k}}: key not in paper/refs.bib`); return k; }
            const who = BIB[k].label.replace(/ \d{4}$/, "");
            return BIB[k].year && out.replace(/[\s\u00A0]+$/, "").endsWith(who) ? `(${BIB[k].year})` : BIB[k].label;
          }).join("; ") + (opt ? `, ${opt}` : "");
          continue;
        }
        if (["code", "run", "texttt", "textbf", "emph", "textit", "textrm", "textsf", "text", "mathrm", "mathbf", "mathit", "mathsf", "mathcal", "operatorname", "boldsymbol", "mbox", "hbox"].includes(name)) {
          const a = readArg(math);
          const inner = texToRich(a.raw.replace(/\\_/g, "_"), cites);
          out += math && ["mathrm", "mathbf", "mathit", "operatorname", "mathcal", "boldsymbol"].includes(name) ? inner : inner;
          continue;
        }
        if (name === "overline" || name === "bar") { const a = readArg(math); out += texToRich(`$${a.raw}$`, cites) + "\u0304"; continue; }
        if (name === "hat" || name === "widehat") { const a = readArg(math); out += texToRich(`$${a.raw}$`, cites) + "\u0302"; continue; }
        if (name === "tilde" || name === "widetilde") { const a = readArg(math); out += texToRich(`$${a.raw}$`, cites) + "\u0303"; continue; }
        if (name === "dot") { const a = readArg(math); out += texToRich(`$${a.raw}$`, cites) + "\u0307"; continue; }
        if (name === "sqrt") { const a = readArg(math); const x = texToRich(`$${a.raw}$`, cites); out += /^[\p{L}\p{N}]+$/u.test(x) ? `√${x}` : `√(${x})`; continue; }
        if (name === "frac" || name === "tfrac" || name === "dfrac") { const a = texToRich(`$${readArg(math).raw}$`, cites); const b = texToRich(`$${readArg(math).raw}$`, cites); const w = (x) => (/^[\p{L}\p{N}.]+$/u.test(x) ? x : `(${x})`); out += `${w(a)}/${w(b)}`; continue; }
        if (name === "ref" || name === "label" || name === "footnote") { readArg(false); continue; }
        warn(`unknown LaTeX macro \\${name} (kept as text)`);
        out += name;
        continue;
      }
      if (c === "{") { i++; const g = readGroup(); out += math ? texToRich(`$${g}$`, cites).replace(/ ([=<>≤≥≈→]) /g, "$1") : texToRich(g, cites); continue; }
      if (c === "}") { i++; continue; }
      if (math && (c === "^" || c === "_")) {
        i++;
        const a = readArg(true);
        const inner = texToRich(`$${a.raw}$`, cites).trim();
        out += c === "^" ? sup(inner) : sub(inner);
        continue;
      }
      if (c === "~") { i++; out += "\u00A0"; continue; }
      if (c === "-" && !math) {
        if (src.startsWith("---", i)) { out += "—"; i += 3; continue; }
        if (src.startsWith("--", i)) { out += "–"; i += 2; continue; }
        out += "-"; i++; continue;
      }
      if (c === "`" && src.startsWith("``", i)) { out += "“"; i += 2; continue; }
      if (c === "'" && src.startsWith("''", i)) { out += "”"; i += 2; continue; }
      if (math) {
        if (c === " " || c === "\n" || c === "\t") { i++; continue; }
        if (c === "-") { out += "−"; i++; continue; }
        if (RELATIONS.has(c)) { out += ` ${c} `; i++; continue; }
        if (c === ",") { out += ","; i++; continue; }
      }
      if (c === "%" && src[i - 1] !== "\\") { const e = src.indexOf("\n", i); i = e < 0 ? n : e + 1; continue; } // comment
      out += c === "\n" ? " " : c;
      i++;
    }
    return out;
  }
  return run(false).replace(/[ \t]+/g, " ").replace(/ ([,.;:)])/g, "$1").replace(/\( /g, "(").replace(/\u00A0 /g, "\u00A0").trim();
}
/** Rich markup → plain text ("Re_c", "j_(m,n)²", "ω^max"). */
function plain(rich) {
  return rich.replace(/_\{([^{}]*)\}/g, (_, x) => ([...x].length === 1 ? `_${x}` : `_(${x})`))
    .replace(/\^\{([^{}]*)\}/g, (_, x) => ([...x].length === 1 ? `^${x}` : `^(${x})`));
}
// Cross-references point into the paper, not the site: "Eq.~\eqref{…}", "of Section~\ref{…}" are dropped.
const dropRefs = (tex) => tex.replace(/,?\s*Eq\.~?\\eqref\{[^}]*\}/g, "")
  .replace(/\s*(?:(?:of|in) )?(?:Sections?|Sec\.|Tables?|Fig\.|Figure)~?\\ref\{[^}]*\}/g, "");
function conv(tex) { const cites = []; const rich = texToRich(dropRefs(tex), cites); return { rich, plain: plain(rich), cites }; }

// ---------------------------------------------------------------------------------------------
// Numbers: deviation tokens and formatting.
// ---------------------------------------------------------------------------------------------
const SUPD = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
/** 3.5e-12 → "3.5×10⁻¹²" (sig significant digits). */
function sci(x, sig = 2) {
  if (x === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(x)));
  let m = +(x / 10 ** e).toPrecision(sig);
  let ee = e;
  if (Math.abs(m) >= 10) { m /= 10; ee++; }
  const ms = String(m);
  const pow = "10" + String(ee).split("").map((c) => SUPD[c]).join("");
  return ms === "1" ? pow : `${ms}×${pow}`;
}
/** First deviation token of a LaTeX result cell: a×10^{-n} / 10^{-n}, or "N (published) digits". */
function firstDeviation(tex) { return allDeviations(tex)[0] ?? null; }
/** Every deviation token of a LaTeX fragment, in text order. */
function allDeviations(tex) {
  const cands = [];
  const reSci = /(?:(\d+(?:\.\d+)?)\s*\\times\s*)?10\^\{?(-\d+)\}?/g;
  let m;
  while ((m = reSci.exec(tex))) {
    const mant = m[1] ? parseFloat(m[1]) : 1;
    const v = mant * 10 ** parseInt(m[2], 10);
    cands.push({ at: m.index, text: sci(v, m[1] ? m[1].replace(".", "").replace(/^0+/, "").length || 1 : 1), log: Math.log10(v) });
  }
  const reDig = /(?:all\s+)?(\d+)\s+(?:published\s+)?digits/g;
  while ((m = reDig.exec(tex))) cands.push({ at: m.index, text: `${m[1]} digits`, log: -parseInt(m[1], 10) });
  return cands.sort((a, b) => a.at - b.at);
}
/** Does the paper's printed number `texStr` agree with `x` at its printed precision? */
function agrees(texStr, x) {
  const s = String(texStr).trim();
  const dec = (s.split(".")[1] || "").length;
  return Math.abs(parseFloat(s) - x) <= 0.5 * 10 ** -dec + 1e-12 * Math.abs(x);
}
function reconcile(what, texStr, jsonVal, source, texSource) {
  if (texStr == null) return jsonVal;
  const t = parseFloat(texStr);
  if (jsonVal != null && !agrees(texStr, jsonVal)) {
    divergences.push({ what, tex: texStr, json: jsonVal, jsonSource: source, texSource, kept: "tex" });
  }
  return t;
}

/** Ledger, performance data and counts, from the paper and the run archive (private inputs). */
function genValidation() {
  // ---------------------------------------------------------------------------------------------
  // Runs, scorecard, INDEX.
  // ---------------------------------------------------------------------------------------------
  const RUNS_DIR = path.join(ROOT, "results", "runs");
  if (!fs.existsSync(RUNS_DIR)) fail("missing results/runs/");
  const runDirs = new Set(fs.readdirSync(RUNS_DIR).filter((d) => /^\d{8}_\d{6}_/.test(d)));
  sources.add("results/runs/");
  function checkRuns(ids, where) {
    return [...new Set(ids)].filter((id) => {
      if (runDirs.has(id)) return true;
      warn(`${where}: run_id ${id} has no directory in results/runs/ (link dropped)`);
      return false;
    });
  }
  const scorecards = fs.readdirSync(path.join(ROOT, "results", "perf")).filter((f) => /^scorecard_\d{4}-\d{2}-\d{2}\.json$/.test(f)).sort();
  if (!scorecards.length) fail("no results/perf/scorecard_YYYY-MM-DD.json");
  const SCORECARD_FILE = `results/perf/${scorecards[scorecards.length - 1]}`;
  const scorecard = readJson(SCORECARD_FILE);
  const metricById = Object.fromEntries(scorecard.metrics.map((m) => [m.id, m]));

  // results/INDEX.md: run_id → metrics listed for it (the column may be truncated with "(+k)").
  const indexMd = read("results/INDEX.md");
  const indexRows = [];
  for (const line of indexMd.split("\n")) {
    const m = line.match(/^\|\s*`(\d{8}_\d{6}_[^`]+)`\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|/);
    if (m) indexRows.push({ runId: m[1], status: m[4].trim(), metrics: m[5].split(",").map((s) => s.trim().replace(/\s*….*$/, "")) });
  }
  if (!indexRows.length) fail("results/INDEX.md: run table not found");
  /** Run of a metric: scorecard first, then the latest run of results/INDEX.md listing it. */
  function runOfMetric(id) {
    const sm = metricById[id];
    if (sm && sm.run_id) return { runId: sm.run_id, value: sm.value, via: SCORECARD_FILE };
    const rows = indexRows.filter((r) => r.metrics.includes(id) && r.status === "ok").map((r) => r.runId).sort();
    return rows.length ? { runId: rows[rows.length - 1], value: null, via: "results/INDEX.md" } : null;
  }

  // ---------------------------------------------------------------------------------------------
  // Paper, Part I.
  // ---------------------------------------------------------------------------------------------
  const TEX_FILE = "paper/conatus.tex";
  const texAll = read(TEX_FILE);
  // The public main text ends at \appendix (arXiv-ready paper, 2026-10-02) or, in older versions, at \ifprivate.
  const privIdx = [/^\\appendix\b/m, /^\\ifprivate\b/m].map((re) => texAll.search(re)).find((i) => i >= 0) ?? -1;
  if (privIdx < 0) fail(`${TEX_FILE}: neither \\appendix nor \\ifprivate (end of the main text) found`);
  const tex = texAll.slice(0, privIdx);
  function sectionSpan(re) {
    const m = re.exec(tex);
    if (!m) fail(`${TEX_FILE}: section ${re} not found in Part I`);
    const start = m.index;
    const next = tex.slice(start + m[0].length).search(/\\section\{/);
    return { start, end: next < 0 ? tex.length : start + m[0].length + next };
  }
  /** Rows of the first longtable after `from`: [{cells: [tex...], line}] */
  function longtable(from, label, ncols) {
    const b = tex.indexOf("\\begin{longtable}", from);
    const e = tex.indexOf("\\end{longtable}", b);
    if (b < 0 || e < 0) fail(`${TEX_FILE}: longtable of section ${label} not found`);
    const headEnd = tex.indexOf("\\endhead", b);
    if (headEnd < 0 || headEnd > e) fail(`${TEX_FILE}: ${label} longtable has no \\endhead`);
    // booktabs tables (arXiv-ready paper) also declare a footer (\bottomrule\endfoot) after the head
    const footEnd = tex.indexOf("\\endfoot", headEnd);
    const bodyStart = footEnd > 0 && footEnd < e ? footEnd + "\\endfoot".length : headEnd + "\\endhead".length;
    const body = tex.slice(bodyStart, e);
    const rows = [];
    let pos = 0;
    for (const chunk of body.split(/\\\\/)) {
      const at = bodyStart + pos;
      pos += chunk.length + 2;
      const clean = chunk.replace(/\\(hline|toprule|midrule|bottomrule)/g, "").trim();
      if (!clean || /^\\multicolumn\{/.test(clean)) continue;     // rules and group headings are not rows
      const lead = chunk.length - chunk.replace(/^[\s\\hline]*/, "").length;
      const cells = clean.split(/(?<!\\)&/).map((c) => c.trim());
      if (cells.length !== ncols) fail(`${TEX_FILE}:${lineOf(texAll, at)}: ${label} row has ${cells.length} cells, expected ${ncols}`);
      rows.push({ cells, line: lineOf(texAll, at + Math.max(0, lead)) });
    }
    if (!rows.length) fail(`${TEX_FILE}: ${label} longtable is empty`);
    return rows;
  }
  const runIdsIn = (s) => [...s.matchAll(/(\d{8}\\?_\d{6}\\?_[A-Za-z0-9\\_]+?)(?=[}\s,.;)]|$)/g)].map((m) => m[1].replace(/\\_/g, "_"));

  // Domains. Order = display order.
  const DOMAINS = [
    ["stability", "Hydrodynamic stability"],
    ["convection", "Convection"],
    ["shear", "Shear flows"],
    ["floquet", "Floquet–Hill"],
    ["adjoint", "Adjoints and non-modal growth"],
    ["dynamics", "Invariant solutions"],
    ["bifurcation", "Bifurcations and continuation"],
    ["bvp", "Boundary-value problems"],
    ["time-integration", "Time integration"],
    ["geometry", "Curvilinear geometries"],
    ["patterns", "Pattern formation"],
    ["operators", "Operators and compilers"],
    ["turbulence", "Turbulence"],
    ["performance", "Performance"],
    ["other", "Other"],
  ];
  // Classification of paper rows: first matching rule wins. `metrics` link a row to scorecard / INDEX
  // metrics (hence to a run_id). `deviation: null` suppresses the automatic deviation token when the
  // first small number of the row is not a deviation from the reference; `pick` selects the token.
  const RULES = [
    { re: /^Orr--Sommerfeld/, domain: "stability", metrics: [".os_digits"] },
    { re: /^Critical Reynolds/, domain: "stability", metrics: [".os_rec_rel_err"] },
    { re: /^Rayleigh--Bénard onset/, domain: "convection", metrics: [".rb_onset_rel_err"] },
    { re: /^Lane--Emden/, domain: "bvp", metrics: [".nlbvp_lane_emden_rel_err"] },
    { re: /^Bratu/, domain: "bifurcation" },
    { re: /^Heat equation/, domain: "time-integration" },
    { re: /^Rayleigh--Bénard 2D/, domain: "convection" },
    { re: /^Discrete adjoint/, domain: "adjoint" },
    { re: /^Mathieu/, domain: "floquet" },
    { re: /^Pulsatile scalar/, domain: "floquet" },
    { re: /^Transient growth/, domain: "adjoint", metrics: [".accord"] },
    { re: /^Kuramoto--Sivashinsky/, domain: "dynamics", metrics: [".ks_equilibria_tw", ".ks_orbits"] },
    { re: /^KS (folds|pitchfork)/, domain: "bifurcation" },
    { re: /^KS /, domain: "dynamics" },
    { re: /^Kolmogorov/, domain: "stability", metrics: [".kolmogorov_re_lin_rel_err"] },
    { re: /^(Disk|Annulus|Nonlinear diffusion|Navier--Stokes|Sphere|Ball|Shell|Shallow water)/, domain: "geometry" },
    { re: /^Rossby--Haurwitz/, domain: "geometry", deviation: null },
    { re: /^Hagen--Poiseuille/, domain: "stability" },
    { re: /^Taylor--Couette/, domain: "stability" },
    { re: /^3D channel/, domain: "shear" },
    { re: /^Rotating convection/, domain: "convection" },
    { re: /^Exact rates/, domain: "operators" },
    { re: /^Stress-free Rayleigh/, domain: "convection" },
    { re: /^Stress-free curved walls/, domain: "geometry" },
    { re: /^(Harmonic oscillator on the line|Heat kernels on the line|Interior layer)/, domain: "geometry" },
    { re: /^Same, equatorially symmetric/, domain: "convection" },
    { re: /^(Symmetry subspaces|Closed forms on every execution path)/, domain: "operators" },
    // Gallery
    { re: /^Kelvin--Helmholtz/, domain: "shear", deviation: null },
    { re: /^Taylor--Goldstein/, domain: "shear", pick: /J_n\(1\/2\)-1\/4\$? ?(?:=|from) ([^;&]*)/, last: true },
    { re: /^Womersley/, domain: "time-integration" },
    { re: /^Pulsatile Poiseuille/, domain: "floquet" },
    { re: /^(Swift--Hohenberg|Complex Ginzburg|Brusselator)/, domain: "patterns" },
    { re: /^(Rotating Rayleigh|Salt fingers|Rayleigh--Bénard 3D)/, domain: "convection" },
  ];

  // Gallery scripts from examples/README.md: best word overlap between the case name and README
  // (case + script path).
  const exReadme = read("examples/README.md");
  const exRows = exReadme.split("\n").map((l) => l.match(/^\|([^|]+)\|\s*`([^`]+\.py)`\s*\|/)).filter(Boolean)
    .map((m) => ({ text: m[1], script: `examples/${m[2]}` }));
  if (!exRows.length) fail("examples/README.md: gallery table not found");
  const words = (s) => new Set(s.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/\\[a-z]+/g, " ").split(/[^a-z0-9]+/).filter((w) => w.length >= 4));
  function scriptFor(caseTex) {
    const w = words(caseTex);
    let best = null, score = 0;
    for (const r of exRows) {
      const rw = words(r.text + " " + r.script);
      const s = [...w].filter((x) => rw.has(x)).length;
      if (s > score) { score = s; best = r.script; }
    }
    return best;
  }

  const entries = [];
  const usedIds = new Set();
  const slug = (s) => s.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60);
  function addEntry(e) {
    let id = slug(e.test), k = 2;
    while (usedIds.has(id)) id = `${slug(e.test)}-${k++}`;
    usedIds.add(id);
    entries.push({ id, ...e });
  }

  function paperRow(section, row, testTex, resultTex, refTex, extra = {}) {
    const rule = RULES.find((r) => r.re.test(testTex.replace(/\\'e/g, "\u00e9")));
    if (!rule) warn(`${TEX_FILE}:${row.line}: no domain rule for "${testTex.slice(0, 50)}" (domain "other")`);
    const t = conv(testTex), v = conv(resultTex), r = conv(refTex);
    // The summary deviation of a row (external review 0.1.0, finding 73: it was the first number of
    // the result text, whatever that number meant). In order: the relative error of a structured
    // metric of the scorecard; a token chosen by the row's rule (`pick`); the result text's only
    // deviation token when it has exactly one. A row with several tokens and no rule shows none
    // (the result column states them in words), with a build warning.
    let dev = null, devNote = null;
    const runIds = runIdsIn(resultTex + " " + testTex);
    for (const mid of rule?.metrics ?? []) {
      const hit = runOfMetric(mid);
      if (!hit) { warn(`metric ${mid} (row "${t.plain}") has no run in the scorecard or INDEX`); continue; }
      runIds.push(hit.runId);
      if (!dev && /rel_err$/.test(mid) && typeof hit.value === "number") {
        dev = { text: sci(hit.value), log: Math.log10(Math.abs(hit.value)) };
        devNote = "the relative error of the recorded measurement";
      }
    }
    if (!dev && rule?.pick) {
      const pm = rule.pick.exec(resultTex);
      if (!pm) fail(`${TEX_FILE}:${row.line}: deviation selector of "${t.plain}" no longer matches`);
      dev = rule.last ? allDeviations(pm[1]).pop() ?? null : firstDeviation(pm[1]);
      devNote = "as stated in the row";
    } else if (!dev && (!rule || rule.deviation !== null)) {
      const all = allDeviations(resultTex);
      if (all.length === 1) { dev = all[0]; devNote = "as stated in the row"; }
      else if (all.length > 1) warn(`${TEX_FILE}:${row.line}: "${t.plain}" states ${all.length} deviations and its rule picks none: no summary deviation`);
    }
    addEntry({
      domain: rule?.domain ?? "other",
      test: t.plain, value: v.plain, reference: r.plain,
      rich: { test: t.rich, value: v.rich, reference: r.rich },
      referenceKey: [...t.cites, ...v.cites, ...r.cites][0] ?? null,
      deviation: dev ? dev.text : null,
      deviationLog10: dev && Number.isFinite(dev.log) ? +dev.log.toFixed(3) : null,
      deviationNote: devNote,
      runIds: checkRuns(runIds, `${TEX_FILE}:${row.line}`),
      source: `${TEX_FILE}:${row.line}`,
      section,
      ...extra,
    });
  }

  const ver = sectionSpan(/\\section\{Verification\}/);
  const verRows = longtable(ver.start, "Verification", 3);
  if (!/\\(?:hline|toprule)\s*Test\s*&\s*Result\s*&\s*Reference/.test(tex.slice(tex.indexOf("\\begin{longtable}", ver.start), tex.indexOf("\\endhead", ver.start)))) fail(`${TEX_FILE}: Verification longtable header is no longer "Test & Result & Reference"`);
  for (const row of verRows) paperRow("Verification", row, row.cells[0], row.cells[1], row.cells[2], { script: null, lines: null });

  // The gallery table: after \subsection{Gallery…} (older paper) or the longtable labelled tab:gallery.
  const galLabel = tex.indexOf("\\label{tab:gallery}");
  const galStart = galLabel >= 0 ? tex.lastIndexOf("\\begin{longtable}", galLabel) : tex.search(/\\subsection\{Gallery/);
  // The gallery table moved to the supplement (after \appendix): the ledger then has no gallery rows,
  // said loudly; the Gallery page itself reads public/gallery/manifest.json.
  if (galStart < 0) warn(`${TEX_FILE}: no gallery longtable in Part I (tab:gallery is in the supplement): ledger without gallery rows`);
  const galRows = galStart < 0 ? [] : longtable(galStart, "Gallery", 4);
  for (const row of galRows) {
    paperRow("Gallery", row, row.cells[0], row.cells[1], row.cells[2], { script: scriptFor(row.cells[0]), lines: conv(row.cells[3]).plain });
  }

  // ---------------------------------------------------------------------------------------------
  // Provenance (appendix, tab:provenance): section label → run ids. The main text carries no run id
  // (arXiv-ready paper); each Performance or itemised Verification row takes the runs of its section.
  // ---------------------------------------------------------------------------------------------
  const provenance = (() => {
    const at = texAll.indexOf("\\label{tab:provenance}");
    if (at < 0) fail(`${TEX_FILE}: provenance table (\\label{tab:provenance}) not found`);
    const end = texAll.indexOf("\\end{longtable}", at);
    const foot = texAll.indexOf("\\endfoot", at);
    const rows = [];
    for (const chunk of texAll.slice(foot > 0 && foot < end ? foot + 8 : at, end).split(/\\\\/)) {
      const cells = chunk.split(/(?<!\\)&/);
      if (cells.length !== 2) continue;
      const labels = [...cells[0].matchAll(/\\ref\{([^}]+)\}/g)].map((m) => m[1]);
      const runs = runIdsIn(cells[1]);
      if (runs.length) rows.push({ result: conv(cells[0].replace(/\s*\([^()]*\\ref\{[^}]*\}[^()]*\)\s*$/, "")).plain, labels, runs });
    }
    if (!rows.length) fail(`${TEX_FILE}: provenance table is empty`);
    return rows;
  })();
  /** Runs of the provenance rows citing `label`, optionally only rows whose result matches `re`. */
  function runsOf(label, re = null) {
    const ids = provenance.filter((r) => r.labels.includes(label) && (!re || re.test(r.result))).flatMap((r) => r.runs);
    if (!ids.length) warn(`provenance: no run listed for ${label}${re ? ` (${re})` : ""}`);
    return checkRuns(ids, `provenance ${label}`);
  }

  // ---------------------------------------------------------------------------------------------
  // Itemised Verification subsections (one row per \item \emph{Title.}): periodic base flows and the
  // spherical-shell adjoint. `pick`: the deviation the row reports (null: none; default: the first).
  // ---------------------------------------------------------------------------------------------
  const ITEM_SECTIONS = [
    { label: "sec:verif-periodic", domain: "floquet", items: [
      { re: /^Optimal growth/, pick: /\$([^$]*10\^\{-\d+\})\$ below/ },
      { re: /^Same operator/ }, { re: /^With rotation/ }, { re: /^Floquet exponents/ }] },
    { label: "sec:verif-shell-adjoint", domain: "adjoint", items: [
      { re: /^Dot product/, pick: /ten steps: \$\\le([^$]+)\$/ }, { re: /^Self-adjoint/, pick: /\$([^$]+)\$ apart/ },
      { re: /^Rotation/, deviation: null }, { re: /^Nonlinear optimal/ }] },
  ];
  for (const sec of ITEM_SECTIONS) {
    const at = tex.indexOf(`\\label{${sec.label}}`);
    if (at < 0) fail(`${TEX_FILE}: \\label{${sec.label}} not found`);
    const head = tex.lastIndexOf("\\subsection{", at);
    const title = conv(tex.slice(head + 12, tex.indexOf("}", head + 12))).plain;
    const b = tex.indexOf("\\begin{itemize}", at), e = tex.indexOf("\\end{itemize}", b);
    const next = tex.indexOf("\\subsection", at + 10);
    if (b < 0 || e < 0 || (next > 0 && b > next)) fail(`${TEX_FILE}: ${sec.label}: itemize not found`);
    const runIds = runsOf(sec.label);
    let pos = b;
    const items = tex.slice(b, e).split(/\\item\s+/).slice(1);
    if (items.length !== sec.items.length) fail(`${TEX_FILE}: ${sec.label}: ${items.length} items, expected ${sec.items.length} (update ITEM_SECTIONS)`);
    for (const it of items) {
      pos = tex.indexOf(it, pos);
      const m = it.match(/^\\emph\{([^}]*)\}\s*([^]*)$/);
      if (!m) fail(`${TEX_FILE}:${lineOf(texAll, pos)}: ${sec.label}: item without \\emph{title}`);
      const name = m[1].replace(/\.$/, "");
      const rule = sec.items.find((r) => r.re.test(name));
      if (!rule) fail(`${TEX_FILE}:${lineOf(texAll, pos)}: ${sec.label}: no rule for item "${name}"`);
      const body = m[2].trim().replace(/^[:.]\s*/, "");
      let dev = null;
      if (rule.pick) { const pm = rule.pick.exec(body); if (!pm) fail(`${TEX_FILE}: ${sec.label} "${name}": deviation selector no longer matches`); dev = firstDeviation(`$${pm[1]}$`); }
      else if (rule.deviation !== null) dev = firstDeviation(body);
      const t = `${title}: ${name}`, v = conv(body);
      addEntry({
        domain: sec.domain, test: t, value: v.plain, reference: "", rich: { test: t, value: v.rich, reference: "" },
        referenceKey: v.cites[0] ?? null, deviation: dev ? dev.text : null,
        deviationLog10: dev && Number.isFinite(dev.log) ? +dev.log.toFixed(3) : null, deviationNote: null,
        runIds, source: `${TEX_FILE}:${lineOf(texAll, pos)}`, section: "Verification", script: null, lines: null,
      });
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Performance section: sentences → ledger rows, and the data of the one chart (perf.json).
  // ---------------------------------------------------------------------------------------------
  // The Performance prose is read sentence by sentence (spaces in the patterns below match any run of
  // white space, line breaks included). When a sentence is no longer found, the Performance part of the
  // Validation page is left out with a loud build warning, never shown half-read.
  function genPerf() {
    const perfSpan = sectionSpan(/\\section\{Performance\}/);
    const perfTex = tex.slice(perfSpan.start, perfSpan.end);
    /** Find a pattern (string, single spaces = any white space) in the Performance section. */
    function perfFind(src, what) {
      const re = new RegExp(src.replace(/ /g, "\\s+"));
      const m = re.exec(perfTex);
      if (!m) throw Object.assign(new Error(`${TEX_FILE}: Performance section: sentence "${what}" not found (${re})`), { perfMissing: true });
      return { m, line: lineOf(texAll, perfSpan.start + m.index), at: `${TEX_FILE}:${lineOf(texAll, perfSpan.start + m.index)}` };
    }
    const N = "(\\d+(?:\\.\\d+)?)", G = "\\$(\\d+)\\\\times(\\d+)\\$";
    const row = (test, value, reference, runIds, at, dev = null, extra = {}) => {
      const v = conv(value), r = conv(reference);
      addEntry({
        domain: extra.domain ?? "performance", test, value: v.plain, reference: r.plain, rich: { test, value: v.rich, reference: r.rich },
        referenceKey: [...v.cites, ...r.cites][0] ?? null, deviation: dev ? dev.text : null,
        deviationLog10: dev && Number.isFinite(dev.log) ? +dev.log.toFixed(3) : null, deviationNote: dev?.note ?? null,
        runIds, source: at, section: "Performance", script: null, lines: null,
      });
    };

    // -- (a) Rayleigh–Bénard 2D on one GPU against Dedalus with 64 MPI ranks on one CPU node (sec:perf-dedalus).
    const hw = perfFind(`Dedalus~([\\d.]+) running with (\\d+) MPI ranks on one CPU node`, "CPU reference");
    const a100 = perfFind(`RK443 and SBDF2 steps on one A100 take ${N} and ${N}\\\\,ms at ${G}, ${N} and ${N}\\\\,ms at ${G} and ${N} and ${N}\\\\,ms at ${G}, i\\.e\\.\\\\ ${N}/${N}, ${N}/${N} and ${N}/${N} times faster than Dedalus`, "A100 times and ratios");
    const l40 = perfFind(`On one L40S, measured with one level per step, the factors were ${N}/${N}, ${N}/${N} and ${N}/${N}`, "L40S ratios");
    const tgt = perfFind(`Our target of a factor (\\d+) on every grid, each grid and scheme counted separately, is met; the smallest factor, ${N}, is for (RK443|SBDF2) at ${G}`, "target met, smallest factor");
    const core = perfFind(`On a single CPU core Conatus and Dedalus run at parity \\(ratio ${N}--${N}\\)`, "single core parity");
    const rbRuns = runsOf("sec:perf-dedalus");
    const gateRun = rbRuns.find((r) => /a100/.test(r));
    const verdict = gateRun ? (() => { try { return readJson(`results/runs/${gateRun}/j0_verdict.json`); } catch { return null; } })() : null;
    if (verdict) sources.add(`results/runs/${gateRun}/j0_verdict.json`);
    else warn(`RB 2D: no j0_verdict.json in the A100 run of sec:perf-dedalus (times not reconciled)`);
    const am = a100.m, target = +tgt.m[1];
    // the paper states the target as met (2026-10-02); the rows below are still checked against it
    const worst = { factor: +tgt.m[2], scheme: tgt.m[3], grid: `${tgt.m[4]}×${tgt.m[5]}` };
    const rb = [0, 1, 2].map((k) => {
      const nx = +am[3 + 4 * k], nz = +am[4 + 4 * k];
      const grid = `${nx}×${nz}`;
      const out = { grid, nx, nz };
      for (const [j, scheme] of [[0, "RK443"], [1, "SBDF2"]]) {
        const v = verdict?.rows?.find((r) => r.nx === nx && r.nz === nz && r.scheme === scheme) ?? null;
        const ms = reconcile(`RB 2D ${grid} ${scheme}: A100 ms/step`, am[1 + 4 * k + j], v ? v.conatus_ms : null, `results/runs/${gateRun}/j0_verdict.json`, a100.at);
        const sp = reconcile(`RB 2D ${grid} ${scheme}: A100 speed-up`, am[13 + 2 * k + j], v ? v.ratio : null, `results/runs/${gateRun}/j0_verdict.json`, a100.at);
        out[scheme] = { a100Ms: ms, a100Speedup: sp, l40sSpeedup: +l40.m[1 + 2 * k + j], dedalusMs: v ? v.dedalus_ms : null };
      }
      out.met = out.RK443.a100Speedup >= target && out.SBDF2.a100Speedup >= target;
      if (!out.met) fail(`RB 2D ${grid}: the paper says the ${target}× target is met, the A100 speed-ups say not`);
      return out;
    });
    for (const r of rb) {
      row(`Rayleigh–Bénard 2D, ${r.grid}: one GPU against Dedalus (${hw.m[2]} MPI ranks)`,
        `A100: ${r.RK443.a100Ms} and ${r.SBDF2.a100Ms}\\,ms per step (RK443, SBDF2), ${r.RK443.a100Speedup}× and ${r.SBDF2.a100Speedup}× faster; L40S: ${r.RK443.l40sSpeedup}× and ${r.SBDF2.l40sSpeedup}× (one level per step in the banded solves)${r.met ? "" : ` — below the ${target}× target`}`,
        `target ${target}× on every grid; Dedalus ${hw.m[1]} with ${hw.m[2]} MPI ranks on one CPU node`, rbRuns, a100.at);
    }
    row("Rayleigh–Bénard 2D on one CPU core against Dedalus on one core", `time ratio ${core.m[1]}–${core.m[2]} (parity)`, `Dedalus ${hw.m[1]}, same case`, rbRuns, core.at);

    // -- (b) Against hand-written solvers (sec:perf-handwritten).
    const hand = perfFind(`runs at ${N}, ${N} and ${N}\\\\,ms per step at \\$64\\^3\\$, \\$128\\^3\\$ and \\$192\\\\times129\\\\times192\\$ on one L40S`, "hand-written channel times");
    const prim = perfFind(`validated against Dedalus to \\$([^$]+)\\$\\) is ${N}--${N} times slower and needs ${N} times more memory`, "primitive channel");
    const pt = perfFind(`runs at ${N}, ${N} and ${N}\\\\,ms per step, i\\.e\\.\\\\ ${N}, ${N} and ${N} times the hand-written solver, with ${N}, ${N} and ${N}\\\\,GB of device memory against ${N}, ${N} and ${N}\\\\,GB`, "poloidal–toroidal channel");
    const tg = perfFind(`a transient-growth production point runs in ${N}\\\\,s on one GPU against ${N}\\\\,s for the original study code, with ${N}\\\\% of its user code`, "transient growth production point");
    const fq = perfFind(`root of Sec\\.~\\\\ref\\{sec:app-stokes\\} takes ${N}\\\\,s on one GPU against ${N}\\\\,s on the host and ${N}\\\\,s for a quadruple-precision C kernel, with ${N}\\\\% of its user code`, "Floquet–Hill root");
    const fb = perfFind(`A batch of (\\d+) Floquet exponents costs ${N}\\\\,s per exponent \\(${N} times the host rate\\), while a single evaluation is ${N} times slower`, "Floquet batch");
    row("3D channel, poloidal–toroidal formulation written as an ordinary problem, against a hand-written CUDA solver (one L40S)",
      `${pt.m[1]}, ${pt.m[2]} and ${pt.m[3]}\\,ms per step at $64^3$, $128^3$, $192\\times129\\times192$: ${pt.m[4]}, ${pt.m[5]} and ${pt.m[6]} times the hand-written solver; ${pt.m[7]}, ${pt.m[8]} and ${pt.m[9]}\\,GB of device memory`,
      `hand-written $v$--$\\eta$ solver~\\cite{kmm1987}: ${hand.m[1]}, ${hand.m[2]} and ${hand.m[3]}\\,ms; ${pt.m[10]}, ${pt.m[11]} and ${pt.m[12]}\\,GB`, runsOf("sec:perf-handwritten", /Channel/), pt.at);
    row("3D channel, general primitive-variable formulation, against the same hand-written solver",
      `${prim.m[2]}–${prim.m[3]} times slower, ${prim.m[4]} times more memory`, "hand-written solver; Dedalus, same formulation", runsOf("sec:perf-handwritten", /Channel/), prim.at, firstDeviation(`$${prim.m[1]}$`) && { ...firstDeviation(`$${prim.m[1]}$`), note: "against Dedalus, same formulation" });
    row(`Floquet–Hill, a batch of ${fb.m[1]} exponents on one GPU`, `${fb.m[2]}\\,s per exponent (${fb.m[3]} times the host rate); a single evaluation is ${fb.m[4]} times slower than on the host (latency-bound recurrence)`, "the host engine", [], fb.at, null, { domain: "floquet" });

    // -- (c) Symmetry subspaces (sec:perf-symmetry).
    const sy = perfFind(`restricting to a symmetry subspace reduces the time per step by ([^]*?) with trajectories identical to \\$([^$]+)\\$`, "symmetry subspaces");
    const si = perfFind(`shift-invert solve of the rotating-convection onset is ${N} times faster`, "shift-invert in a subspace");
    row("Symmetry subspaces on one CPU core: time per step, full over reduced", sy.m[1].replace(/,\s*$/, ""), "the full problem, same trajectory", runsOf("sec:perf-symmetry"), sy.at, firstDeviation(`$${sy.m[2]}$`) && { ...firstDeviation(`$${sy.m[2]}$`), note: "reduced against full trajectories" });
    row("Rotating-convection onset in the equatorially symmetric subspace", `shift-invert solve ${si.m[1]} times faster`, "the full problem", [], si.at, null, { domain: "convection" });

    // -- (d) Several GPUs: weak scaling (sec:perf-multigpu), against the 70 % target (open).
    const w1 = perfFind(`one GPU takes ${N}\\\\,ms per step\\. From one to two GPUs on two nodes the efficiency is ${N}\\\\%; on four GPUs over three nodes it is ${N}\\\\% \\(${N}\\\\,ms\\) and on eight L40S over three nodes \\(three, three and two ranks\\) ${N}\\\\% \\(${N}\\\\,ms\\)\\. Four A100 on two nodes reach ${N}\\\\%`, "weak scaling");
    const wt = perfFind(`With \\$(\\d+)\\\\times(\\d+)\\\\times(\\d+)\\$ modes per GPU`, "weak scaling grid");
    // the honest limit: the network of the busiest node bounds the efficiency; best measured on 8 GPUs
    const bound = perfFind(`/B\\) = ${N}\\\\%\\$ on the eight-GPU layouts we obtained`, "network bound");
    const unshared = perfFind(`Where a node's network is not shared, the efficiency is ${N}\\\\% on four GPUs over four nodes and ${N}\\\\% on two`, "unshared network");
    const best8 = perfFind(`On eight GPUs the best measured efficiency is ${N}\\\\%`, "best on eight GPUs");
    const goal = 70;   // the dashed line of fig:scaling ("the dashed line marks 70\%")
    perfFind(`the dashed line marks ${goal}\\\\%`, "70 % line of the scaling figure");
    const mg = runsOf("sec:perf-multigpu");
    const pts = [
      { label: "2 GPUs, 2 nodes (L40S)", gpus: 2, nodes: 2, efficiency: +w1.m[2] },
      { label: "4 GPUs, 3 nodes (L40S)", gpus: 4, nodes: 3, efficiency: +w1.m[3] },
      { label: "4 GPUs, 2 nodes (A100)", gpus: 4, nodes: 2, efficiency: +w1.m[7] },
      { label: "8 GPUs, 3 nodes (L40S)", gpus: 8, nodes: 3, efficiency: +w1.m[5] },
    ];
    row(`Weak scaling, 3D channel step (${wt.m[1]}×${wt.m[2]}×${wt.m[3]} modes per GPU)`,
      `one GPU ${w1.m[1]}\\,ms per step; efficiency ${w1.m[2]}\\% on 2 GPUs (2 nodes), ${w1.m[3]}\\% on 4 (3 nodes), ${w1.m[5]}\\% on 8 L40S (3 nodes), ${w1.m[7]}\\% on 4 A100 (2 nodes); best on 8 GPUs ${best8.m[1]}\\%, below the ${goal}\\% target; eight ranks equal one rank bit for bit`,
      `target ${goal}\\%, not met on 8 GPUs: two or three GPUs share one node's network path, which bounds any implementation of the exchange at ${bound.m[1]}\\%; with one GPU per node, ${unshared.m[1]}\\% on 4 nodes and ${unshared.m[2]}\\% on 2`, mg, w1.at);

    const perf = {
      generated: new Date().toISOString(),
      rb2d: {
        target, worst, dedalusVersion: hw.m[1], dedalusRanks: +hw.m[2], rows: rb, verdictRun: verdict ? gateRun : null,
        texSource: a100.at, runIds: rbRuns,
      },
      multigpu: { goal, points: pts, modesPerGpu: `${wt.m[1]}×${wt.m[2]}×${wt.m[3]}`, texSource: w1.at, runIds: mg },
      divergences,
    };
    return { perf, rb, pts };
  }
  const entriesBeforePerf = entries.length;
  let P = null;
  try { P = genPerf(); } catch (e) {
    if (!e.perfMissing) throw e;
    entries.length = entriesBeforePerf;                      // no half-read performance rows
    warn(`${e.message}: the Performance part of the Validation page is left out until gen-data follows the new wording`);
  }
  const perf = P ? { available: true, ...P.perf } : { available: false, generated: new Date().toISOString(), divergences };

  // ---------------------------------------------------------------------------------------------
  // Counts, tutorials, citation.
  // ---------------------------------------------------------------------------------------------
  const countsTex = read("paper/generated/counts.tex");
  const cnt = (name) => { const m = countsTex.match(new RegExp(`\\\\newcommand\\{\\\\${name}\\}\\{([^}]*)\\}`)); if (!m) fail(`paper/generated/counts.tex: \\${name} not found`); return m[1]; };
  const counts = {
    tests: +cnt("NTests"), runs: +cnt("NRuns"), decisions: +cnt("NDecisions"), errors: +cnt("NErrors"),
    scorecardPass: +cnt("ScorePass"), scorecardFail: +cnt("ScoreFail"), scorecardNotMeasured: +cnt("ScoreNM"), date: cnt("GenDate"),
  };
  return { entries, counts, runDirs, perf, P, galStart, SCORECARD_FILE, DOMAINS };
}
const V = PRIVATE_INPUTS ? genValidation() : null;
const runDirs = V ? V.runDirs : new Set();

const tutIndex = read("docs/site/tutorials/index.md");
// Execution status: docs/site/tutorials/outputs/STATUS.json (written by docs/site/run_tutorials.py)
// is the primary source; the rendered table _generated/status.md is the fallback.
const statusJsonText = read("docs/site/tutorials/outputs/STATUS.json", { optional: true });
const statusMd = statusJsonText ? null : read("docs/site/tutorials/_generated/status.md", { optional: true });
let statusRows;
if (statusJsonText) {
  statusRows = Object.entries(JSON.parse(statusJsonText)).map(([name, r]) => {
    const m = name.match(/^t(\d+)[a-z]?_/);
    return m && { script: `${name}.py`, n: +m[1], status: String(r.status ?? "").toLowerCase(), time: r.elapsed_s != null ? `${r.elapsed_s} s` : "", run: r.run_id ?? "local", commit: r.commit ?? null };
  }).filter(Boolean).sort((a, b) => a.script.localeCompare(b.script));
} else {
  statusRows = (statusMd ?? "").split("\n").map((l) => l.match(/^\|\s*`(t(\d+)[a-z]?_[^`]+\.py)`\s*\|\s*(?:\{[^}]*\})?`?(\w+)`?\s*\|\s*([^|]*)\|\s*`([^`]+)`\s*(?:\(([^)]*)\))?\s*\|/)).filter(Boolean)
    .map((m) => ({ script: m[1], n: +m[2], status: m[3].toLowerCase(), time: m[4].trim(), run: m[5], commit: m[6] ?? null }));
}
const statusSource = statusJsonText ? "docs/site/tutorials/outputs/STATUS.json" : statusMd ? "docs/site/tutorials/_generated/status.md" : null;
const tutorials = [];
for (const line of tutIndex.split("\n")) {
  const m = line.match(/^\|\s*(\d+)\s*\|\s*\{doc\}`([^`]+)`\s*\|([^|]*)\|([^|]*)\|/);
  if (!m) continue;
  const n = +m[1], slugT = m[2];
  const md = read(`docs/site/tutorials/${slugT}.md`, { optional: true });
  const title = md ? (md.match(/^#\s+(?:\d+\.\s*)?(.+)$/m)?.[1] ?? slugT) : slugT;
  const mdToText = (s) => conv(s.trim().replace(/`([^`]+)`/g, "\\code{$1}").replace(/_/g, "\\_").replace(/\$([^$]*)\$/g, (x) => x.replace(/\\_/g, "_"))).plain;
  const scripts = statusRows.filter((r) => r.n === n).map((r) => ({
    script: r.script, status: r.status, time: r.time, run: r.run, commit: r.commit,
    date: /^\d{8}_\d{6}/.test(r.run) ? `${r.run.slice(0, 4)}-${r.run.slice(4, 6)}-${r.run.slice(6, 8)}` : null,
    runId: runDirs.has(r.run) ? r.run : null,
  }));
  tutorials.push({
    n, slug: slugT, title: mdToText(title), learn: mdToText(m[3]), reference: mdToText(m[4]).replace(/^—$/, "") || null,
    status: !scripts.length ? "not-run" : scripts.every((s) => s.status === "pass") ? "pass" : scripts.some((s) => s.status === "pass") ? "partial" : "fail",
    scripts,
  });
}
if (tutorials.length < 1) fail("docs/site/tutorials/index.md: tutorial table not found");

/** Minimal YAML reader for CITATION.cff (maps, lists of maps, scalars; no anchors). */
function parseCff(text) {
  const lines = text.split("\n").filter((l) => l.trim() && !l.trim().startsWith("#"));
  let k = 0;
  const scalar = (s) => { s = s.trim(); if (/^".*"$/.test(s) || /^'.*'$/.test(s)) return s.slice(1, -1); if (/^-?\d+(\.\d+)?$/.test(s) && !/^\d{4}-/.test(s)) return +s; return s; };
  const ind = (l) => l.match(/^ */)[0].length;
  function block(indent) {
    const first = lines[k];
    if (first.trim().startsWith("- ")) {
      const arr = [];
      while (k < lines.length && ind(lines[k]) === indent && lines[k].trim().startsWith("- ")) {
        const rest = lines[k].trim().slice(2);
        if (/^[\w-]+:/.test(rest)) { lines[k] = " ".repeat(indent + 2) + rest; arr.push(block(indent + 2)); }
        else { arr.push(scalar(rest)); k++; }
      }
      return arr;
    }
    const obj = {};
    while (k < lines.length && ind(lines[k]) === indent && !lines[k].trim().startsWith("- ")) {
      const m = lines[k].trim().match(/^([\w-]+):\s*(.*)$/);
      if (!m) { k++; continue; }
      k++;
      if (m[2] !== "") obj[m[1]] = scalar(m[2]);
      else if (k < lines.length && ind(lines[k]) >= indent) obj[m[1]] = block(ind(lines[k]));
    }
    return obj;
  }
  return block(0);
}
const cff = parseCff(read("CITATION.cff"));
if (!cff.title || !cff.authors) fail("CITATION.cff: title or authors missing");
const bibAuthors = (as) => as.map((a) => `${a["family-names"]}, ${a["given-names"]}`).join(" and ");
const year = String(cff["date-released"] ?? "").slice(0, 4) || String(cff["preferred-citation"]?.year ?? "");
const keyBase = `${String(cff.authors[0]["family-names"]).toLowerCase()}${year}`;
const bibtexSoftware = [
  `@software{${keyBase}conatus,`,
  `  author  = {${bibAuthors(cff.authors)}},`,
  `  title   = {{${cff.title}}},`,
  `  year    = {${year}},`,
  cff.version ? `  version = {${cff.version}},` : null,
  cff.license ? `  license = {${cff.license}},` : null,
  cff["repository-code"] ? `  url     = {${cff["repository-code"]}},` : null,
  `}`,
].filter(Boolean).join("\n");
const pc = cff["preferred-citation"];
const bibtexPaper = pc ? [
  `@misc{${keyBase}paper,`,
  `  author = {${bibAuthors(pc.authors ?? cff.authors)}},`,
  `  title  = {{${pc.title}}},`,
  `  year   = {${pc.year ?? year}},`,
  // the public site cites the preprint as such; the repository-internal note stays in CITATION.cff
  pc.notes ? `  note   = {Preprint},` : null,
  `}`,
].filter(Boolean).join("\n") : null;
const citation = { cff, bibtex: { software: bibtexSoftware, paper: bibtexPaper }, source: "CITATION.cff" };

// ---------------------------------------------------------------------------------------------
// Write.
// ---------------------------------------------------------------------------------------------
let commit = null;
try { commit = execSync("git rev-parse --short HEAD", { cwd: ROOT, stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); } catch { /* no git */ }
const generated = new Date().toISOString();
fs.mkdirSync(OUT, { recursive: true });
const write = (name, obj) => fs.writeFileSync(path.join(OUT, name), JSON.stringify(obj, null, 1) + "\n");
write("tutorials.json", { generated, source: "docs/site/tutorials/index.md", statusSource, tutorials });
write("citation.json", citation);
if (V) {
  const { entries, counts, perf, P, galStart, SCORECARD_FILE, DOMAINS } = V;
  const domainsUsed = DOMAINS.filter(([id]) => entries.some((e) => e.domain === id)).map(([id, label]) => ({ id, label, count: entries.filter((e) => e.domain === id).length }));
  const nVer = entries.filter((e) => e.section === "Verification").length;
  const nGal = entries.filter((e) => e.section === "Gallery").length;
  if (nVer < 10 || (galStart >= 0 && nGal < 3)) fail(`ledger too small (Verification ${nVer}, Gallery ${nGal}): parsing failed`);
  const ledger = {
    generated, commit, sources: [...sources].sort(), counts, scorecard: SCORECARD_FILE,
    version: (read("python/conatus/__init__.py").match(/^__version__ = "([^"]+)"/m) ?? [])[1] ?? null,
    domains: domainsUsed,
    entries: entries.sort((a, b) => DOMAINS.findIndex(([d]) => d === a.domain) - DOMAINS.findIndex(([d]) => d === b.domain)),
  };
  // Keep paper order inside a domain (stable sort above) and expose it.
  ledger.entries.forEach((e, k) => { e.order = k; });
  const stats = {
    generated, commit, counts,
    ledger: {
      entries: entries.length, verification: nVer, gallery: nGal, performance: entries.length - nVer - nGal,
      domains: domainsUsed.length, withRunId: entries.filter((e) => e.runIds.length).length,
      withDeviation: entries.filter((e) => e.deviationLog10 != null).length,
    },
    runsArchived: runDirs.size,
    tutorials: { total: tutorials.length, passing: tutorials.filter((t) => t.status === "pass").length },
    warnings: warnings.length, divergences: divergences.length,
  };
  write("ledger.json", ledger);
  write("perf.json", { ...perf, sources: [...sources].filter((s) => s.startsWith("results/")).sort() });
  write("stats.json", stats);

  const byDomain = domainsUsed.map((d) => `${d.id} ${d.count}`).join(", ");
  console.log(`gen-data: ledger ${entries.length} entries (Verification ${nVer}, Gallery ${nGal}, Performance ${entries.length - nVer - nGal}); ${byDomain}`);
  console.log(P ? `gen-data: perf: rb2d ${P.rb.length} grids (target met on ${P.rb.filter((r) => r.met).length}), weak scaling ${P.pts.length} points; tutorials ${tutorials.length}; ${warnings.length} warnings` : `gen-data: perf: left out (paper wording); tutorials ${tutorials.length}; ${warnings.length} warnings`);
  for (const d of divergences) console.log(`gen-data: divergence: ${d.what}: tex ${d.tex} (${d.texSource}) vs json ${d.json} (${d.jsonSource}); kept ${d.kept}`);
} else {
  for (const f of ["ledger.json", "perf.json", "stats.json"]) {
    if (!fs.existsSync(path.join(OUT, f))) fail(`no paper/ in this tree and no shipped src/generated/${f}: the validation data cannot be made here`);
  }
  console.log(`gen-data: no paper/ in this tree: the validation data shipped in src/generated/ is kept; tutorials ${tutorials.length}; ${warnings.length} warnings`);
}

// ---------------------------------------------------------------------------------------------
// Documentation map (src/pages/Documentation.tsx): read from the Sphinx sources, so the site's map
// follows the documentation as it is written. docs/site/guide/index.md holds one {toctree} per part
// (":caption: <part>"); each entry's card is the page's "# Title" and its first paragraph. The
// dynamics pages (guide/dynamics/index.md) are listed the same way. Strict: a listed page without a
// title fails the build.
// ---------------------------------------------------------------------------------------------
function genDocsMap() {
  const SITE_DIR = "docs/site";
  const firstPara = (md) => {
    // fenced code and directive blocks first (they contain blank lines), then labels and the title
    const body = md.replace(/^(`{3,}|:{3,})[^\n]*\n[\s\S]*?^\1\s*$/gm, "").replace(/^\(.*?\)=\s*$/gm, "").replace(/^#\s+.*$/m, "");
    for (const block of body.split(/\n\s*\n/)) {
      const b = block.trim();
      if (!b || /^(```|:::|\{|#|\||<|-|\*|>|\d+\.)/.test(b)) continue;
      return b.replace(/\s+/g, " ").replace(/\{[a-z:]+\}`([^`<]*?)(?:\s*<[^>]*>)?`/g, "$1").replace(/`([^`]+)`/g, "$1")
        .replace(/\*\*([^*]+)\*\*/g, "$1").replace(/\*([^*]+)\*/g, "$1").replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
        .replace(/\$([^$]+)\$/g, "$1");
    }
    return "";
  };
  const page = (doc) => {
    const rel = `${SITE_DIR}/${doc}.md`;
    const md = read(rel, { optional: true });
    if (md == null) fail(`docs map: ${rel} is listed in a toctree but does not exist`);
    const t = md.match(/^#\s+(.+)$/m);
    if (!t) fail(`docs map: ${rel} has no "# Title"`);
    const text = firstPara(md);
    const sentence = (text.match(/^.*?[.!?](?=\s|$)/) ?? [text])[0];
    return { title: t[1].replace(/`/g, "").trim(), text: sentence, href: `${doc}.html` };
  };
  const toctrees = (rel, base) => {
    const md = read(rel, { optional: true });
    if (md == null) return [];
    const parts = [];
    for (const m of md.matchAll(/```\{toctree\}\n([\s\S]*?)```/g)) {
      const opts = {}, docs = [];
      for (const l of m[1].split("\n")) {
        const o = l.match(/^:(\w+):\s*(.*)$/);
        if (o) opts[o[1]] = o[2];
        else if (l.trim() && !l.trim().startsWith(":")) docs.push(l.trim());
      }
      if (docs.length) parts.push({ caption: opts.caption ?? null, docs: docs.map((d) => base + d.replace(/\.md$/, "")) });
    }
    return parts;
  };
  const guide = toctrees(`${SITE_DIR}/guide/index.md`, "guide/").map((p) => ({
    caption: p.caption ?? "User guide",
    entries: p.docs.map(page),
  }));
  const dyn = toctrees(`${SITE_DIR}/guide/dynamics/index.md`, "guide/dynamics/").flatMap((p) => p.docs.map(page));
  write("docs-map.json", { generated, source: `${SITE_DIR}/guide/index.md`, guide, dynamics: dyn });
  console.log(`gen-data: docs map: ${guide.length} parts, ${guide.reduce((a, p) => a + p.entries.length, 0)} guide pages, ${dyn.length} dynamics pages`);
}
genDocsMap();

// ---------------------------------------------------------------------------------------------
// Tutorial output freshness (read by src/lib/snippet.ts). An output is shown next to its code only if
// it was produced by that code. docs/site/run_tutorials.py writes the sha256 of the script it executed
// into the first line of the output ("| script sha256 <16 hex>"): the output is fresh when that hash is
// the hash of the script now (external review 0.1.0, finding 72: file dates said nothing about what
// ran). An output written before the hash was recorded falls back to the dates (the output must be at
// least as recent as the script: file times when either has uncommitted changes, else the times of
// their last commits), said by a warning. A stale output is never shown: the page shows the code alone
// with a note, and this script warns.
// ---------------------------------------------------------------------------------------------
function genTutorialFreshness() {
  const CODE_DIR = "docs/site/tutorials/code", OUT_DIR = "docs/site/tutorials/outputs";
  const git = (args) => { try { return execSync(`git ${args}`, { cwd: ROOT, stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); } catch { return ""; } };
  const dirty = new Set(git(`status --porcelain -- ${CODE_DIR} ${OUT_DIR}`).split("\n").filter(Boolean).map((l) => l.slice(3).trim()));
  const commitTime = (rel) => Number(git(`log -1 --format=%ct -- ${rel}`) || 0);
  const out = {};
  for (const f of fs.readdirSync(path.join(ROOT, CODE_DIR)).filter((f) => /^t\d\d\w*\.py$/.test(f)).sort()) {
    const name = f.replace(/\.py$/, ""), code = `${CODE_DIR}/${f}`, log = `${OUT_DIR}/${name}.txt`;
    if (!fs.existsSync(path.join(ROOT, log))) { out[name] = { fresh: false, reason: "no output" }; warn(`tutorial ${name}: no executed output`); continue; }
    const head = fs.readFileSync(path.join(ROOT, log), "utf8").split("\n", 1)[0];
    const recorded = (head.match(/\| script sha256 ([0-9a-f]{16})\s*$/) ?? [])[1] ?? null;
    if (recorded) {
      const now = createHash("sha256").update(fs.readFileSync(path.join(ROOT, code))).digest("hex").slice(0, 16);
      const fresh = recorded === now;
      out[name] = { fresh, by: "script hash", reason: fresh ? null : "the script changed after its output was produced" };
      if (!fresh) warn(`tutorial ${name}: the output was produced by another version of the script (sha256 ${recorded}, now ${now}): the site shows the code without output until it is re-executed`);
      continue;
    }
    const useMtime = dirty.has(code) || dirty.has(log) || [...dirty].some((d) => d.endsWith("/") && code.startsWith(d));
    const tc = useMtime ? fs.statSync(path.join(ROOT, code)).mtimeMs / 1000 : commitTime(code);
    const to = useMtime ? fs.statSync(path.join(ROOT, log)).mtimeMs / 1000 : commitTime(log);
    const fresh = to + 1 >= tc;
    out[name] = { fresh, by: useMtime ? "mtime" : "commit", reason: fresh ? null : "the script changed after its output was produced" };
    warn(`tutorial ${name}: the output records no script hash (written before it was recorded): freshness from ${useMtime ? "file times" : "commit times"}`);
    if (!fresh) warn(`tutorial ${name}: output older than the script (${useMtime ? "file times" : "commits"}): the site shows the code without output until it is re-executed`);
  }
  write("tutorial-freshness.json", { generated, tutorials: out });
  // Sanitised copies of the outputs, the only ones the site imports (lib/snippet.ts): stdout only (the
  // "[stderr]" part holds warnings with absolute paths of the machine that ran them), and no line that
  // names a local path. The client bundle must never carry machine paths.
  const SAN = path.join(OUT, "outputs");
  fs.rmSync(SAN, { recursive: true, force: true });
  fs.mkdirSync(SAN, { recursive: true });
  for (const f of fs.readdirSync(path.join(ROOT, OUT_DIR)).filter((f) => /^t\d\d\w*\.txt$/.test(f))) {
    let text = fs.readFileSync(path.join(ROOT, OUT_DIR, f), "utf8");
    const cut = text.search(/^\[stderr\]\s*$/m);
    if (cut >= 0) text = text.slice(0, cut);
    text = text.split("\n").filter((l) => !/(\/home\/|scratch|\/store\/|\/tmp\/)/.test(l)).join("\n").replace(/\s+$/, "") + "\n";
    fs.writeFileSync(path.join(SAN, f), text);
  }
}
genTutorialFreshness();

// ---------------------------------------------------------------------------------------------
// Getting started (src/pages/Tutorial.tsx): the page of the site is the documentation page
// docs/site/tutorials/first_steps.md, converted here (external review 0.1.0, finding 80: the site and
// the docs told the getting-started path twice and the two had drifted apart). The MyST subset of that
// page is read strictly: "##" headings, paragraphs, numbered lists, $$ display math $$, ```bash /
// ```python fences, {literalinclude} of a region of a tutorial script ("# [name]") or of lines of its
// output (:start-at: / :end-at:), {image} of a script's figure; inline `code`, **bold**, $math$,
// {doc}`page` and [label](link). Anything else stops the build, naming the line, so the page cannot
// say something the documentation does not.
// ---------------------------------------------------------------------------------------------
function genFirstSteps() {
  const REL = "docs/site/tutorials/first_steps.md";
  const md = read(REL);
  const lines = md.split("\n");
  const where = (k) => `${REL}:${k + 1}`;
  const docHref = (target) => {                       // relative to tutorials/, as Sphinx resolves it
    const parts = ["tutorials", ...target.split("/")], out = [];
    for (const p of parts) { if (p === "..") out.pop(); else out.push(p); }
    return out.join("/") + ".html";
  };
  const inline = (s, k) => {
    const out = [];
    const re = /`([^`]+)`|\*\*([^*]+)\*\*|\$([^$]+)\$|\{doc\}`([^`<]*?)(?:\s*<([^>]+)>)?`|\[([^\]]+)\]\(([^)]+)\)/g;
    let last = 0, m;
    while ((m = re.exec(s))) {
      if (m.index > last) out.push({ k: "t", v: s.slice(last, m.index) });
      if (m[1] !== undefined) out.push({ k: "code", v: m[1] });
      else if (m[2] !== undefined) out.push({ k: "b", v: m[2] });
      else if (m[3] !== undefined) out.push({ k: "m", v: texToRich(`$${m[3]}$`) });
      else if (m[4] !== undefined) {
        const target = (m[5] ?? m[4]).trim();
        const page = read(`docs/site/tutorials/${target}.md`, { optional: true }) ?? read(`docs/site/${target.replace(/^\.\.\//, "")}.md`, { optional: true });
        if (page == null) fail(`${where(k)}: {doc}\`${target}\` names no page`);
        const label = m[5] ? m[4].trim() : (page.match(/^#\s+(?:\d+\.\s*)?(.+)$/m)?.[1] ?? target);
        out.push({ k: "doc", to: docHref(target), v: label });
      } else {
        const href = m[7], label = m[6].replace(/`/g, "");
        if (/^https?:/.test(href)) out.push({ k: "a", href, v: label });
        else if (href.startsWith("code/")) out.push({ k: "repo", path: `docs/site/tutorials/${href}`, v: label });
        else fail(`${where(k)}: link ${href}: only https links and code/ files are converted`);
      }
      last = re.lastIndex;
    }
    if (last < s.length) out.push({ k: "t", v: s.slice(last) });
    for (const x of out) if (x.k === "t" && /[{}]|\*|`/.test(x.v)) fail(`${where(k)}: unsupported inline markup in ${JSON.stringify(x.v)}`);
    return out;
  };
  const slugOf = (t) => t.toLowerCase().replace(/[^a-z0-9 -]/g, "").trim().replace(/\s+/g, "-");   // MyST heading anchors
  const blocks = [];
  let title = null, k = 0;
  const directive = (name) => {                     // ```{name} arg + :opt: value lines, up to ```
    const arg = lines[k].replace(/^```\{[\w-]+\}\s*/, "").trim();
    const opts = {};
    k++;
    while (k < lines.length && lines[k].startsWith(":")) {
      const o = lines[k].match(/^:([\w-]+):\s*(.*)$/);
      opts[o[1]] = o[2].replace(/^"(.*)"$/, "$1");
      k++;
    }
    if (lines[k]?.trim() !== "```") fail(`${where(k)}: {${name}} with a body is not converted`);
    k++;
    return { arg, opts };
  };
  while (k < lines.length) {
    const l = lines[k];
    if (!l.trim()) { k++; continue; }
    let m;
    if ((m = l.match(/^# (.+)$/))) { title = m[1].trim(); k++; continue; }
    if ((m = l.match(/^## (.+)$/))) { blocks.push({ t: "h2", id: slugOf(m[1]), text: m[1].trim() }); k++; continue; }
    if (l.startsWith("```{literalinclude}")) {
      const { arg, opts } = directive("literalinclude");
      const c = arg.match(/^code\/(t\d\d\w*)\.py$/), o = arg.match(/^outputs\/(t\d\d\w*)\.txt$/);
      if (c) {
        const region = (opts["start-after"] ?? "").match(/^# \[([\w-]+)\]$/)?.[1];
        if (!region || opts["end-before"] !== `# [/${region}]`) fail(`${where(k)}: a script is included by one region "# [name]" … "# [/name]"`);
        blocks.push({ t: "code", script: c[1], region });
      } else if (o) blocks.push({ t: "out", script: o[1], startAt: opts["start-at"] ?? null, endAt: opts["end-at"] ?? null });
      else fail(`${where(k)}: literalinclude of ${arg} is not converted`);
      continue;
    }
    if (l.startsWith("```{image}")) {
      const { arg, opts } = directive("image");
      const im = arg.match(/^outputs\/(t\d\d\w*\.png)$/);
      if (!im || !opts.alt) fail(`${where(k)}: an image is a script's figure (outputs/<name>.png) with an :alt:`);
      blocks.push({ t: "img", file: im[1], alt: opts.alt });
      continue;
    }
    if ((m = l.match(/^```(bash|python)\s*$/))) {
      const body = [];
      for (k++; k < lines.length && lines[k] !== "```"; k++) body.push(lines[k]);
      k++;
      blocks.push({ t: "fence", lang: m[1], text: body.join("\n") });
      continue;
    }
    if (l.startsWith("```") || l.startsWith(":::")) fail(`${where(k)}: block ${l.trim()} is not converted`);
    if (l.trim() === "$$") {
      const body = [];
      for (k++; k < lines.length && lines[k].trim() !== "$$"; k++) body.push(lines[k]);
      k++;
      blocks.push({ t: "math", rich: texToRich(`$${body.join(" ")}$`) });
      continue;
    }
    if (/^\d+\.\s/.test(l)) {
      const items = [];
      while (k < lines.length && lines[k].trim() && (/^\d+\.\s/.test(lines[k]) || /^\s+\S/.test(lines[k]))) {
        if (/^\d+\.\s/.test(lines[k])) items.push({ at: k, text: lines[k].replace(/^\d+\.\s+/, "") });
        else items[items.length - 1].text += " " + lines[k].trim();
        k++;
      }
      blocks.push({ t: "ol", items: items.map((it) => inline(it.text, it.at)) });
      continue;
    }
    if (/^([#>|<-]|\*(?!\*))/.test(l)) fail(`${where(k)}: line not converted: ${l.slice(0, 40)}`);
    const at = k, para = [];
    while (k < lines.length && lines[k].trim() && !/^(```|##|\$\$|\d+\.\s)/.test(lines[k])) para.push(lines[k++].trim());
    blocks.push({ t: "p", inl: inline(para.join(" "), at) });
  }
  if (!title) fail(`${REL}: no "# Title"`);
  write("first-steps.json", { generated, source: REL, title, blocks });
  console.log(`gen-data: getting started: ${blocks.length} blocks from ${REL}`);
}
genFirstSteps();
