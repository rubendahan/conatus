// Code and output shown on the site are cut from the executed tutorials (docs/site/tutorials/code/*.py
// and outputs/*.txt, run by docs/site/run_tutorials.py), never retyped. Every cut is strict: when a
// tutorial is rewritten and a marker, a region or an output line disappears, the build stops with a
// message naming the file, the marker and the page to update, instead of showing stale or empty code.

const CODE = import.meta.glob<string>("../../../docs/site/tutorials/code/*.py", { query: "?raw", import: "default", eager: true });
// Outputs are imported from the sanitised copies written by scripts/gen-data.mjs (stdout only, no local paths).
const OUT = import.meta.glob<string>("../generated/outputs/*.txt", { query: "?raw", import: "default", eager: true });

// Written by scripts/gen-data.mjs: is each output newer than its script (see genTutorialFreshness)?
import freshness from "../generated/tutorial-freshness.json";
const FRESH = (freshness as { tutorials: Record<string, { fresh: boolean }> }).tutorials;

/** A source file and its repository path (for error messages and captions). */
export interface Src { path: string; text: string }

function fail(src: Src, what: string, page?: string): never {
  throw new Error(
    `[web/snippet] ${src.path}: ${what}.\n` +
    `  The site cuts this from the executed tutorial; the tutorial changed. Update the cut` +
    (page ? ` in ${page}` : "") + ` (web/src/lib/snippet.ts explains the markers).`,
  );
}

/** The executed tutorial `name` (e.g. "t01_orr_sommerfeld"): its code and its output log. The output is
 *  null when it is older than the script (not yet re-executed): pages then show the code alone. */
export function tutorial(name: string): { code: Src; out: Src | null } {
  const c = Object.entries(CODE).find(([k]) => k.endsWith(`/code/${name}.py`));
  const o = Object.entries(OUT).find(([k]) => k.endsWith(`/outputs/${name}.txt`));
  if (!c) throw new Error(`[web/snippet] docs/site/tutorials/code/${name}.py not found (renamed or removed tutorial?)`);
  const fresh = !!o && FRESH[name]?.fresh === true;
  return {
    code: { path: `docs/site/tutorials/code/${name}.py`, text: c[1] },
    out: fresh ? { path: `docs/site/tutorials/outputs/${name}.txt`, text: o![1] } : null,
  };
}

/** Lines of a tutorial marked "# [name]" … "# [/name]" (the markers themselves are dropped). */
export function region(src: Src, name: string, page?: string): string {
  const lines = src.text.split("\n");
  const i = lines.findIndex((l) => l.trim() === `# [${name}]`);
  const j = lines.findIndex((l, k) => k > i && l.trim() === `# [/${name}]`);
  if (i < 0 || j < 0) fail(src, `region "# [${name}]" … "# [/${name}]" not found`, page);
  const body = lines.slice(i + 1, j).filter((l) => !/^\s*# \[\/?[\w-]+\]\s*$/.test(l));
  const out = body.join("\n").replace(/^\n+|\s+$/g, "");
  if (!out) fail(src, `region "${name}" is empty`, page);
  return out;
}

/** Lines from the first containing `from` to the first later one containing `to` (inclusive). */
export function between(src: Src, from: string, to: string, opts: { includeTo?: boolean; page?: string } = {}): string {
  const lines = src.text.split("\n");
  const i = lines.findIndex((l) => l.includes(from));
  const j = lines.findIndex((l, k) => k > i && l.includes(to));
  if (i < 0) fail(src, `start marker ${JSON.stringify(from)} not found`, opts.page);
  if (j < 0) fail(src, `end marker ${JSON.stringify(to)} not found after ${JSON.stringify(from)}`, opts.page);
  return lines.slice(i, opts.includeTo === false ? j : j + 1)
    .filter((l) => !/^\s*# \[\/?[\w-]+\]\s*$/.test(l)).join("\n").replace(/\s+$/, "");
}

/** Lines matching any of the patterns, in order. Every pattern must match at least one line. */
export function pick(src: Src, patterns: RegExp[], page?: string): string {
  const lines = src.text.split("\n");
  for (const p of patterns) if (!lines.some((l) => p.test(l))) fail(src, `no line matches ${p}`, page);
  return lines.filter((l) => patterns.some((p) => p.test(l))).join("\n");
}

/** Header of an executed output: "# t05_ks_orbits: PASS (exit 0) in 61.4 s | run_id … | commit … | host | date
 *  | script sha256 …" (the public export keeps "… in 61.4 s | date | script sha256 …"). */
export function outputHeader(src: Src): { script: string; status: string; seconds: number; runId: string; date: string } {
  const m = src.text.match(/^# (\S+): (\w+) \(exit \d+\) in ([\d.]+) s(?: \| run_id (\S+) \| commit \S+ \| \S+)? \| (\S+)/);
  if (!m) fail(src, "unexpected output header (expected '# name: PASS (exit 0) in … s | run_id …')");
  if (m[2] !== "PASS") fail(src, `the tutorial did not pass (status ${m[2]}): the site shows only passing executions`);
  return { script: m[1], status: m[2], seconds: Number(m[3]), runId: m[4] ?? "", date: m[5] };
}
