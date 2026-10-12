// Typed access to the generated data. The JSON files are written by scripts/gen-data.mjs at every
// build (web/build.sh runs it before the typecheck), so they always exist when this module compiles.
import ledgerJson from "../generated/ledger.json";
import perfJson from "../generated/perf.json";
import tutorialsJson from "../generated/tutorials.json";
import citationJson from "../generated/citation.json";
import statsJson from "../generated/stats.json";
import type { Citation, Ledger, Perf, Stats, Tutorials } from "./types";

export type * from "./types";

export const ledger = ledgerJson as unknown as Ledger;
export const perf = perfJson as unknown as Perf;
export const tutorials = tutorialsJson as unknown as Tutorials;
export const citation = citationJson as unknown as Citation;
export const stats = statsJson as unknown as Stats;

/** "2026-09-28T21:04:30Z" → "28 September 2026" (UTC, identical on server and client). */
export function formatDate(iso: string): string {
  const d = new Date(iso);
  const months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  return `${d.getUTCDate()} ${months[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** Fixed-precision number, never locale-dependent (server and client render the same text). */
export function fmt(x: number, digits = 1): string {
  return x.toFixed(digits);
}
