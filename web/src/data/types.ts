// Types of the generated data (src/generated/*.json, written by scripts/gen-data.mjs from the
// repository). Keep in sync with the generator: it is the only writer of these files.

/** Plain text and "rich" text: rich strings keep sub/superscripts that have no Unicode form as
 *  `_{…}` / `^{…}` (rendered by <Rich>); plain strings flatten them ("Re_c"). */
export interface RichText { test: string; value: string; reference: string }

export type Section = "Verification" | "Gallery" | "Performance";

export interface LedgerEntry {
  id: string;
  domain: string;
  test: string;
  value: string;
  reference: string;
  rich: RichText;
  /** First bibliography key cited by the row (paper/refs.bib), if any. */
  referenceKey: string | null;
  /** Deviation as printed ("3.5×10⁻¹²", "8 digits") or null when the row states none. */
  deviation: string | null;
  deviationLog10: number | null;
  /** Where a deviation not printed in the row comes from (e.g. a scorecard metric). */
  deviationNote: string | null;
  runIds: string[];
  /** "paper/conatus.tex:229" */
  source: string;
  section: Section;
  /** Gallery rows: example script and "Lines" column. */
  script: string | null;
  lines: string | null;
  /** Paper order after grouping by domain. */
  order: number;
}

export interface Domain { id: string; label: string; count: number }

export interface Counts {
  tests: number; runs: number; decisions: number; errors: number;
  scorecardPass: number; scorecardFail: number; scorecardNotMeasured: number; date: string;
}

export interface Ledger {
  generated: string;
  commit: string | null;
  sources: string[];
  counts: Counts;
  scorecard: string;
  domains: Domain[];
  entries: LedgerEntry[];
}

// ---- performance (scripts/gen-data.mjs, genPerf: the paper's Performance section) ---------------
export interface Rb2dScheme { a100Ms: number; a100Speedup: number; l40sSpeedup: number; dedalusMs: number | null }
export interface Rb2dRow { grid: string; nx: number; nz: number; RK443: Rb2dScheme; SBDF2: Rb2dScheme; met: boolean }
export interface WeakPoint { label: string; gpus: number; nodes: number; efficiency: number }
export interface Divergence { what: string; tex: string; json: number | string; jsonSource: string; texSource: string; kept: string }

export interface Perf {
  available: boolean;     // false when gen-data could not read the paper's Performance prose
  generated: string;
  sources: string[];
  rb2d: {
    target: number; worst: { factor: number; scheme: string; grid: string }; dedalusVersion: string; dedalusRanks: number; rows: Rb2dRow[];
    verdictRun: string | null; texSource: string; runIds: string[];
  };
  multigpu: { goal: number; points: WeakPoint[]; modesPerGpu: string; texSource: string; runIds: string[] };
  divergences: Divergence[];
}

// ---- tutorials, citation, stats --------------------------------------------------------------
export interface TutorialScript {
  script: string; status: string; time: string; run: string; commit: string | null; date: string | null; runId: string | null;
}
export interface Tutorial {
  n: number; slug: string; title: string; learn: string; reference: string | null;
  status: "pass" | "partial" | "fail" | "not-run"; scripts: TutorialScript[];
}
export interface Tutorials { generated: string; source: string; statusSource: string | null; tutorials: Tutorial[] }

export interface Citation {
  cff: Record<string, unknown> & { title: string; version?: string; authors: { "family-names": string; "given-names": string }[] };
  bibtex: { software: string; paper: string | null };
  source: string;
}

export interface Stats {
  generated: string; commit: string | null; counts: Counts;
  ledger: { entries: number; verification: number; gallery: number; performance: number; domains: number; withRunId: number; withDeviation: number };
  runsArchived: number;
  tutorials: { total: number; passing: number };
  warnings: number; divergences: number;
}
