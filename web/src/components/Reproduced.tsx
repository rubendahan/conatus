// The public scripts that print a Validation row's number (src/content/reproduced.json; external
// review of the live 0.1.2 site, chain of evidence): script and recorded output pinned to a release
// tag, the command and the text it prints. The version is read from the header of the recorded output
// ("... | conatus V | ..."); an output without that header (the tutorials) is pinned to the tag of
// the site's version and names no version. Checked by tests/docs/test_validation_sources.py and by
// scripts/gen-data.mjs. Used by the Validation page only.
import data from "../content/reproduced.json";
import { RepoPath } from "./ui";

export interface Reproduction { script: string; command: string; output: string; printed: string[]; numbers: string[][]; note?: string }

const ROWS = data.rows as Record<string, Reproduction[]>;
/** The release the site is built for (python/conatus/__init__.py, read by vite.config.ts). */
export const REPRODUCED_VERSION: string = __CN_VERSION__;
const HEADED = import.meta.glob<string>(["../../../web/home/*.out", "../../../examples/published/*/output.txt"],
  { query: "?raw", import: "default", eager: true });
/** The version that wrote a recorded output: "conatus V" in its header line; null without that header. */
export function recordedVersion(output: string): string | null {
  const head = (HEADED[`../../../${output}`] ?? "").split("\n", 1)[0];
  return / \| conatus ([^ |]+) \| /.exec(head)?.[1] ?? null;
}
export const reproductionsOf = (id: string): Reproduction[] => ROWS[id] ?? [];

export function Reproductions({ items }: { items: Reproduction[] }) {
  return (
    <>
      {items.map((r) => { const version = recordedVersion(r.output), tag = `v${version ?? REPRODUCED_VERSION}`; return (
        <p key={r.script + r.command}>
          <RepoPath path={r.script} at={tag}>{r.script}</RepoPath> ({version ? `Conatus ${version}, ` : ""}tag {tag}).
          Run <code>{r.command}</code> from the repository root; it prints{" "}
          {r.printed.map((t, k) => <span key={t}>{k ? " and " : ""}<code>{t}</code></span>)}, as recorded in{" "}
          <RepoPath path={r.output} at={tag}>{r.output}</RepoPath>.{r.note ? ` ${r.note}` : ""}
        </p>
      ); })}
    </>
  );
}
