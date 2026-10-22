// The public scripts that print a Validation row's number (src/content/reproduced.json; external
// review of the live 0.1.2 site, chain of evidence): script and recorded output pinned to the release
// tag, the command and the text it prints. Checked by tests/docs/test_validation_sources.py and by
// scripts/gen-data.mjs. Used by the Validation page only.
import data from "../content/reproduced.json";
import { RepoPath } from "./ui";

export interface Reproduction { script: string; command: string; output: string; printed: string[]; numbers: string[][] }

const ROWS = data.rows as Record<string, Reproduction[]>;
export const REPRODUCED_VERSION: string = data.version;
export const reproductionsOf = (id: string): Reproduction[] => ROWS[id] ?? [];

export function Reproductions({ items }: { items: Reproduction[] }) {
  const tag = `v${REPRODUCED_VERSION}`;
  return (
    <>
      {items.map((r) => (
        <p key={r.script + r.command}>
          <RepoPath path={r.script} at={tag}>{r.script}</RepoPath> (Conatus {REPRODUCED_VERSION}, tag {tag}).
          Run <code>{r.command}</code> from the repository root; it prints{" "}
          {r.printed.map((t, k) => <span key={t}>{k ? " and " : ""}<code>{t}</code></span>)}, as recorded in{" "}
          <RepoPath path={r.output} at={tag}>{r.output}</RepoPath>.
        </p>
      ))}
    </>
  );
}
