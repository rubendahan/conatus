// The executed output of a tutorial next to its code, as printed (static: no replay effect). When the script changed after its last execution
// (lib/snippet.ts marks that output stale), nothing is shown rather than an output the code above did
// not produce; gen-data warns at build time.
import type { ReactNode } from "react";
import { outputHeader, pick, type Src } from "../lib/snippet";
import "../styles/replay.css";

export default function Executed({ out, patterns, page, label, note }: {
  out: Src | null; patterns: RegExp[]; page: string; label?: string; note?: ReactNode;
}) {
  if (!out) return null;                                 // stale output: show the code alone
  outputHeader(out);                                     // refuses an output that did not pass
  const rows = pick(out, patterns, page).replace(/\s+$/, "").split("\n");
  return (
    <figure className="replay">
      <figcaption className="replay-head"><span className="replay-label">{label ?? "Output"}</span></figcaption>
      <pre tabIndex={0}><code>{rows.map((l, i) => (
        <span key={i} className={/^OK:|\[PASS\]/.test(l.trim()) ? "replay-ok" : undefined}>{l}{"\n"}</span>
      ))}</code></pre>
      {note && <div className="replay-note">{note}</div>}
    </figure>
  );
}
