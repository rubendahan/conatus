// Renders the "rich" strings of the generated data: `_{…}` → <sub>, `^{…}` → <sup>; everything else
// is already Unicode text (converted from the paper's LaTeX by scripts/gen-data.mjs).
import type { ReactNode } from "react";

export function Rich({ text }: { text: string }) {
  const out: ReactNode[] = [];
  const re = /([_^])\{([^{}]*)\}/g;
  let last = 0, k = 0, m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(m[1] === "_" ? <sub key={k++}>{m[2]}</sub> : <sup key={k++}>{m[2]}</sup>);
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return <>{out}</>;
}
