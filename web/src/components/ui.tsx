// Shared presentational components: page and section heads, figure plates, statistics, source
// references and code blocks.
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { SITE } from "../content/site";
import { withBase } from "../lib/base";

/** Stable anchor from a heading text. */
export function slugify(s: string): string {
  return s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

/** Page title and lede. Text only (Ruben, 2026-10-02: text-first pages, at most one visual per page,
 *  and only where it explains something). */
export function PageHead({ eyebrow, title, children }: { eyebrow?: string; title: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head wrap">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <h1>{title}</h1>
      {children && <div className="lede">{children}</div>}
    </header>
  );
}

export function Section({ id, eyebrow, title, intro, children, className }: {
  id?: string; eyebrow?: string; title?: ReactNode; intro?: ReactNode; children?: ReactNode; className?: string;
}) {
  const anchor = id ?? (typeof title === "string" ? slugify(title) : undefined);
  return (
    <section className={"section" + (className ? " " + className : "")} aria-labelledby={anchor ? anchor + "-h" : undefined}>
      <div className="wrap">
        {(title || eyebrow || intro) && (
          <div className="section-head">
            {eyebrow && <p className="eyebrow">{eyebrow}</p>}
            {title && <h2 id={anchor ? anchor + "-h" : undefined}><span id={anchor} className="anchor-target" />{title}</h2>}
            {intro && <div className="section-intro">{intro}</div>}
          </div>
        )}
        {children}
      </div>
    </section>
  );
}

/** Dark photographic plate (world B) with a numbered caption. */
export function Plate({ fig, caption, children, className }: { fig?: string | number; caption?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <figure className={"plate" + (className ? " " + className : "")}>
      <div className="plate-body">{children}</div>
      {caption && <figcaption className="plate-caption">{fig !== undefined && <span className="fig-num">Fig. {fig}.</span>} {caption}</figcaption>}
    </figure>
  );
}

export function Stat({ value, label, source }: { value: ReactNode; label: ReactNode; source?: ReactNode }) {
  return (
    <div className="stat">
      <p className="stat-value num">{value}</p>
      <p className="stat-label">{label}</p>
      {source && <p className="stat-source">{source}</p>}
    </div>
  );
}

/** A repository path; becomes a link once the repository is public (REPO_URL at build time).
 *  `at`: a tag or branch to pin the link to (a release tag such as "v0.1.2"); default main. */
export function RepoPath({ path, children, at = "main" }: { path: string; children?: ReactNode; at?: string }) {
  const label = children ?? path;
  if (SITE.repo) return <a className="repo-path" href={`${SITE.repo}/blob/${at}/${path}`}><code>{label}</code></a>;
  return <code className="repo-path" title="Path in the Conatus repository">{label}</code>;
}

/** A run identifier: every cited number points to one (results/runs/<run_id>/). */
export function RunId({ id }: { id: string }) {
  if (SITE.repo) return <a className="run-id" href={`${SITE.repo}/tree/main/results/runs/${id}`}>{id}</a>;
  return <code className="run-id" title={`results/runs/${id}/`}>{id}</code>;
}

/** A page of the Sphinx documentation (served under /docs/ by `web/build.sh --with-docs`). Without
 *  that build the label is shown as plain text, never as a dead link. */
export function DocLink({ to, children }: { to: string; children: ReactNode }) {
  if (!SITE.docs) return <span className="doc-link-off">{children}</span>;
  return <a href={withBase("/docs/" + to)}>{children}</a>;
}

export function ButtonLink({ to, children, variant = "primary" }: { to: string; children: ReactNode; variant?: "primary" | "ghost" }) {
  const cls = "btn btn-" + variant;
  if (/^(https?:|mailto:)/.test(to) || to.startsWith("/docs/")) return <a className={cls} href={to}>{children}</a>;
  return <Link className={cls} to={to}>{children}</Link>;
}

// ---- code blocks with a tiny, dependency-free Python highlighter ---------------------------------
const PY_KW = new Set(["from", "import", "as", "def", "return", "for", "in", "if", "else", "elif", "while", "with", "print", "lambda", "True", "False", "None", "and", "or", "not", "class", "assert"]);
function highlightPython(src: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\b\d+(?:\.\d+)?(?:e[-+]?\d+)?\b)|([A-Za-z_][A-Za-z_0-9]*)|(\s+)|(.)/g;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(src))) {
    const [tok, com, str, num, id] = m;
    if (com) out.push(<span key={k++} className="tk-com">{tok}</span>);
    else if (str) out.push(<span key={k++} className="tk-str">{tok}</span>);
    else if (num) out.push(<span key={k++} className="tk-num">{tok}</span>);
    else if (id && PY_KW.has(id)) out.push(<span key={k++} className="tk-kw">{tok}</span>);
    else out.push(tok);
  }
  return out;
}

export function Code({ children, lang = "python", title }: { children: string; lang?: "python" | "bash" | "text" | "bibtex"; title?: string }) {
  const body = lang === "python" ? highlightPython(children) : lang === "bash" ? children.split("\n").map((l, i) => (
    <span key={i}>{l.startsWith("#") ? <span className="tk-com">{l}</span> : l.startsWith("$ ") ? <><span className="tk-prompt">$ </span>{l.slice(2)}</> : l}{"\n"}</span>
  )) : children;
  const copy = <CopyButton text={copyText(children, lang)} label="Copy" />;
  return (
    <figure className="code">
      {title ? <div className="code-head"><figcaption className="code-title">{title}</figcaption>{copy}</div> : copy}
      <pre tabIndex={0}><code className={"lang-" + lang}>{body}</code></pre>
    </figure>
  );
}

/** What the copy button of a code block puts on the clipboard: the code as shown, except shell
 *  sessions, where only the commands are copied (the "$ " prompts removed, output lines and comment
 *  lines left out), so the result pastes into a terminal as is. A line after a command is kept only
 *  as its continuation (the kept line before it ends with "\"): indented output, such as doctor's
 *  "  [ok     ] ..." lines, is never copied. */
export function copyText(src: string, lang: string): string {
  if (lang !== "bash" || !/^\$ /m.test(src)) return src;
  const out: string[] = [];
  let cont = false;
  for (const l of src.split("\n")) {
    if (l.startsWith("$ ")) out.push(l.slice(2));
    else if (cont) out.push(l);
    else continue;
    cont = /\\\s*$/.test(out[out.length - 1]);
  }
  return out.join("\n");
}

/** Small "copy" affordance for citation blocks (progressive: hidden until hydrated). */
export function CopyButton({ text, label = "Copy" }: { text: string; label?: string }) {
  return (
    // aria-live: the "Copied" / "Select and copy" outcome is announced by screen readers too
    <button type="button" className="copy-btn" aria-live="polite" onClick={async (e) => {
      const b = e.currentTarget;
      try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; } catch { b.textContent = "Select and copy"; }
      setTimeout(() => { b.textContent = label; }, 1600);
    }}>{label}</button>
  );
}
