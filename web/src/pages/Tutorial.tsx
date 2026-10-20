// Getting started: the ten-minute entry (one equation, one plot, one physical number, four guided
// changes). The page is the documentation page docs/site/tutorials/first_steps.md, converted at build
// time by scripts/gen-data.mjs into src/generated/first-steps.json: the site and the documentation
// tell the path once, from one source (external review 0.1.0, findings 62 and 80). Every code block is
// a region of the executed script docs/site/tutorials/code/t00_first_steps.py and every output line
// comes from its executed log (lib/snippet.ts: strict cuts, an output older than the script is never
// shown). The longer path through one problem is tutorial 0 of the documentation.
import type { ReactNode } from "react";
import { Code, DocLink, PageHead, RepoPath } from "../components/ui";
import { Rich } from "../components/Rich";
import { region, tutorial, type Src } from "../lib/snippet";
import firstSteps from "../generated/first-steps.json";
import "../styles/tutorial.css";
import "../styles/replay.css";

const PAGE = "web/src/pages/Tutorial.tsx";

type Inl = { k: "t" | "code" | "b" | "m" | "doc" | "a" | "repo"; v: string; to?: string; href?: string; path?: string };
type Block =
  | { t: "h2"; id: string; text: string }
  | { t: "p"; inl: Inl[] }
  | { t: "ol"; items: Inl[][] }
  | { t: "math"; rich: string }
  | { t: "fence"; lang: "bash" | "python"; text: string }
  | { t: "code"; script: string; region: string }
  | { t: "out"; script: string; startAt: string | null; endAt: string | null }
  | { t: "img"; file: string; alt: string };
const DOC = firstSteps as unknown as { title: string; source: string; blocks: Block[] };

// The figures written by the scripts (synced next to their outputs); shown only with a fresh output.
const FIGS = import.meta.glob<string>("../../../docs/site/tutorials/outputs/*.png", { query: "?url", import: "default", eager: true });

function Inline({ parts }: { parts: Inl[] }) {
  return <>{parts.map((x, i) => {
    switch (x.k) {
      case "code": return <code key={i}>{x.v}</code>;
      case "b": return <strong key={i}>{x.v}</strong>;
      case "m": return <span key={i} className="tu-math"><Rich text={x.v} /></span>;
      case "doc": return <DocLink key={i} to={x.to!}>{x.v}</DocLink>;
      case "a": return <a key={i} href={x.href}>{x.v}</a>;
      case "repo": return <RepoPath key={i} path={x.path!}>{x.v}</RepoPath>;
      default: return <span key={i}>{x.v}</span>;
    }
  })}</>;
}

/** Lines of an executed output from the first containing `from` to the first later one containing `to`
 *  (the :start-at: / :end-at: of the documentation page), or the whole output. */
function Output({ out, from, to }: { out: Src; from: string | null; to: string | null }) {
  const lines = out.text.replace(/\s+$/, "").split("\n").slice(1);          // the header line is not output
  const i = from ? lines.findIndex((l) => l.includes(from)) : 0;
  const j = to ? lines.findIndex((l, k) => k >= i && l.includes(to)) : lines.length - 1;
  if (i < 0 || j < 0) throw new Error(`[web/snippet] ${out.path}: lines "${from}" … "${to}" not found (${PAGE}, ${DOC.source})`);
  return (
    <figure className="replay">
      <figcaption className="replay-head"><span className="replay-label">Output</span></figcaption>
      <pre tabIndex={0}><code>{lines.slice(i, j + 1).map((l, k) => (
        <span key={k} className={/^OK:/.test(l.trim()) ? "replay-ok" : undefined}>{l}{"\n"}</span>
      ))}</code></pre>
    </figure>
  );
}

function Body({ b }: { b: Block }): ReactNode {
  switch (b.t) {
    case "p": return <p><Inline parts={b.inl} /></p>;
    case "ol": return <ol className="tu-steps">{b.items.map((it, i) => <li key={i}><Inline parts={it} /></li>)}</ol>;
    case "math": return <div className="tu-eqs" role="math"><p><Rich text={b.rich} /></p></div>;
    case "fence": return <Code lang={b.lang}>{b.lang === "bash" ? b.text.split("\n").map((l) => "$ " + l).join("\n") : b.text}</Code>;
    case "code": return <Code title={`${b.script}.py`}>{region(tutorial(b.script).code, b.region, PAGE)}</Code>;
    case "out": {
      const out = tutorial(b.script).out;
      return out ? <Output out={out} from={b.startAt} to={b.endAt} /> : null;     // stale: the code alone
    }
    case "img": {
      const fresh = tutorial(b.file.replace(/_[^_]+\.png$/, "")).out;
      const src = Object.entries(FIGS).find(([k]) => k.endsWith("/" + b.file))?.[1];
      return fresh && src ? <figure className="tu-fig tu-fig-plot"><img src={src} loading="lazy" alt={b.alt} /></figure> : null;
    }
    default: return null;
  }
}

// The page in sections, one per "##" heading of the source.
const SECTIONS: { id: string; title: string; blocks: Block[] }[] = [];
const LEAD: Block[] = [];
for (const b of DOC.blocks) {
  if (b.t === "h2") SECTIONS.push({ id: b.id, title: b.text, blocks: [] });
  else (SECTIONS.length ? SECTIONS[SECTIONS.length - 1].blocks : LEAD).push(b);
}

export default function Tutorial() {
  return (
    <>
      <PageHead eyebrow="Getting started" title={DOC.title}>
        {LEAD.filter((b) => b.t === "p").slice(0, 1).map((b, i) => <Body key={i} b={b} />)}
      </PageHead>
      <div className="wrap tu-layout">
        <nav className="tu-toc" aria-label="On this page">
          <p className="tu-toc-t">On this page</p>
          <ol>{SECTIONS.map((s) => <li key={s.id}><a href={`#${s.id}`}>{s.title}</a></li>)}</ol>
        </nav>
        <article className="tu-body">
          {LEAD.slice(1).map((b, i) => <Body key={i} b={b} />)}
          {SECTIONS.map((s, n) => (
            <section key={s.id} className="tu-part" id={s.id} aria-labelledby={s.id + "-h"}>
              <h2 id={s.id + "-h"}><span className="tu-n">{n + 1}.</span> {s.title}</h2>
              {s.blocks.map((b, i) => <Body key={i} b={b} />)}
            </section>
          ))}
          <p className="tu-source">This page is the documentation page{" "}
            <DocLink to="tutorials/first_steps.html">Getting started in ten minutes</DocLink>; its script is{" "}
            <RepoPath path="docs/site/tutorials/code/t00_first_steps.py">t00_first_steps.py</RepoPath>.</p>
        </article>
      </div>
    </>
  );
}
