// Published cases: results of published papers computed again with Conatus. For each case: the
// reference, what is computed, a figure with the paper's values (rings) over Conatus's (discs or
// lines), the gap and its definition, and how to reproduce it (script, command, recorded output,
// run time and CPU), pinned to a release tag.
// Data: src/content/published.json (text, and the numbers shown, each with the line of the recorded
// output that prints it) and, per case, examples/published/<case>/result.json (the figure) and
// output.txt (the header gives the version, run time and CPU). Checked by
// tests/docs/test_published_cases.py. Self-contained: removing the route removes the page.
import { type ReactNode } from "react";
import { Code, PageHead, PageLink, RepoPath } from "../components/ui";
import { ChartFigure, Legend, type SeriesDef } from "../components/charts/core";
import { XYPlot, type XYSeries } from "../components/charts/XYPlot";
import data from "../content/published.json";
import "../styles/charts.css";
import "../styles/published.css";

interface Fact { label: string; value: string; printed: string }
interface Case {
  slug: string; dir: string; title: string; reference: string; doi: string; source: string; shows: string;
  computed: string; gap: string; facts: Fact[]; note?: string; validation: string[];
}

/* eslint-disable @typescript-eslint/no-explicit-any */
const RESULTS = import.meta.glob<any>("../../../examples/published/*/result.json", { import: "default", eager: true });
const OUTPUTS = import.meta.glob<string>("../../../examples/published/*/output.txt", { query: "?raw", import: "default", eager: true });
const CASES = data.cases as Case[];
const resultOf = (c: Case) => RESULTS[`../../../${c.dir}/result.json`];

/** The header line of a recorded output: "# <case>/run.py | script sha256 … | conatus V | python P | date | T s | CPU | n threads". */
function header(c: Case) {
  const head = (OUTPUTS[`../../../${c.dir}/output.txt`] ?? "").split("\n", 1)[0].replace(/^#\s*/, "").split(" | ");
  return { conatus: head[2]?.replace("conatus ", "") ?? "?", date: head[4] ?? "?", seconds: head[5] ?? "?", cpu: head[6] ?? "?", threads: head[7] ?? "?" };
}

const CONATUS: SeriesDef = { id: "c", label: "Conatus", cls: "s1" };
const PAPER: SeriesDef = { id: "p", label: "published", cls: "s2" };
const f = (x: number, d = 6) => x.toFixed(d);

function DataTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <table className="data">
      <thead><tr>{head.map((h) => <th key={h} className="r">{h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className="r num">{v}</td>)}</tr>)}</tbody>
    </table>
  );
}

// ---- one figure per case ---------------------------------------------------------------------
function OrrSommerfeld({ r, c }: { r: any; c: Case }) {
  const series: XYSeries[] = [
    { id: "c", label: "Conatus, every resolved mode", cls: "s1", points: r.spectrum, r: 3, quiet: true },
    { id: "p", label: "Orszag (1971)", cls: "s2", points: r.points.map((p: any) => p.ref), r: 5 },
  ];
  return (
    <ChartFigure id={`${c.slug}-fig`} title="Eigenvalues c in the complex plane, Re = 10⁴, α = 1"
      subtitle="A mode grows when Im c > 0. Rings: the 32 eigenvalues Orszag lists; discs: every mode Conatus resolves."
      legend={<Legend series={[{ ...CONATUS, label: "Conatus" }, { ...PAPER, label: "Orszag (1971)" }]} />}
      source={c.source}
      data={<DataTable head={["n", "Orszag c_r", "Orszag c_i", "Conatus c_r", "Conatus c_i", "gap"]}
        rows={r.points.map((p: any) => [p.n, p.ref_text[0], p.ref_text[1], f(p.conatus[0], 10), f(p.conatus[1], 10), p.gap.toExponential(1)])} />}>
      <XYPlot series={series} xLabel="c_r" yLabel="c_i" xDomain={[0.15, 1.0]} yDomain={[-0.7, 0.05]}
        ariaLabel="Orr–Sommerfeld eigenvalues of plane Poiseuille flow at Re = 10000 and alpha = 1: Orszag's 32 values lie on Conatus's spectrum." />
    </ChartFigure>
  );
}

function RayleighBenard({ r, c }: { r: any; c: Case }) {
  const k: number[] = r.k;
  const w = r.walls;
  const line = (id: string, label: string, cls: XYSeries["cls"]): XYSeries =>
    ({ id, label, cls, line: true, points: k.map((x, i) => [x, w[id].Ra[i]] as [number, number]) });
  const lines: XYSeries[] = [line("rigid-rigid", "Conatus, rigid–rigid", "s1"), line("rigid-free", "Conatus, rigid–free", "s3"), line("free-free", "Conatus, free–free", "s5")];
  const closed = k.filter((_, i) => i % 3 === 0).map((x) => [x, (x * x + Math.PI ** 2) ** 3 / (x * x)] as [number, number]);
  const crit = ["rigid-rigid", "rigid-free"].map((id) => [w[id].ref[1], w[id].ref[0]] as [number, number]);   // ref = [Ra_c, k_c]
  const refs: XYSeries = { id: "ref", label: "published", cls: "s2", r: 5, points: [...closed, ...crit] };
  const lineSeries: SeriesDef[] = lines.map((s) => ({ id: s.id, label: s.label, cls: s.cls }));
  return (
    <ChartFigure id={`${c.slug}-fig`} title="Marginal curves Ra(k) of the layer heated from below"
      subtitle="Rings: Rayleigh's closed form for two stress-free walls, and the critical points Chandrasekhar gives for rigid walls. Lines: the neutral Rayleigh number Conatus finds at 25 wavenumbers."
      legend={<><Legend kind="line" series={lineSeries} /><Legend series={[{ ...PAPER, label: "published" }]} /></>}
      source={c.source}
      data={<DataTable head={["k", "rigid–rigid", "rigid–free", "free–free", "closed form"]}
        rows={k.map((x, i) => [x.toFixed(2), f(w["rigid-rigid"].Ra[i], 4), f(w["rigid-free"].Ra[i], 4), f(w["free-free"].Ra[i], 4), f(w["free-free"].exact[i], 4)])} />}>
      <XYPlot series={[...lines, refs]} xLabel="wavenumber k" yLabel="Rayleigh number" yLog yDomain={[500, 20000]} xDomain={[1, 7]}
        ariaLabel="Marginal Rayleigh number against wavenumber for rigid-rigid, rigid-free and free-free walls; the published closed form and critical points lie on the curves." />
    </ChartFigure>
  );
}

function Pipe({ r, c }: { r: any; c: Case }) {
  const pick = (mode: string, key: "ref" | "conatus") =>
    r.rows.filter((x: any) => x.mode === mode).map((x: any) => [x.Re, -x[key][0]] as [number, number]);
  const series: XYSeries[] = [
    { id: "c0", label: "Conatus (1, 0)", cls: "s1", points: pick("(1,0)", "conatus") },
    { id: "cc", label: "Conatus (1, 1) centre", cls: "s3", points: pick("centre", "conatus") },
    { id: "cw", label: "Conatus (1, 1) wall", cls: "s4", points: pick("wall", "conatus") },
    { id: "p", label: "Meseguer & Trefethen", cls: "s2", r: 7, points: [...pick("(1,0)", "ref"), ...pick("centre", "ref"), ...pick("wall", "ref")] },
  ];
  return (
    <ChartFigure id={`${c.slug}-fig`} title="Decay rate of the least damped modes, Re = 10² to 10⁷"
      subtitle="−Re s of the rightmost eigenvalue of each branch, on logarithmic axes. Rings: Table 1 of Meseguer & Trefethen."
      legend={<Legend series={series.map((s) => ({ id: s.id, label: s.label, cls: s.cls }))} />}
      source={c.source}
      data={<DataTable head={["Re", "mode", "published (conj.)", "Conatus", "gap"]}
        rows={r.rows.map((x: any) => [x.Re.toExponential(0), x.mode, `${x.ref_text[0]} ${x.ref[1] < 0 ? "−" : "+"} ${x.ref_text[1]}i`,
          `${f(x.conatus[0], 12)} ${x.conatus[1] < 0 ? "−" : "+"} ${f(Math.abs(x.conatus[1]), 12)}i`, x.gap.toExponential(1)])} />}>
      <XYPlot series={series} xLabel="Reynolds number" yLabel="decay rate −Re s" xLog yLog
        ariaLabel="Decay rates of the rightmost pipe-flow eigenvalues against Reynolds number for three branches; Conatus and Meseguer and Trefethen coincide." />
    </ChartFigure>
  );
}

function KuramotoSivashinsky({ r, c }: { r: any; c: Case }) {
  const sols = Object.values(r.solutions) as any[];
  const series: XYSeries[] = [
    { id: "c", label: "Conatus", cls: "s1", r: 3, quiet: true, points: sols.flatMap((s) => s.all.map((z: number[]) => [z[0], Math.abs(z[1])] as [number, number])) },
    { id: "p", label: "Cvitanović et al. (2010)", cls: "s2", r: 5, points: sols.flatMap((s) => s.eigenvalues.map((e: any) => e.ref as [number, number])) },
  ];
  return (
    <ChartFigure id={`${c.slug}-fig`} title="Leading stability eigenvalues of E1, E2, E3, TW1 and TW2, L = 22"
      subtitle="μ ± iν, plotted with ν ≥ 0. Rings: Table 5.1 of Cvitanović, Davidchack & Siminos; discs: the eigenvalues Conatus computes."
      legend={<Legend series={[CONATUS, { ...PAPER, label: "Cvitanović et al. (2010)" }]} />}
      source={c.source}
      data={<DataTable head={["solution", "published μ", "published ν", "Conatus μ", "Conatus ν", "gap"]}
        rows={Object.entries(r.solutions).flatMap(([name, s]: [string, any]) => s.eigenvalues.map((e: any) =>
          [name, e.ref[0].toFixed(4), e.ref[1].toFixed(4), f(e.conatus[0]), f(Math.abs(e.conatus[1])), e.gap.toExponential(1)]))} />}>
      <XYPlot series={series} xLabel="μ" yLabel="ν" xDomain={[-2.2, 0.4]} yDomain={[-0.05, 1.05]}
        ariaLabel="Leading eigenvalues of five invariant solutions of the Kuramoto–Sivashinsky equation; the published values lie on Conatus's eigenvalues." />
    </ChartFigure>
  );
}

const FIGURES: Record<string, (p: { r: any; c: Case }) => ReactNode> = {
  "orr-sommerfeld-spectrum": OrrSommerfeld, "rayleigh-benard-onset": RayleighBenard,
  "pipe-flow-spectrum": Pipe, "kuramoto-sivashinsky-l22": KuramotoSivashinsky,
};

function CaseSection({ c }: { c: Case }) {
  // the version and tag come from the header of the recorded output, not from a typed field
  const h = header(c), TAG = `v${h.conatus}`;
  const Fig = FIGURES[c.slug];
  return (
    <section className="pc-case" aria-labelledby={`${c.slug}-h`}>
      <header className="pc-head">
        <h2 id={`${c.slug}-h`}><span id={c.slug} className="anchor-target" />{c.title}</h2>
        <p className="pc-ref">{c.reference} <a href={`https://doi.org/${c.doi}`}>doi:{c.doi}</a></p>
      </header>
      <p className="pc-shows">{c.shows}</p>
      <p className="pc-what">{c.computed}</p>
      <div className="pc-fig">{Fig && <Fig r={resultOf(c)} c={c} />}</div>
      <div className="pc-cols">
        <div>
          <h3>Gap</h3>
          <p>{c.gap}</p>
          <dl className="pc-facts">
            {c.facts.map((x) => <div key={x.label}><dt>{x.label}</dt><dd className="num">{x.value}</dd></div>)}
          </dl>
          {c.note && <p className="pc-note">{c.note}</p>}
        </div>
        <div>
          <h3>Reproduce</h3>
          <p>Script <RepoPath path={`${c.dir}/run.py`} at={TAG}>{`${c.dir}/run.py`}</RepoPath>, from the repository root
            after <code>pip install conatus=={h.conatus}</code>:</p>
          <Code lang="bash">{`python ${c.dir}/run.py`}</Code>
          <p>The recorded output is <RepoPath path={`${c.dir}/output.txt`} at={TAG}>{`${c.dir}/output.txt`}</RepoPath>:
            Conatus {h.conatus}, {h.seconds.replace(" s", " s of wall time")} on {h.threads.replace(" threads", " threads of")} {h.cpu}, {h.date}.</p>
          {c.validation.length > 0 && <p>Rows of the <PageLink to={`/validation#t-${c.validation[0]}`}>Validation</PageLink> page that this case reproduces link here.</p>}
        </div>
      </div>
    </section>
  );
}

export default function Published() {
  return (
    <>
      <PageHead eyebrow="Evidence" title="Published cases">
        <p>Results of published papers, computed again with Conatus. Each case gives the reference, the paper's
          values next to Conatus's, the gap between them with its definition, and the script and command that
          produce the numbers on this page.</p>
        <p>The published values are read from tables of the papers (or, where stated, of a paper that
          reproduces them) and from closed forms; none is read off a plot.</p>
      </PageHead>
      <nav className="wrap pc-toc" aria-label="Cases">
        <ul>{CASES.map((c) => <li key={c.slug}><a href={`#${c.slug}`}>{c.title}</a></li>)}</ul>
      </nav>
      <div className="wrap pc-page">
        {CASES.map((c) => <CaseSection key={c.slug} c={c} />)}
      </div>
    </>
  );
}
