// Validation: the one page of the site with numbers. The verification results of the release,
// grouped by topic: value obtained, reference, deviation; each row says where it comes from (the
// table of the Conatus paper in preparation, how its deviation was obtained) and, when there is one,
// the script of this repository that reproduces it. The run records are not published: the page
// labels the values as author-reported (external review 0.1.0, finding 70; review of 0.1.1, F24).
// Rows compare with published values only (cross-checks against other codes stay in the paper).
// Data: src/generated/ledger.json (scripts/gen-data.mjs).
// Self-contained: no other page imports anything from here, so it can be removed with its route.
import { PageHead, RepoPath } from "../components/ui";
import { Rich } from "../components/Rich";
import { ledger, type LedgerEntry } from "../data";
import "../styles/ledger.css";

// Topics of the page, from the ledger's domains (scripts/gen-data.mjs); an unknown domain lands in "Other".
const TOPICS: { id: string; title: string; lede: string; domains: string[] }[] = [
  { id: "stability", title: "Linear stability", lede: "Eigenvalues and critical parameters of classical flows, against published tables.",
    domains: ["stability", "convection", "shear", "bvp"] },
  { id: "time", title: "Time integration", lede: "Orders of the time steppers, measured and against the theory.",
    domains: ["time-integration", "turbulence"] },
  { id: "geometries", title: "Geometries and bases", lede: "The disk, annulus, sphere, shell and ball, the whole and half line, compound intervals and stress-free walls, against closed forms.",
    domains: ["geometry"] },
  { id: "dynamics", title: "Dynamics", lede: "Periodic orbits, bifurcations, Floquet exponents and adjoints, including the spherical shell.",
    domains: ["dynamics", "bifurcation", "floquet", "adjoint"] },
  { id: "other", title: "Patterns and other checks", lede: "Pattern-forming systems, operator compilers and symmetry subspaces.",
    domains: ["patterns", "operators", "other"] },
];
const known = new Set(TOPICS.flatMap((t) => t.domains));
const topicOf = (e: LedgerEntry) => TOPICS.find((t) => t.domains.includes(e.domain)) ?? TOPICS.find((t) => t.id === "other")!;
const rowsOf = (id: string) => ledger.entries.filter((e) => topicOf(e).id === id || (id === "other" && !known.has(e.domain)));

function Row({ e }: { e: LedgerEntry }) {
  return (
    <tr id={`t-${e.id}`}>
      <th scope="row" className="v-test">
        <Rich text={e.rich.test} />
        <details className="v-more">
          <summary>Source</summary>
          <p>
            {e.section} table of the Conatus paper (in preparation){e.deviationNote ? <>; deviation: {e.deviationNote}</> : null}
            {e.script && <> · reproduced by <RepoPath path={e.script}>{e.script}</RepoPath></>}
          </p>
        </details>
      </th>
      <td className="v-value" data-label="Value"><span><Rich text={e.rich.value} /></span></td>
      <td className="v-ref" data-label="Reference"><span>{e.reference.trim() && !/^[—–-]+$/.test(e.reference.trim()) ? <Rich text={e.rich.reference} /> : "—"}</span></td>
      <td className="v-dev num" data-label="Deviation">
        {e.deviation ? <span title={e.deviationNote ?? undefined}>{e.deviation}</span> : <span className="muted" aria-label="not stated">—</span>}
      </td>
    </tr>
  );
}

function Table({ rows, caption }: { rows: LedgerEntry[]; caption: string }) {
  return (
    <table className="v-table">
      <caption className="visually-hidden">{caption}: test, value obtained, reference and deviation.</caption>
      <thead><tr><th scope="col">Test</th><th scope="col">Value obtained</th><th scope="col">Reference</th><th scope="col" className="v-dev">Deviation</th></tr></thead>
      <tbody>{rows.map((e) => <Row key={e.id} e={e} />)}</tbody>
    </table>
  );
}

export default function Validation() {
  return (
    <>
      <PageHead eyebrow="Evidence" title="Validation">
        <p>The verification values of this release, in one place. Each value is compared with a published result or a closed
          form. The deviation is the difference between them.</p>
      </PageHead>
      <nav className="wrap v-topics" aria-label="Topics">
        <ul>{TOPICS.map((t) => <li key={t.id}><a href={`#${t.id}`}>{t.title} <span className="num">{rowsOf(t.id).length}</span></a></li>)}</ul>
      </nav>
      <div className="wrap v-page">
        {TOPICS.map((t) => {
          const rows = rowsOf(t.id);
          if (!rows.length) return null;
          return (
            <section key={t.id} className="v-group" aria-labelledby={`${t.id}-h`}>
              <header className="v-group-head">
                <h2 id={`${t.id}-h`}><span id={t.id} className="anchor-target" />{t.title}</h2>
                <p>{t.lede}</p>
              </header>
              <Table rows={rows} caption={t.title} />
            </section>
          );
        })}
        <section className="v-note" aria-labelledby="how-h">
          <h2 id="how-h">How these numbers are produced</h2>
          <p>The values are measurements reported by the author. They are copied at each build of this site from the
            Verification table of the Conatus paper (in preparation), for Conatus {ledger.version ?? ""}. A deviation is the
            relative error of the measurement, or the one deviation the row states. A dash means the row states the
            agreement in words. The run records (inputs, code revision, logs) are not published with this release. When a
            row names a script, that script reproduces the value from this repository.</p>
        </section>
      </div>
    </>
  );
}
