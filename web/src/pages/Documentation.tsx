// Documentation page: the map of the reference documentation (the Sphinx site served under /docs/, built
// with `bash web/build.sh --with-docs`). The user-guide parts and their pages are read from the Sphinx
// sources at build time (scripts/gen-data.mjs → generated/docs-map.json), so the map follows the
// documentation as it is written. Learning (the Tutorial) and reference (this page) are kept apart.
// Without the Sphinx build, entries are listed without a link, never as dead links.
import { Link } from "react-router";
import { PageHead, Section } from "../components/ui";
import { withBase } from "../lib/base";
import { SITE } from "../content/site";
import docsMap from "../generated/docs-map.json";
import { GETTING_STARTED } from "../content/nav";
import "../styles/docs.css";

type Entry = { title: string; text: string; to?: string; docs?: string; file?: string };
type MapEntry = { title: string; text: string; href: string };
const MAP = docsMap as { guide: { caption: string; entries: MapEntry[] }[]; dynamics: MapEntry[] };

// Start here: the getting-started route of content/nav.json (the same four steps as the Home page and the
// documentation home), then the tutorials index.
const START: Entry[] = [
  ...GETTING_STARTED.map((s, i) => ({ title: `${i + 1}. ${s.title}`, text: s.text, to: s.to, docs: s.docs })),
  { title: "Tutorials", to: "/tutorials", text: "All the tutorials, by topic, each a script that ends on a published value or a closed form." },
];

const REFERENCE: Entry[] = [
  { title: "API reference", docs: "api/index.html", text: "The public objects, from their docstrings: the user layer, then the expert layer." },
  { title: "Mathematics", docs: "math/index.html", text: "The methods behind Conatus: discretisation, time stepping, eigenproblems, orbits, continuation." },
];

function EntryCard({ e }: { e: Entry }) {
  const inner = <><h3>{e.title}</h3><p>{e.text}</p></>;
  if (e.to) return <Link className="doc-entry is-link" to={e.to}>{inner}</Link>;
  if (e.file) return <a className="doc-entry is-link" href={withBase(e.file)}>{inner}</a>;
  if (e.docs && SITE.docs) return <a className="doc-entry is-link" href={withBase("/docs/" + e.docs)}>{inner}</a>;
  return <div className="doc-entry">{inner}</div>;
}

const asEntry = (m: MapEntry): Entry => ({ title: m.title, text: m.text, docs: m.href });

export default function Documentation() {
  // The dynamics part of the guide is one index page in the toctree: list its pages instead.
  const parts = MAP.guide.map((p) => ({
    caption: p.caption,
    entries: p.entries.flatMap((e) => e.href === "guide/dynamics/index.html" && MAP.dynamics.length ? MAP.dynamics : [e]).map(asEntry),
  }));
  return (
    <>
      <PageHead eyebrow="Documentation" title="Documentation">
        <p>Find task guides, API documentation and the numerical methods reference. To learn, start with the four steps
          below.</p>
      </PageHead>
      <Section id="start" title="Start here" className="doc-sec">
        <div className="doc-grid">{START.map((e) => <EntryCard key={e.title} e={e} />)}</div>
      </Section>
      {parts.map((p, i) => (
        <Section key={p.caption} id={"guide-" + (i + 1)} eyebrow="User guide" title={p.caption} className="doc-sec">
          <div className="doc-grid">{p.entries.map((e) => <EntryCard key={e.title + e.docs} e={e} />)}</div>
        </Section>
      ))}
      <Section id="reference" title="Reference" className="doc-sec">
        <div className="doc-grid">{REFERENCE.map((e) => <EntryCard key={e.title} e={e} />)}</div>
      </Section>
    </>
  );
}
