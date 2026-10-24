// Tutorials: the one index of the tutorials (W1: "Tutorial", "Lessons" and "Tutorials" were three
// names for one thing). Titles, groups and references come from docs/site/tutorials/index.md
// (generated/tutorials.json), the same source as the documentation's own index, so the two lists never
// differ. Tutorials are shown by topic and title, without course numbers (presentation review of 0.1.1,
// priority 9: the numbered sequence had gaps). The first-steps tutorial is a page of this site; the others
// are pages of the documentation.
import { Link } from "react-router";
import { DocLink, PageHead, Section } from "../components/ui";
import { tutorials } from "../data";
import type { Tutorial } from "../data/types";
import "../styles/tutorials.css";

function Title({ t }: { t: Tutorial }) {
  if (t.slug === "first_steps") return <Link to="/tutorials/getting-started">{t.title}</Link>;
  return <DocLink to={`tutorials/${t.slug}.html`}>{t.title}</DocLink>;
}

export default function Tutorials() {
  const list = tutorials.tutorials;
  const groups = [...new Set(list.map((t) => t.group ?? "Tutorials"))];
  return (
    <>
      <PageHead eyebrow="Tutorials" title="Tutorials">
        <p>Each tutorial is a plain Python script, explained piece by piece, that ends by comparing its result with a
          published value or a closed form; what you see on the page is what the script printed. New to Conatus? Follow
          the three tutorials of <a href="#start-here">Start here</a> in order.</p>
      </PageHead>
      {groups.map((g) => (
        <Section key={g} id={g.toLowerCase().replace(/[^a-z]+/g, "-")} title={g} className="tut-section">
          <ol className={"tut-list" + (g === "Start here" ? " tut-route" : "")}>
            {list.filter((t) => (t.group ?? "Tutorials") === g).map((t) => (
              <li key={t.slug} className="tut">
                <div className="tut-body">
                  <h3 className="tut-title"><Title t={t} /></h3>
                  {t.learn && <p className="tut-learn">{(t.learn.charAt(0).toUpperCase() + t.learn.slice(1)).replace(/\*/g, "")}.</p>}
                  {t.reference && <p className="tut-ref">Checked against {t.reference}.</p>}
                </div>
              </li>
            ))}
          </ol>
        </Section>
      ))}
      <Section id="run" title="Run them yourself" intro={<p>
        Once Conatus is <Link to="/install">installed</Link> with the <code>tutorials</code> extra (<code>pip install
        "conatus[tutorials]"</code>: matplotlib for the figures, xarray for the tracers tutorial), each tutorial is a script you run
        with Python, on your laptop or on a cluster. The GPU part of the Rayleigh–Bénard tutorial needs a CUDA device. The
        same list, with when each script last ran, is the <DocLink to="tutorials/index.html">tutorials index of the
        documentation</DocLink>.</p>} />
    </>
  );
}
