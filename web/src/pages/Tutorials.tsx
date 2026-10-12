// Tutorials: the one index of the tutorials (W1: "Tutorial", "Lessons" and "Tutorials"
// were three names for one thing). Tutorial 0 (getting started) is a page of this site; tutorials
// 1 to 14 are pages of the documentation. Each is a short script that ends on a published result it
// reproduces. Titles and references come from docs/site/tutorials (generated/tutorials.json), the
// same source as the documentation's own tutorials index, so the two lists never differ.
import { Link } from "react-router-dom";
import { DocLink, PageHead, Section } from "../components/ui";
import { tutorials } from "../data";
import "../styles/tutorials.css";

export default function Tutorials() {
  const list = tutorials.tutorials;
  return (
    <>
      <PageHead eyebrow="Tutorials" title="Learn by reproducing a published result">
        <p>Tutorials from a first model to bifurcations, chaos and a GPU run. Each one is a plain Python
          script, explained piece by piece, that ends on a result from the literature; what you see on the page is what the
          script printed. New to Conatus? Start with tutorial 0, <Link to="/tutorials/getting-started">getting started</Link>.</p>
      </PageHead>
      <Section id="list" className="tut-section">
        <ol className="tut-list">
          {list.map((t) => (
            <li key={t.slug} className="tut">
              <span className="tut-n" aria-hidden="true">{String(t.n).padStart(2, "0")}</span>
              <div className="tut-body">
                <h2 className="tut-title">{t.n === 0
                  ? <Link to="/tutorials/getting-started">{t.title}</Link>
                  : <DocLink to={`tutorials/${t.slug}.html`}>{t.title}</DocLink>}</h2>
                {t.learn && <p className="tut-learn">{(t.learn.charAt(0).toUpperCase() + t.learn.slice(1)).replace(/\*/g, "")}.</p>}
                {t.reference && <p className="tut-ref">Checked against {t.reference}.</p>}
              </div>
            </li>
          ))}
        </ol>
      </Section>
      <Section id="run" title="Run them yourself" intro={<p>
        Once Conatus is <Link to="/install">installed</Link> with the <code>tutorials</code> extra (<code>pip install
        "conatus[tutorials]"</code>: matplotlib for the figures), each tutorial is a script you run with Python, on your laptop or on
        a cluster; tutorial 2b needs a CUDA device and the check script of tutorial 14 needs Dedalus. The same list, with when each script last ran, is
        the <DocLink to="tutorials/index.html">tutorials index of the documentation</DocLink>.</p>} />
    </>
  );
}
