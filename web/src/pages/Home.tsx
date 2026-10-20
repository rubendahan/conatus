// Home page: a thesis over a live simulation (a real computation), two short films of Conatus
// runs, then the code of an executed tutorial with its output, what Conatus lets you do, and the way in.
// Text-first, like a project home page (Ruben, 2026-10-02). No measured figures here: the numbers
// live on the Validation page only. Code and output are cut from the executed tutorial
// (lib/snippet.ts), never retyped.
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import FluidHero from "../sim/FluidHero";
import { ButtonLink, Code } from "../components/ui";
import { SITE } from "../content/site";
import { region, tutorial } from "../lib/snippet";
import Executed from "../components/Executed";
import Film from "../components/Film";
import { films } from "../data/films";
import "../styles/home.css";
import "../styles/hero.css";
import "../styles/films.css";

const PAGE = "web/src/pages/Home.tsx";

// Tutorial 1, as executed: the problem, the solve, and what it printed.
const t01 = tutorial("t01_orr_sommerfeld");
const CODE = region(t01.code, "problem", PAGE).replace(/\nprint\([^\n]*explain\(\)\)[^\n]*$/, "") + "\n\n" + region(t01.code, "solve", PAGE);
const OUTPUT = [/^c = /, /^Re_c = /, /\[PASS\] (right residual|spectral tail)/, /^OK:/];

// ---- what you can do ---------------------------------------------------------------------------------
const DOES: { k: string; title: string; text: ReactNode }[] = [
  { k: "Write", title: "Equations as on paper", text: "Declare the domain, the fields and the equations in their usual form; name the boundaries and say what holds there. Conatus places the rest and tells you what it did." },
  { k: "Simulate", title: "On a CPU or a GPU", text: "The same model runs on your laptop or on a GPU with one keyword. Resolution, scheme and time step are always yours to choose." },
  { k: "Analyse", title: "Where a flow becomes unstable", text: "Eigenvalues with their residuals, critical parameters, exact adjoints, transient growth and Floquet analysis of time-periodic flows." },
  { k: "Discover", title: "Orbits and bifurcations", text: "Equilibria, travelling waves and periodic orbits by Newton–Krylov; branches followed through folds, pitchforks and Hopf points." },
  { k: "Keep track", title: "Runs you choose to record", text: "Ask for it (record=True, or cn.runs.track) and the parameters, code version and results of a run are kept, so you can find it, compare it and reproduce it later." },
];

export default function Home() {
  return (
    <>
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-sim">
          <FluidHero interactive label="Kelvin–Helmholtz illustration, computed in your browser by a WebGL grid scheme, not by the Conatus solver" />
        </div>
        <div className="hero-veil" aria-hidden="true" />
        <div className="wrap hero-content">
          <p className="hero-eyebrow">Spectral PDEs · Dynamical systems · GPU</p>
          <h1 id="hero-title" className="hero-title">Write the equations.<br />Conatus finds the structure.</h1>
          <p className="hero-lede">
            A spectral framework for partial differential equations and the dynamical systems they define, native on GPUs.
            Write the equations as on paper, simulate them, find where the flow becomes unstable, follow its orbits and
            bifurcations.
          </p>
          <div className="row hero-cta">
            <ButtonLink to="/tutorials/getting-started">Get started</ButtonLink>
            <ButtonLink to="/documentation" variant="ghost">Documentation</ButtonLink>
          </div>
          {/* first screen (W1): which version, its status, and how to get the code */}
          <p className="hero-status">
            Version {SITE.version} · CPU supported, GPU in beta · {SITE.repo
              ? <a href={SITE.repo}>source on GitHub</a>
              : <>repository private during the preview, source on request</>}
            {" "}· install: <code>pip install conatus</code> (<Link to="/install">Install</Link>)
          </p>
        </div>
        <p className="hero-caption wrap">
          A Kelvin–Helmholtz shear layer rolling up: an illustration computed in your browser by a simple WebGL grid
          scheme, not by the Conatus spectral solver. The films below are Conatus runs.
        </p>
      </section>

      {/* two short films of real runs (web/gallery/films.py), each frame a state the solver computed */}
      {films.length > 0 && (
        <section className="section home-films" aria-labelledby="films-h">
          <div className="wrap">
            <div className="section-head">
              <p className="eyebrow">In motion</p>
              <h2 id="films-h">Computed with Conatus</h2>
              <p className="section-intro">Each film is a Conatus run, frame by frame: a shear layer rolling up, and heavy
                fluid falling into light fluid.</p>
            </div>
            <ul className="films">
              {films.map((f) => <li key={f.slug}><Film f={f} /></li>)}
            </ul>
            <p className="more-link"><Link to="/gallery">The gallery <span aria-hidden="true">→</span></Link></p>
          </div>
        </section>
      )}

      <section className="section lab home-code" aria-labelledby="code-h">
        <div className="wrap">
          <div className="section-head">
            <p className="eyebrow">Write</p>
            <h2 id="code-h">The equation on the page is the equation that runs</h2>
            <p className="section-intro">Conatus reads the problem as mathematics, compiles it for the CPU or the GPU, and
              tells you exactly what it did. Nothing changes behind your back. Below, the first tutorial as it ran, and what
              it printed.</p>
          </div>
          <div className="code-grid">
            <Code title={`${t01.code.path.split("/").pop()} (excerpt)`}>{CODE}</Code>
            <div className="code-side">
              <Executed out={t01.out} patterns={OUTPUT} page={PAGE} label="Output" />
              <ol className="code-steps">
                <li><strong>Declare</strong> the domain, the fields and the equations, as written on paper.</li>
                <li><strong>Discretise and check.</strong> A rank report refuses a missing or extra boundary condition before anything runs.</li>
                <li><strong>Solve, and see how far to trust it.</strong> Every result comes with its own checks.</li>
              </ol>
            </div>
          </div>
        </div>
      </section>

      <section className="section does" aria-labelledby="does-h">
        <div className="wrap">
          <div className="section-head">
            <p className="eyebrow">What it does</p>
            <h2 id="does-h">From the equations to the structure of the flow</h2>
          </div>
          <ol className="does-grid">
            {DOES.map((d) => (
              <li key={d.k} className="does-item">
                <p className="does-k">{d.k}</p>
                <h3>{d.title}</h3>
                <p>{d.text}</p>
              </li>
            ))}
          </ol>
          <p className="more-link"><Link to="/capabilities">All capabilities <span aria-hidden="true">→</span></Link></p>
        </div>
      </section>


      <section className="section start" aria-labelledby="start-h">
        <div className="wrap">
          <div className="section-head">
            <p className="eyebrow">Get started</p>
            <h2 id="start-h">Install, learn, read, cite</h2>
            <p className="section-intro">Install it, work through the tutorials, then find each part described once in the
              documentation.</p>
          </div>
          <div className="start-grid">
            <Link className="start-card" to="/install"><span className="start-k">Install</span><strong><code>pip install conatus</code></strong><span>CPU wheel with the compiled core, no compiler needed; a CUDA 12 wheel for NVIDIA GPUs (beta). <code>python -m conatus doctor</code> says what this machine can run.</span></Link>
            <Link className="start-card" to="/tutorials"><span className="start-k">Learn</span><strong>Tutorials</strong><span>Executed tutorials, each ending on a published result: start with 0, getting started.</span></Link>
            <Link className="start-card" to="/documentation"><span className="start-k">Read</span><strong>Documentation</strong><span>The user guide, the API reference and the mathematics behind each method.</span></Link>
            <Link className="start-card" to="/citing"><span className="start-k">Cite</span><strong>Citing</strong><span>CITATION.cff and BibTeX. Licence: BSD-3-Clause.</span></Link>
            <Link className="start-card" to="/coming-next"><span className="start-k">Next</span><strong>Coming next</strong><span>Multi-GPU runs, the Studio from anywhere and further stability tools: under development, not in this release.</span></Link>
          </div>
        </div>
      </section>
    </>
  );
}
