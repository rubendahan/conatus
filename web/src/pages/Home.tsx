// Home page, built around one research task (presentation review of 0.1.1, priorities 1, 3, 6, 12, 14, 15;
// lot F): what Conatus is for, one complete stability calculation that runs as shown next to what it
// printed, a short list of what else it does, and one route to get started. Optional tools, plans and the
// origin of the name are on the About page.
//
// The example is the whole file web/home/critical_reynolds.py and its output is web/home/critical_reynolds.out,
// written by web/home/run.sh when the script runs (header: sha256 of the script, Conatus and Python versions).
// tests/docs/test_home_example.py checks that the output belongs to the script shown and that the README
// shows the same code and output.
import { Link } from "react-router-dom";
import { ButtonLink, Code, DocLink, RepoPath } from "../components/ui";
import { SITE } from "../content/site";
import Film from "../components/Film";
import { films } from "../data/films";
import { GETTING_STARTED } from "../content/nav";
import "../styles/home.css";
import "../styles/hero.css";
import "../styles/films.css";
import "../styles/replay.css";

const RAW = import.meta.glob<string>("../../home/critical_reynolds.{py,out}", { query: "?raw", import: "default", eager: true });
const raw = (ext: string): string => {
  const hit = Object.entries(RAW).find(([k]) => k.endsWith("." + ext));
  if (!hit) throw new Error(`[home] web/home/critical_reynolds.${ext} is missing (bash web/home/run.sh writes the output)`);
  return hit[1];
};
const CODE = raw("py").replace(/\s+$/, "");
const OUT_LINES = raw("out").replace(/\s+$/, "").split("\n");
// header: "# critical_reynolds.py | script sha256 … | conatus 0.1.2 | python 3.12.11 | 2026-10-21 | 41 s"
const HEAD: RegExpMatchArray = OUT_LINES[0].match(/^# critical_reynolds\.py \| script sha256 [0-9a-f]{64} \| conatus (\S+) \| python (\S+) \| (\S+) \| (\d+) s$/)
  ?? (() => { throw new Error("[home] unexpected header in web/home/critical_reynolds.out (re-run bash web/home/run.sh)"); })();
const OUTPUT = OUT_LINES.slice(1).join("\n");

// The film beside the title: a solver-produced field, small and pauseable (review: "Homepage motion").
const film = films.find((f) => f.slug === "kelvin-helmholtz-film") ?? films[0];

// What else it does: one line each, each pointing at a page that shows it.
const ALSO: { what: string; to: string; doc?: boolean }[] = [
  { what: "Time-dependent simulations of the same kind of model, with implicit–explicit time stepping.", to: "/tutorials/getting-started" },
  { what: "Channels, periodic boxes, disks, annuli, pipes, spheres, balls and spherical shells.", to: "/geometries" },
  { what: "Adjoint modes, transient growth, and Floquet analysis of time-periodic base flows.", to: "/dynamics" },
  { what: "Steady states, travelling waves and periodic orbits by Newton–Krylov; continuation through folds and bifurcations.", to: "/dynamics" },
  { what: "Lyapunov exponents, covariant Lyapunov vectors, tracers and dye in a simulated flow.", to: "/capabilities" },
];

export default function Home() {
  return (
    <>
      <section className="hero" aria-labelledby="hero-title">
        <div className="wrap hero-grid">
          <div className="hero-content">
            <p className="hero-eyebrow">Conatus</p>
            <h1 id="hero-title" className="hero-title">Write a PDE model once, then simulate it, compute its stability and find its invariant solutions from the same object.</h1>
            <p className="hero-lede">Conatus is experimental research software for fluid instability and dynamical systems.</p>
            <div className="row hero-cta">
              <ButtonLink to="#example">See a first calculation</ButtonLink>
              <ButtonLink to="/tutorials/getting-started" variant="ghost">Getting started</ButtonLink>
            </div>
            <p className="hero-status">
              Version {SITE.version}: CPU wheels for Linux x86-64 (<code>pip install conatus</code>); CUDA support is
              experimental. The interface may still change.{SITE.repo && <> <a href={SITE.repo}>Source on GitHub</a>.</>}
            </p>
          </div>
          {film && <div className="hero-film"><Film f={film} /></div>}
        </div>
      </section>

      <section id="example" className="section home-code" aria-labelledby="code-h">
        <div className="wrap">
          <div className="section-head">
            <p className="eyebrow">A first calculation</p>
            <h2 id="code-h">When does flow between two plates become unstable?</h2>
            <p className="section-intro">Small wavy disturbances of plane Poiseuille flow obey the Orr–Sommerfeld equation.
              The script below writes that equation, computes the least stable mode at one Reynolds number, then searches
              for the critical Reynolds number. Save it as a file and run it with <code>python</code> after{" "}
              <code>pip install conatus</code>.</p>
          </div>
          <div className="code-grid">
            <Code title="critical_reynolds.py">{CODE}</Code>
            <figure className="replay">
              <figcaption className="replay-head">
                <span className="replay-label">Output</span>
                <span>what this script printed (Conatus {HEAD[1]}, Python {HEAD[2]}, {HEAD[4]} s on a CPU)</span>
              </figcaption>
              <pre tabIndex={0}><code>{OUTPUT}</code></pre>
            </figure>
            <div className="code-notes">
              <p><strong>Reference.</strong> Orszag (1971) gives c = 0.23752649 + 0.00373967i at Re = 10 000, α = 1, and
                Re<sub>c</sub> = 5772.22 at α<sub>c</sub> = 1.02056; the script's 1.02055 differs in the last digit.</p>
              <p><strong>What the checks mean.</strong> The residual is the backward error of the computed eigenpair.
                With <code>drift=96</code>, every eigenvalue is computed again with 96 Chebyshev polynomials
                instead of 64. Eigenvalues that change between the two resolutions are not converged by this test
                and are set aside.</p>
              <p><strong>Next.</strong> The <DocLink to="tutorials/01_orr_sommerfeld.html">stability tutorial</DocLink>{" "}
                continues from this script: the eigenfunction, its spectral coefficients and the sensitivity of c to Re.
                The file is <RepoPath path="web/home/critical_reynolds.py">web/home/critical_reynolds.py</RepoPath>.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="section does" aria-labelledby="does-h">
        <div className="wrap">
          <div className="section-head">
            <p className="eyebrow">What else it does</p>
            <h2 id="does-h">The same model, other questions</h2>
          </div>
          <ul className="also-list">
            {ALSO.map((a) => (
              <li key={a.what}><Link to={a.to}>{a.what}</Link></li>
            ))}
          </ul>
          <p className="more-link"><Link to="/capabilities">Capabilities, with what is experimental <span aria-hidden="true">→</span></Link></p>
        </div>
      </section>

      <section className="section start" aria-labelledby="start-h">
        <div className="wrap">
          <div className="section-head">
            <p className="eyebrow">Get started</p>
            <h2 id="start-h">Four steps</h2>
          </div>
          <ol className="route">
            {GETTING_STARTED.map((s) => (
              <li key={s.title}>
                {s.docs ? <DocLink to={s.docs}><strong>{s.title}</strong></DocLink> : <Link to={s.to!}><strong>{s.title}</strong></Link>}
                <span>{s.text}</span>
              </li>
            ))}
          </ol>
          <p className="home-about">
            <Link to="/documentation">Documentation</Link> · <Link to="/gallery">Gallery</Link> ·{" "}
            <Link to="/validation">Validation</Link> · <Link to="/citing">Citing</Link> ·{" "}
            <Link to="/about">About: maintenance, optional tools and plans</Link>
          </p>
        </div>
      </section>
    </>
  );
}
