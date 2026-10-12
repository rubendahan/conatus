// Dynamics: what Conatus lets you ask of the equations beyond simulating them, in plain words, each
// topic pointing to the tutorial that does it and to the user guide. Text only (Ruben, 2026-10-02:
// text-first pages; the former scroll scene is gone).
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { DocLink, PageHead, Section } from "../components/ui";
import { tutorials } from "../data";
import "../styles/dynamics.css";

const TOPICS: { id: string; title: string; body: ReactNode; tut: number[] }[] = [
  { id: "eigen", title: "Where a flow becomes unstable", tut: [1, 8],
    body: <p>Linearise the equations you already wrote about a base state and ask for their eigenmodes, with left
      eigenvectors and the residual of each. Follow the least stable one as a parameter changes and find the critical
      point where it starts to grow. Modes that do not survive a finer resolution are removed, and you are told.</p> },
  { id: "symmetry", title: "Symmetry subspaces", tut: [3],
    body: <p>Declare a reflection or a rotation and solve in the symmetric subspace: the same eigenvalues for a fraction
      of the cost, after a check that the equations respect the symmetry.</p> },
  { id: "bifurcations", title: "Bifurcations, followed", tut: [4, 10],
    body: <p>Follow a branch of steady solutions as a parameter moves, through folds, by pseudo-arclength continuation.
      Pitchforks and Hopf points are detected and located on the way; switch branches, follow the cycles born at a Hopf
      point, continue a fold or a Hopf point in two parameters, and read the normal form that says what happens next.</p> },
  { id: "orbits", title: "Equilibria, travelling waves and periodic orbits", tut: [5],
    body: <p>Find the invariant solutions that organise a turbulent or chaotic flow, from a guess or from the state of a
      running simulation, by Newton–Krylov with a hookstep, and compute their stability. Each solution is reported with its
      residual.</p> },
  { id: "floquet", title: "Time-periodic flows: Floquet analysis", tut: [],
    body: <p>When the base flow oscillates, as in a pulsating channel or pipe, stability is a question
      about periodic motion. Conatus answers it with a Floquet–Hill method (Fourier in time, spectral in space), for any
      time-periodic linear problem written in the same language: give the base flow as a function of time and its period,
      and ask for its Floquet exponents or its optimal growth.</p> },
  { id: "nonmodal", title: "Transient growth", tut: [],
    body: <p>Stable flows can still amplify a disturbance enormously before it decays. Find the optimal perturbation and its
      growth, about a steady or a time-periodic base flow, in the
      energy norm you declare.</p> },
  { id: "chaos", title: "Chaos: Lyapunov exponents and ensembles", tut: [12],
    body: <p>Lyapunov spectra, covariant vectors, the Kaplan–Yorke dimension, Poincaré maps and basins of attraction, from
      ensembles of trajectories integrated together.</p> },
  { id: "adjoint", title: "Adjoints, sensitivity and data", tut: [13],
    body: <p>The physical adjoint of an operator in the inner product you declare (never a bare transpose), direct and
      adjoint global modes, the wavemaker and base-flow sensitivity; the exact adjoint of the time stepper for gradients,
      in a channel or a spherical shell, with the nonlinear optimal perturbation that grows most at a given energy.
      DMD and SPOD of data in the same norm.</p> },
];

export default function Dynamics() {
  const byN = new Map(tutorials.tutorials.map((t) => [t.n, t]));
  return (
    <>
      <PageHead eyebrow="Dynamics" title="The dynamical system behind the equations">
        <p>The equations you write for a simulation also define a dynamical system. Conatus lets you ask it the questions that
          matter: where it becomes unstable, how its solutions branch, which orbits organise it, how it answers a push. Same
          equations, no problem-specific code.</p>
      </PageHead>
      <nav className="wrap toc-inline" aria-label="On this page">
        {TOPICS.map((t) => <a key={t.id} href={`#${t.id}`}>{t.title}</a>)}
      </nav>
      <Section id="topics" className="dyn-topics">
        <div className="dyn-list">
          {TOPICS.map((t) => (
            <article key={t.id} id={t.id} className="dyn-topic">
              <h2>{t.title}</h2>
              {t.body}
              <p className="dyn-tut">{t.tut.map((n, i) => {
                const tt = byN.get(n);
                if (!tt) return null;
                return <span key={n}>{i > 0 && " · "}Tutorial {n}: <DocLink to={`tutorials/${tt.slug}.html`}>{tt.title}</DocLink></span>;
              })}</p>
            </article>
          ))}
        </div>
        <p className="more-link"><DocLink to="guide/index.html">The user guide describes each method in detail →</DocLink></p>
        <p className="more-link"><Link to="/tutorials">All tutorials →</Link></p>
      </Section>
    </>
  );
}
