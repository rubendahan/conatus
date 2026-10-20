// Capabilities: what Conatus lets you do, in short plain sentences, grouped by the way you work: write,
// discretise, run, analyse, keep track. No measured figures here: the numbers are on the
// Validation page, linked once at the end.
import { Link } from "react-router-dom";
import { DocLink, PageHead, Section } from "../components/ui";
import "../styles/capabilities.css";

// more: a route of the site, or "docs:<page>" for a page of the Sphinx documentation (DocLink)
interface Cap { title: string; text: string; more?: [string, string] }
interface Part { id: string; label: string; title: string; intro: string; caps: Cap[] }

const PARTS: Part[] = [
  {
    id: "language", label: "Write", title: "The language of equations",
    intro: "You write the problem the way it is written on paper. Conatus understands it as mathematics and explains what it built.",
    caps: [
      { title: "Equations as on paper", text: "Domains, fields, parameters and equations in their usual form; boundary conditions by the name of the boundary. What Conatus places for you, it lists." },
      { title: "Refused, not repaired", text: "A missing or extra boundary condition, an ill-posed split or an undeclared choice is refused with a message that says what, where and why, and proposes a fix it never applies itself." },
      { title: "Exact derivatives, for free", text: "Every expression knows its derivative and its adjoint, so linearisations, tangents and adjoints follow from the equations you wrote." },
    ],
  },
  {
    id: "discretization", label: "Discretise", title: "Spectral accuracy, in any geometry",
    intro: "Sparse spectral methods, from the channel to the ball, with bases that are regular by construction.",
    caps: [
      { title: "Boxes, channels and infinite lines", text: "Fourier and Chebyshev directions; sine and cosine series for stress-free or fixed-temperature walls, with no boundary terms at all; Hermite and Laguerre functions on the whole and the half line; compound intervals for interior layers; mapped bases and complex paths for inviscid modes.", more: ["/geometries#lines", "Geometries"] },
      { title: "Disks, pipes, spheres, shells and balls", text: "Curved geometries without axis conditions or fixes at the poles: every field is regular by construction. Rigid or stress-free walls are one declaration.", more: ["/geometries", "Geometries"] },
      { title: "Symmetries that halve the work", text: "Declare a reflection or a rotation and Conatus solves in the symmetric subspace, after checking that the equations respect it." },
    ],
  },
  {
    id: "runtime", label: "Run", title: "On a laptop or a GPU",
    intro: "The same model runs on a CPU or a GPU. Resolution, scheme and time step are yours to choose, never guessed.",
    caps: [
      { title: "Time integration", text: "Implicit–explicit schemes, multistep (backward differentiation and Crank–Nicolson) and Runge–Kutta: the schemes of Dedalus, with the same numbers. Boundary values that move in time are imposed at every stage: second order or better for every scheme (RK443 about 2.8), where the usual convention falls to first order. Exact restarts from a checkpoint.", more: ["/tutorials/getting-started#simulate", "Tutorial 0, step 6"] },
      { title: "GPU-native", text: "One keyword moves a simulation to the GPU, in a box, a channel, a disk, a pipe, on a sphere, in a ball or a spherical shell; nonlinear terms become fused kernels, and in boxes, channels and shells whole steps can be replayed as one graph." },
    ],
  },
  {
    id: "dynamics", label: "Analyse", title: "The dynamical system behind the equations",
    intro: "The problem you simulate is also a dynamical system: Conatus finds its structure.",
    caps: [
      { title: "Stability", text: "Eigenvalues with their residuals, critical parameters and sensitivities, with spurious modes removed in the open.", more: ["/dynamics#eigen", "Dynamics"] },
      { title: "Orbits and bifurcations", text: "Equilibria, travelling waves and periodic orbits; branches followed through folds, pitchforks and Hopf points.", more: ["/dynamics#bifurcations", "Dynamics"] },
      { title: "Time-periodic flows", text: "Floquet exponents and optimal transient growth about any flow that repeats in time, written as a function of time, such as pulsatile flow in a channel.", more: ["/dynamics#floquet", "Dynamics"] },
      { title: "Adjoints and transient growth", text: "Exact adjoints of the time stepper, in boxes, channels and the spherical shell, for gradients and optimisation; linear and nonlinear optimal perturbations of non-normal flows.", more: ["/dynamics#adjoint", "Dynamics"] },
    ],
  },
  {
    id: "trust", label: "Keep track", title: "Observable and reproducible",
    intro: "A computation you can find again afterwards.",
    caps: [
      { title: "Runs recorded on request", text: "With record=True or cn.runs.track, the parameters, code version and results of a run are kept; find, compare and reproduce them later. Nothing is recorded unless you ask." },
      { title: "Files any tool can read", text: "Snapshots and checkpoints in HDF5, with the provenance of the run inside; a run split over several jobs continues exactly as if it had never stopped.", more: ["docs:guide/io.html", "Input and output"] },
      { title: "A number with its error bar", text: "From the resolutions and time steps you declare, a computed quantity is reported with an error estimate and the checks behind it; time averages come with their statistical error.", more: ["docs:guide/uncertainty.html", "Numerical uncertainty"] },
      { title: "A Methods paragraph, written for you", text: "Each computation can describe itself: the discretisation, the scheme and the references, ready for a paper." },
    ],
  },
];

export default function Capabilities() {
  return (
    <>
      <PageHead eyebrow="Capabilities" title="What you can do with Conatus">
        <p>Write the equations, run them where you like, and let Conatus find the structure of the solutions: instabilities,
          orbits, bifurcations. Everything below is available from the same language of equations.</p>
      </PageHead>
      <nav className="wrap parts-nav" aria-label="Parts">
        <ol>{PARTS.map((p) => <li key={p.id}><a href={`#${p.id}`}>{p.label}</a></li>)}</ol>
      </nav>
      {PARTS.map((part, i) => (
        <Section key={part.id} id={part.id} eyebrow={part.label} title={part.title}
          intro={<p>{part.intro}</p>} className={i % 2 === 1 ? "lab" : undefined}>
          <ul className="caps-grid">
            {part.caps.map((c) => (
              <li key={c.title} className="cap">
                <h3>{c.title}</h3>
                <p>{c.text}</p>
                {c.more && <p className="cap-more">{c.more[0].startsWith("docs:")
                  ? <DocLink to={c.more[0].slice(5)}>{c.more[1]} <span aria-hidden="true">→</span></DocLink>
                  : <Link to={c.more[0]}>{c.more[1]} <span aria-hidden="true">→</span></Link>}</p>}
              </li>
            ))}
          </ul>
        </Section>
      ))}
      <div className="wrap caps-validation">
        <p>How these capabilities compare with published results and independent codes is on the <Link to="/validation">validation page</Link>.</p>
      </div>
    </>
  );
}
