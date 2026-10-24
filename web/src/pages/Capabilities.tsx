// Capabilities: what Conatus lets you do, in short plain sentences, grouped by the way you work: write,
// discretise, run, analyse, keep track. No measured figures here: the numbers are on the
// Validation page, linked once at the end.
import { Link } from "react-router";
import { DocLink, PageHead, Section } from "../components/ui";
import "../styles/capabilities.css";

// more: a route of the site, or "docs:<page>" for a page of the Sphinx documentation (DocLink)
interface Cap { title: string; text: string; more?: [string, string] }
interface Part { id: string; label: string; title: string; intro: string; caps: Cap[] }

const PARTS: Part[] = [
  {
    id: "language", label: "Write", title: "The language of equations",
    intro: "You write the problem the way it is written on paper. explain() shows what Conatus built from it.",
    caps: [
      { title: "Equations as on paper", text: "Domains, fields, parameters and equations in their usual form. Boundary conditions are given by the name of the boundary. explain() lists what Conatus placed for you." },
      { title: "Errors with a suggested fix", text: "A missing or extra boundary condition, an ill-posed split or an undeclared choice raises an error. The message says what is wrong, where and why, and suggests a correction for you to apply." },
      { title: "Symbolic derivatives", text: "Supported expressions are differentiated symbolically. Linearisations, tangents and adjoints are derived from the equations you wrote." },
    ],
  },
  {
    id: "discretization", label: "Discretise", title: "Spectral discretisation on the supported geometries",
    intro: "Sparse spectral methods on the geometries listed below, from the channel to the ball. The curved bases are regular at the axis, the centre and the poles.",
    caps: [
      { title: "Boxes, channels and infinite lines", text: "Fourier and Chebyshev directions; sine and cosine series for stress-free or fixed-temperature walls, with no boundary terms at all; Hermite and Laguerre functions on the whole and the half line; compound intervals for interior layers; mapped bases and complex paths for inviscid modes.", more: ["/geometries#lines", "Geometries"] },
      { title: "Disks, pipes, spheres, shells and balls", text: "The bases of the curved geometries are regular at the axis and the poles, so no axis condition is needed. Rigid or stress-free walls are one declaration.", more: ["/geometries", "Geometries"] },
      { title: "Symmetries that halve the work", text: "Declare a reflection or a rotation and Conatus solves in the symmetric subspace, after checking that the equations respect it." },
    ],
  },
  {
    id: "runtime", label: "Run", title: "On a laptop or a GPU",
    intro: "The same model runs on a CPU or a GPU. You choose the resolution, the scheme and the time step in the call.",
    caps: [
      { title: "Time integration", text: "Implicit–explicit multistep schemes (backward differentiation and Crank–Nicolson) and Runge–Kutta schemes, of order 1 to 4. Boundary values that move in time are imposed at the stages; on the tested heat problem RK443 then reaches order 2.8 and RK222 order 2, while first-order schemes stay first order. The guide gives the order of each scheme. Restarts from a checkpoint continue bit for bit.", more: ["docs:guide/timesteppers.html", "Time steppers"] },
      { title: "Single GPU (experimental)", text: "Not retested on GPU hardware for this release. One keyword moves a simulation to the GPU, in a box, a channel, a disk, a pipe, on a sphere, in a ball or a spherical shell; nonlinear terms become fused kernels, and in boxes, channels and shells whole steps can be replayed as one graph." },
    ],
  },
  {
    id: "dynamics", label: "Analyse", title: "The dynamical system behind the equations",
    intro: "The problem you simulate is also a dynamical system. These tools study its structure.",
    caps: [
      { title: "Stability", text: "Eigenvalues with their residuals, critical parameters and sensitivities. The result lists the modes removed by the drift filter.", more: ["/dynamics#eigen", "Dynamics"] },
      { title: "Orbits and bifurcations", text: "Equilibria, travelling waves and periodic orbits; branches followed through folds, pitchforks and Hopf points.", more: ["/dynamics#bifurcations", "Dynamics"] },
      { title: "Time-periodic flows", text: "Floquet exponents and optimal transient growth about a flow given as a periodic function of time, such as pulsatile flow in a channel.", more: ["/dynamics#floquet", "Dynamics"] },
      { title: "Adjoints and transient growth", text: "Adjoints of the discrete time stepper, in boxes, channels and the spherical shell, for gradients and optimisation; linear and nonlinear optimal perturbations of non-normal flows.", more: ["/dynamics#adjoint", "Dynamics"] },
    ],
  },
  {
    id: "trust", label: "Keep track", title: "Observable and reproducible",
    intro: "A computation you can find again afterwards.",
    caps: [
      { title: "Runs recorded on request", text: "With record=True or cn.runs.track, the parameters, code version and results of a run are kept. You can find, compare and reproduce them later. Recording is off by default." },
      { title: "Files any tool can read", text: "Snapshots and checkpoints in HDF5, with the provenance of the run inside. A run split over several jobs continues bit for bit.", more: ["docs:guide/io.html", "Input and output"] },
      { title: "A number with its error bar", text: "From three resolutions and three time steps that you supply, a computed quantity is reported with an error estimate and the checks behind it. Time averages come with their statistical error.", more: ["docs:guide/uncertainty.html", "Numerical uncertainty"] },
      { title: "A draft Methods paragraph", text: "methods() drafts a paragraph from the recorded settings: the discretisation, the scheme and the references. Add your study's validation and context." },
    ],
  },
];

export default function Capabilities() {
  return (
    <>
      <PageHead eyebrow="Capabilities" title="What you can do with Conatus">
        <p>Write the equations, run them on a CPU or a GPU, and study the structure of the solutions: instabilities,
          orbits, bifurcations. The native model compiles to the problems these tools use; some tools need an object of the
          expert layer.</p>
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
