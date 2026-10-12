// Tutorial: the guided path from install to a first real result and beyond . One continuous story,
// Rayleigh-Benard convection: the model as on paper, how to read explain(), stability of the fluid
// at rest, the onset found to the published digits, a simulation that grows at the predicted rate
// and saturates into rolls, the Methods paragraph, the run registry, the GPU. Every code block is a
// region of the executed tutorial docs/site/tutorials/code/t00_getting_started.py and every output
// line comes from its executed log (lib/snippet.ts: strict cuts, stale outputs never shown).
// The one figure is the script's own output.
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ButtonLink, Code, CopyButton, DocLink, PageHead } from "../components/ui";
import Executed from "../components/Executed";
import { pick, region, tutorial } from "../lib/snippet";
import { tutorials } from "../data";
import "../styles/tutorial.css";

const PAGE = "web/src/pages/Tutorial.tsx";
const { code: t00, out: t00out } = tutorial("t00_getting_started");
const { code: t02b, out: t02bout } = tutorial("t02b_rayleigh_benard_gpu");
const r = (name: string) => region(t00, name, PAGE);

// The figure written by the script (synced next to its output); shown only with a fresh output.
const FIGS = import.meta.glob<string>("../../../docs/site/tutorials/outputs/t00_getting_started_*.png",
  { query: "?url", import: "default", eager: true });
const ROLLS = t00out ? Object.entries(FIGS).find(([k]) => k.endsWith("_rolls.png"))?.[1] ?? null : null;

const METHODS = t00out ? pick(t00out, [/^The IVP 'Rayleigh-Benard'/], PAGE).split(". ").slice(0, 3).join(". ") + ". …" : null;

const INSTALL = `$ python3.12 -m venv venv && . venv/bin/activate
$ pip install "conatus[tutorials]"  # the CPU wheel, compiled core included (+ matplotlib for the figures)
$ python -m conatus doctor          # what this machine can run, and what is missing`;

const SECTIONS: [string, string][] = [
  ["install", "Install"], ["model", "Write the model"], ["explain", "Read what Conatus decided"],
  ["stability", "Is the fluid at rest stable?"], ["onset", "Where convection starts"],
  ["simulate", "Simulate above onset"], ["rolls", "Steady rolls"], ["record", "The Methods paragraph and the run"],
  ["gpu", "On a GPU"], ["check", "The checks"], ["next", "What next"],
];

function Snippet({ code, lang = "python", title }: { code: string; lang?: "python" | "bash"; title?: string }) {
  const plain = lang === "bash" ? code.split("\n").filter((l) => l.startsWith("$ ")).map((l) => l.slice(2)).join("\n") : code;
  return (
    <div className="tu-snippet">
      <CopyButton text={plain} />
      <Code lang={lang} title={title}>{code}</Code>
    </div>
  );
}

function Part({ id, n, title, children }: { id: string; n: number; title: string; children: ReactNode }) {
  return (
    <section className="tu-part" id={id} aria-labelledby={id + "-h"}>
      <h2 id={id + "-h"}><span className="tu-n">{n}.</span> {title}</h2>
      {children}
    </section>
  );
}

/** The equations of the model, set as mathematics (no script: plain HTML). */
function Equations() {
  return (
    <div className="tu-eqs" role="math" aria-label="Boussinesq equations">
      <p>∂<sub>t</sub><b>u</b> + <b>u</b>·∇<b>u</b> = −∇<i>p</i> + Pr ∇²<b>u</b> + Pr Ra <i>T</i> <b>e</b><sub><i>z</i></sub></p>
      <p>∇·<b>u</b> = 0</p>
      <p>∂<sub>t</sub><i>T</i> + <b>u</b>·∇<i>T</i> = ∇²<i>T</i></p>
      <p className="tu-eqs-bc"><b>u</b> = 0 and <i>T</i> = ±½ on the plates <i>z</i> = ∓½</p>
    </div>
  );
}

export default function Tutorial() {
  const more = tutorials.tutorials.filter((t) => t.n > 0);
  return (
    <>
      <PageHead eyebrow="Tutorial 0" title="Getting started with Conatus">
        <p>One problem from beginning to end: a layer of fluid heated from below. You will write its equations as on paper,
          ask whether the fluid at rest is stable, find where convection starts to the digits of the published value, then
          simulate the rolls that appear above onset and measure the heat they carry. About half an hour, on a laptop.</p>
      </PageHead>

      <div className="wrap tu-layout">
        <nav className="tu-toc" aria-label="On this page">
          <p className="tu-toc-t">On this page</p>
          <ol>{SECTIONS.map(([id, t]) => <li key={id}><a href={`#${id}`}>{t}</a></li>)}</ol>
        </nav>

        <article className="tu-body">
          <p>Everything below is one Python script, shown piece by piece. It ran as shown, and each output is what it printed.
            The whole script is <DocLink to="tutorials/00_getting_started.html">in the documentation</DocLink>; run it
            with <code>python t00_getting_started.py</code>.</p>

          <Part id="install" n={1} title="Install">
            <p>Conatus needs Linux x86-64 and Python 3.11, 3.12 or 3.13 (3.12 recommended); the wheel carries its
              compiled core, so no compiler is needed. The{" "}
              <code>tutorials</code> extra adds matplotlib, which draws this tutorial's figure. The last line checks the
              installation and says what this machine can run: the compiled core, a GPU.</p>
            <Snippet lang="bash" code={INSTALL} />
            <p>For a GPU, see <Link to="/install">Install</Link>: the CUDA 12 wheel (beta) of the same release.</p>
          </Part>

          <Part id="model" n={2} title="Write the model">
            <p>Rayleigh–Bénard convection is the flow of a fluid layer between a hot plate below and a cold plate above.
              Warm fluid is lighter and wants to rise; viscosity and heat diffusion hold it back. In the Boussinesq
              approximation, with lengths in units of the layer depth and times in units of the thermal diffusion time, the
              velocity <b>u</b>, the pressure <i>p</i> and the temperature <i>T</i> obey</p>
            <Equations />
            <p>Two numbers set the physics: the Rayleigh number Ra, the strength of the heating against the two diffusions, and
              the Prandtl number Pr, the ratio of viscosity to heat diffusivity. In Conatus the model is these lines:</p>
            <Snippet code={r("imports") + "\n\n" + r("model")} title="t00_getting_started.py" />
            <p>Read it as you would read the equations. <code>cn.Box</code> declares the domain: periodic in <i>x</i> with
              length 2 (one pair of counter-rotating rolls), between walls at <i>z</i> = ±½. The domain names its
              boundaries, <code>bottom</code> and <code>top</code>. The model declares its unknowns (a velocity, a pressure,
              a scalar) and its parameters, then the equations in a string, written as on paper: <code>u·grad(u)</code> is
              the advection, <code>ez</code> the vertical unit vector. The conditions are stated where they hold. There is
              no basis to choose, no resolution yet, and nothing to say about how the boundary conditions are imposed.</p>
          </Part>

          <Part id="explain" n={3} title="Read what Conatus decided">
            <p>Some decisions are needed to turn the model into a computation. Conatus makes them by fixed rules, and{" "}
              <code>explain()</code> lists every one. A few lines of its report:</p>
            <Executed out={t00out} page={PAGE} label="m.explain() (excerpt)" patterns={[
              /^Time stepping:/, /^Boundary conditions:/, /^Gauge:/, /^Dealiasing:/]} />
            <p>Linear terms are treated implicitly and nonlinear ones explicitly; <code>m.explain(full=True)</code> lists
              every term with the side it went to and why. Each boundary condition is attached to the equation where its
              unknown has its highest derivative across the walls. The pressure is recognised from the structure of the equations, not
              from its name, and its free constant is fixed. Dealiasing uses the standard 3/2 rule, said aloud. Anything the
              rules cannot decide is refused with a message that says what to write instead: nothing is guessed.</p>
          </Part>

          <Part id="stability" n={4} title="Is the fluid at rest stable?">
            <p>With the plates at fixed temperatures, the fluid can stay at rest while heat simply conducts through it:{" "}
              <i>T</i> = −<i>z</i>, <b>u</b> = 0. Whether it does depends on Ra. Ask the model for the normal modes of small
              disturbances to that state, for the rolls that fit the box (wavenumber <i>k</i> = π):</p>
            <Snippet code={r("stability")} />
            <Executed out={t00out} page={PAGE} patterns={[/^Ra = 1500: /, /^Ra = 2500: /]} />
            <p><code>linearize</code> derives the linear equations from the ones you wrote; <code>eigenvalues</code> solves
              them with 24 Chebyshev modes across the layer. Here you choose the resolution: Conatus never does. At
              Ra = 1500 every disturbance decays; at Ra = 2500 one grows. Convection starts in between.</p>
          </Part>

          <Part id="onset" n={5} title="Where convection starts">
            <p>To find the threshold, write the problem as it appears in textbooks. A disturbance of wavenumber <i>k</i>{" "}
              behaves as e<sup><i>ikx</i> + <i>st</i></sup>; its vertical velocity <i>w</i>(<i>z</i>) and temperature
              θ(<i>z</i>) satisfy two ordinary differential equations in which <i>s</i> is the eigenvalue. This is a model on
              a line, with the eigenvalue declared:</p>
            <Snippet code={r("onset")} />
            <Executed out={t00out} page={PAGE} patterns={[/^Ra_c = /]} />
            <p><code>parametric</code> keeps Ra and <i>k</i> as live parameters of the discretised problem;{" "}
              <code>critical</code> finds the Ra at which the least stable mode stops decaying, and minimises it over{" "}
              <i>k</i>. The filter <code>abs(s) &lt; 100</code> is a declared choice: it sets aside the spurious modes that
              boundary conditions add to any polynomial discretisation, far out in the spectrum. The result is the classical
              onset of convection between rigid plates, Ra<sub>c</sub> = 1707.762 at <i>k</i><sub>c</sub> = 3.117 (Reid &amp;
              Harris 1958; Chandrasekhar 1961), to every published digit.</p>
          </Part>

          <Part id="simulate" n={6} title="Simulate above onset">
            <p>Back to the full equations, at Ra = 2500. A simulation needs a resolution, a time-stepping scheme and a time
              step: three arguments, never defaults. Start from the conduction state with a tiny roll, and measure how fast
              it grows:</p>
            <Snippet code={r("simulate")} />
            <Executed out={t00out} page={PAGE} patterns={[/^measured growth rate/]} />
            <p>The simulation and the eigenvalue problem are two different computations from the same model; while the
              disturbance is small they must agree, and they do, to the sixth digit. <code>run(until=...)</code> refuses a
              time that is not a whole number of steps rather than change the step behind your back.</p>
          </Part>

          <Part id="rolls" n={7} title="Steady rolls">
            <p>Let the rolls grow until the nonlinear terms stop them. The run is recorded in a registry as it goes (next
              section). Then measure the Nusselt number Nu, the heat carried across the layer in units of what conduction
              alone would carry, three ways: averaged over the layer, and through each plate.</p>
            <Snippet code={r("nusselt") + "\n\n" + r("saturate").replace(/\nprint\(run\.explain[^\n]*\nprint\(sim\.methods[^\n]*$/, "")} />
            <Executed out={t00out} page={PAGE} patterns={[/^t = 6\.0: Nu = /]} />
            <p>In a steady state the heat that enters through the hot plate leaves through the cold one, and crosses every
              level in between: the three numbers agree. The rolls carry about half as much heat again as conduction.</p>
            {ROLLS && (
              <figure className="tu-fig">
                <img src={ROLLS} width={960} height={510} loading="lazy"
                  alt="A rectangle of fluid on a black ground: on the left and right edges red regions of warm fluid rising, in the middle a grey-green region of cooler fluid sinking, with thin white isotherms bent down by the flow in the middle." />
                <figcaption className="caption">The saturated rolls at Ra = 2500: temperature departure from conduction (red
                  warm, green cool) and isotherms. Drawn by the script from the computed field.</figcaption>
              </figure>
            )}
          </Part>

          <Part id="record" n={8} title="The Methods paragraph and the run">
            <p>Every simulation can describe itself: <code>sim.methods()</code> writes the paragraph of a paper that says how
              the result was computed, with its references. Its opening:</p>
            {METHODS && <blockquote className="tu-methods">{METHODS}</blockquote>}
            <p>The <code>with cn.runs.track(...)</code> block above kept the run in a registry (a folder,
              here <code>runs</code>): the problem, the numerics, the parameters, the code version, the start and end states,
              and the results you attach. Later, <code>cn.runs.find(Nu__gt=1.4)</code> finds it again,{" "}
              <code>compare</code> shows what differs between two runs, and <code>reproduce</code> checks a rerun bit for
              bit. See <DocLink to="guide/runs.html">the run registry</DocLink>.</p>
          </Part>

          <Part id="gpu" n={9} title="On a GPU">
            <p>The same model runs on an NVIDIA GPU with one more argument, <code>backend="cuda"</code>; the results agree with
              the CPU to rounding. From the GPU tutorial, which compares the two:</p>
            <Snippet code={region(t02b, "backends", PAGE)} title="t02b_rayleigh_benard_gpu.py (excerpt)" />
            <Executed out={t02bout} page={PAGE} label="Output (GPU)" patterns={[/^after 20 steps/]} />
            <p>To watch a run while it computes, from a browser or a phone, declare what to show with{" "}
              <code>sim.watch(...)</code>: see the user guide.</p>
          </Part>

          <Part id="check" n={10} title="The checks">
            <p>The script ends by checking itself against the literature and against the identities above. If Conatus ever
              stopped reproducing them, the script would fail: the outputs on this page come from its last execution, which
              passed.</p>
            <Snippet code={r("check")} />
            <Executed out={t00out} page={PAGE} patterns={[/^OK:/]} />
          </Part>

          <Part id="next" n={11} title="What next">
            <p>Things to try with this script: raise Ra and measure how Nu grows (a faster flow may need a smaller time step:{" "}
              <code>sim.cfl()</code> suggests one, it never imposes one); change Pr; make the box longer, <code>cn.periodic(4.0)</code>, and see which roll pair grows first. Other
              domains, a disk or a spherical shell, are one line each: see <Link to="/geometries">Geometries</Link> and,
              for shells, <DocLink to="guide/shells.html">the user guide</DocLink>. Then tutorials 1 to 14, each a
              short script ending on a published result:</p>
            <ol className="tu-more">
              {more.map((t) => (
                <li key={t.slug}><DocLink to={`tutorials/${t.slug}.html`}>{t.title}</DocLink>
                  {t.learn && <span className="tu-more-l"> — {t.learn}</span>}</li>
              ))}
            </ol>
            <p>Every part of Conatus is described once in the <Link to="/documentation">documentation</Link>: the user guide,
              the API reference and the mathematics behind each method.</p>
            <div className="row tu-cta">
              <ButtonLink to="/tutorials">All tutorials</ButtonLink>
              <ButtonLink to="/documentation" variant="ghost">Documentation</ButtonLink>
            </div>
          </Part>
        </article>
      </div>
    </>
  );
}
