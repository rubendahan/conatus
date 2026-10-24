// About: purpose, status, maintenance, how to report a problem, the optional tools, the plans, the
// acknowledgements and the origin of the name (presentation review of 0.1.1, priorities 12, 14 and 15).
// These were spread over the home page, the footer and the README; they are kept here, one short
// section each, so the home page can stay on the research task.
import { Link } from "react-router";
import { DocLink, PageHead, RepoPath, Section } from "../components/ui";
import { SITE } from "../content/site";
import { routes } from "../routes";
import { withBase } from "../lib/base";
import "../styles/about.css";

const has = (path: string) => routes.some((r) => r.path === path);

export default function About() {
  const issues = SITE.repo ? <a href={`${SITE.repo}/issues`}>the issue tracker</a> : <>the issue tracker of the repository</>;
  return (
    <>
      <PageHead eyebrow="About" title="About Conatus">
        <p>Conatus is a Python library for spectral discretisations of partial differential equations, written for
          hydrodynamic-stability and dynamical-systems research: linear stability, time integration of the same models,
          and the search for steady and periodic solutions.</p>
      </PageHead>

      <Section id="status" title="Status">
        <div className="about-body">
          <p>Version {SITE.version} is experimental research software. The supported platform is Linux x86-64 with CPython
            3.11 to 3.13, through the CPU wheels on PyPI. CUDA support is experimental. The interface can change between
            minor versions; changes are announced by a deprecation warning and listed in the changelog.</p>
          <p>The examples and tutorials compare their results with published values or closed forms. These checks cover the
            cases shown; they do not establish that every combination of geometry, equation and option works.{" "}
            <Link to="/capabilities">Capabilities</Link> marks what is experimental, and{" "}
            <Link to="/validation">Validation</Link> lists the comparisons.</p>
        </div>
      </Section>

      <Section id="maintenance" title="Who maintains it">
        <div className="about-body">
          <p>Conatus is written and maintained by Ruben Dahan. It is developed with AI coding agents under the direction of
            one researcher.</p>
          <p>Support is best effort. To report a problem, open an issue on {issues} with a short script that reproduces it,
            what it printed, and the output of <code>python -m conatus doctor</code>.</p>
        </div>
      </Section>

      <Section id="tools" title="Optional tools">
        <ul className="about-list">
          <li><strong>Studio.</strong> Watch a running simulation in a browser and pause, checkpoint or stop it:{" "}
            {has("/studio") ? <Link to="/studio">Studio</Link> : <DocLink to="guide/studio.html">the Studio guide</DocLink>}.</li>
          <li><strong>Coding agents.</strong> <code>python -m conatus agent</code> describes the installation and the API in
            JSON, and {SITE.repo ? <RepoPath path="AGENTS.md">AGENTS.md</RepoPath> : <a href={withBase("/AGENTS.md")}>AGENTS.md</a>} is a short page written for coding agents.</li>
          <li><strong>GPU.</strong> The same model runs on one NVIDIA GPU with <code>backend="cuda"</code> (experimental; CUDA 12
            wheel): see <Link to="/install">Install</Link>.</li>
          <li><strong>Plans.</strong>{" "}
            {has("/coming-next")
              ? <><Link to="/coming-next">Coming next</Link> lists planned work; none of it is in this release.</>
              : <>Planned work (several GPUs, further stability tools) is not part of this release.</>}</li>
        </ul>
      </Section>

      <Section id="acknowledgements" title="Methods and acknowledgements">
        <div className="about-body">
          <p>Conatus implements published methods: among them the sparse ultraspherical spectral method (Olver &amp; Townsend
            2013), the generalised tau method (Burns et al. 2020), polar and spherical bases (Vasil
            et al. 2016, 2019) and implicit–explicit Runge–Kutta schemes (Ascher, Ruuth &amp; Spiteri 1997). A run lists the
            methods it used, with references, through <code>methods()</code>; <Link to="/citing">Citing</Link> has the
            BibTeX of the software. Licence: BSD-3-Clause ({SITE.repo ? <RepoPath path="LICENSE">LICENSE</RepoPath> : <code>LICENSE</code>}).</p>
        </div>
      </Section>

      <Section id="name" title="The name">
        <p className="about-epigraph">
          <span lang="la">Unaquæque res, quantum in se est, in suo esse perseverare conatur.</span>
          <span className="caption">Each thing, as far as it is in itself, strives to persevere in its being. Spinoza,{" "}
            <i>Ethica</i> III, proposition 6.</span>
        </p>
      </Section>
    </>
  );
}
