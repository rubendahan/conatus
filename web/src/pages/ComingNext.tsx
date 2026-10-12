// Coming next: features under development that are not part of 0.1.0. No dates, no performance
// claims: each item says what it is for and that it is not in this release.
import { Link } from "react-router-dom";
import { PageHead, Section } from "../components/ui";
import "../styles/tutorials.css";

export default function ComingNext() {
  return (
    <>
      <PageHead eyebrow="Roadmap" title="Coming next">
        <p>What is being built beyond 0.1.0. None of it is part of this release, and nothing here is a promise of a date
          or of a performance figure; each part will ship when it is validated like the rest of the library.
          For what 0.1.0 does today, see <Link to="/capabilities">Capabilities</Link>.</p>
      </PageHead>
      <Section id="multi-gpu" title="Multi-GPU runs across nodes" intro={<p>
        A distributed spectral solver that splits one simulation over several GPUs and several nodes, with the same model
        and the same script as on one GPU. It works and is being optimised; the exchange between nodes is where the work
        is.</p>} />
      <Section id="studio" title="Conatus Studio" intro={<p>
        Watch and steer a running simulation from a browser, on a laptop or a phone: pause it, change a parameter, fork the
        run into a new branch, or seed a Newton search from the live state, without changing a bit of the computation it
        does not touch. A hosted hub with authenticated access connects the browser to runs on a cluster, without opening
        the cluster to the outside.</p>} />
      <Section id="agents" title="An interface for AI agents" intro={<p>
        A compact, machine-readable description of the API and a JSON command-line entry point, so that an assistant can
        set up a run, check a script before it runs, and explain what a model, a solver or a result contains, from the same
        facts the library prints for a person.</p>} />
      <Section id="stability" title="Further stability tools" intro={<p>
        Resolvent analysis and pseudospectra in a declared energy norm; periodic orbits of three-dimensional channel and
        pipe flows; certified counting of all eigenvalues inside a region of the complex plane by contour methods, so that
        no mode is missed; and high-resolution spherical shells.</p>} />
    </>
  );
}
