// Coming next: features under development that are not part of this release. No dates, no performance
// claims: each item says what it is for and that it is not in this release.
import { Link } from "react-router-dom";
import { PageHead, Section } from "../components/ui";
import "../styles/tutorials.css";

export default function ComingNext() {
  return (
    <>
      <PageHead eyebrow="Roadmap" title="Coming next">
        <p>What is being built beyond this release. None of it is part of this release, and nothing here is a promise of a date
          or of a performance figure; each part will ship when it is validated like the rest of the library.
          For what this release does today, see <Link to="/capabilities">Capabilities</Link>.</p>
      </PageHead>
      <Section id="multi-gpu" title="Multi-GPU runs across nodes" intro={<p>
        A distributed spectral solver that splits one simulation over several GPUs and several nodes, with the same model
        and the same script as on one GPU. It works and is being optimised; the exchange between nodes is where the work
        is.</p>} />
      <Section id="studio" title="Conatus Studio from anywhere" intro={<p>
        The Studio of this release runs on your machine, or inside a batch job reached through an ssh tunnel. Next: a hosted
        hub with authenticated access that connects a browser or a phone to runs on a cluster from anywhere, through a relay
        that only makes outgoing connections, so the cluster is never opened to the outside.</p>} />
      <Section id="stability" title="Further stability tools" intro={<p>
        Resolvent analysis and pseudospectra in a declared energy norm; periodic orbits of three-dimensional channel and
        pipe flows; certified counting of all eigenvalues inside a region of the complex plane by contour methods, so that
        no mode is missed; and high-resolution spherical shells.</p>} />
    </>
  );
}
