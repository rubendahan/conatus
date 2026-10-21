// Coming next: planned features (not in this release) and extensions of tools this release ships.
// No dates and no performance targets.
import { Link } from "react-router-dom";
import { DocLink, PageHead, Section } from "../components/ui";
import "../styles/tutorials.css";

export default function ComingNext() {
  return (
    <>
      <PageHead eyebrow="Roadmap" title="Coming next">
        <p>This page lists work in progress. Some items are new features. Others extend tools that this release already ships
          as experimental. No item has a date or a performance target. For what this release does, see{" "}
          <Link to="/capabilities">Capabilities</Link>.</p>
      </PageHead>
      <Section id="planned" title="Planned features" intro={<p>
        These are not in this release.</p>} />
      <Section id="multi-gpu" title="Multi-GPU runs across nodes" intro={<p>
        A distributed spectral solver that splits one simulation over several GPUs and nodes, driven by the same model
        and script as a one-GPU run. The solver runs today in development. The current work is on the exchange of data
        between nodes.</p>} />
      <Section id="studio" title="Studio from a remote browser" intro={<p>
        The Studio of this release runs on your machine, or inside a batch job reached through an ssh tunnel. The next step
        is a hosted hub with authenticated access. A relay on the cluster would only make outgoing connections, so the
        cluster stays closed to inbound traffic.</p>} />
      <Section id="resolvent" title="Resolvent analysis and pseudospectra" intro={<p>
        Response to forcing and pseudospectra in a declared energy norm.</p>} />
      <Section id="extensions" title="Experimental tools in this release" intro={<p>
        Contour counting of eigenvalues ships in this release. Three tools ship as experimental and are still being
        validated: orbits of 3-D channel and pipe flow, spherical shells larger than 64 × 48 × 48, and the local
        Studio. The <DocLink to="guide/experimental.html">experimental features</DocLink> page gives their current
        limits.</p>} />
    </>
  );
}
