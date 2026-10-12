// "Next step" band at the end of every page: the reading path continues instead of ending on the
// footer. The path (web/README.md, "Reading path"): Getting started → Tutorials → Documentation → Capabilities → Dynamics →
// Gallery; side pages lead back into the Tutorial.
import { Link, useLocation } from "react-router-dom";
import { findRoute } from "../routes";

const NEXT: Record<string, [string, string]> = {
  "/tutorials/getting-started": ["/tutorials", "The other tutorials, each a script that ends on a published result."],
  "/tutorials": ["/documentation", "The user guide, the API reference and the mathematics behind each method."],
  "/documentation": ["/capabilities", "Everything Conatus does, in one page."],
  "/capabilities": ["/dynamics", "Stability, bifurcations, periodic orbits and Floquet analysis."],
  "/dynamics": ["/gallery", "Flows computed with Conatus."],
  "/geometries": ["/gallery", "The disk, the pipe, the shell and the sphere, computed."],
  "/gallery": ["/tutorials/getting-started", "From the equations to a first result, every step executed."],
  "/install": ["/tutorials/getting-started", "Your first model, from the equations to the onset of convection."],
  "/citing": ["/validation", "Every published value Conatus reproduces."],
};

export default function NextStep() {
  const { pathname } = useLocation();
  const here = findRoute(pathname).path;
  const next = NEXT[here];
  if (!next) return null;
  const r = findRoute(next[0]);
  return (
    <nav className="next-step" aria-label="Next step">
      <div className="wrap">
        <Link to={r.path} className="next-step-link">
          <span className="next-step-k">Next</span>
          <span className="next-step-t">{r.title} <span aria-hidden="true">→</span></span>
          <span className="next-step-d">{next[1]}</span>
        </Link>
      </div>
    </nav>
  );
}
