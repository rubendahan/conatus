// Geometries: the domains of the native layer (cn.Box, cn.Channel, cn.Disk, ...), each with the line
// that declares it, its coordinates and the names of its boundaries, checked against
// python/conatus/model/geometry.py. Text and code only (Ruben, 2026-10-02: text-first pages).
import { Link } from "react-router-dom";
import { Code, DocLink, PageHead, Section } from "../components/ui";
import "../styles/geometries.css";

interface Geo { id: string; name: string; code: string; coords: string; walls: string; text: string; uses: string; guide?: [string, string] }

const GEOS: Geo[] = [
  { id: "box", name: "Box", code: "cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))",
    coords: "x, z (one to three axes, in the order written)", walls: "bottom, top, walls (or z=-0.5, z=0.5)",
    text: "Periodic directions and at most one direction between two walls. Fourier along periodic axes, Chebyshev between walls, or sine and cosine series (basis=\"sincos\") when the walls are stress-free or held at a fixed value.",
    uses: "Convection, Couette flow, boundary layers, pattern formation." },
  { id: "lines", name: "Lines, half lines and compound intervals", code: "cn.Box(x=cn.periodic(2.0), y=cn.half_line(scale=0.5))",
    coords: "as named in the box", walls: "wall (the half line); none on cn.line(scale=...)",
    text: "Unbounded directions with the decay built into the basis (Hermite functions on the line, Laguerre functions on the half line); a width you choose, like a resolution. An interval cut into pieces, cn.walls(-1, -0.15, 0.15, 1), puts modes where a thin layer needs them, with continuity across the cuts added for you.",
    uses: "Boundary layers over a plate, localised waves, fronts and interior layers." },
  { id: "channel", name: "Channel", code: "cn.Channel(2 * math.pi, math.pi)",
    coords: "x, y, z (2D without the second length)", walls: "bottom, top, walls",
    text: "A box with walls at y = ±1, periodic in x and, in 3D, in z.",
    uses: "Plane Poiseuille and Couette flow, turbulent channels." },
  { id: "disk", name: "Disk", code: "cn.Disk(1.0)",
    coords: "phi, r", walls: "wall",
    text: "The radial basis enforces regularity at the centre, so no axis condition is needed.",
    uses: "Vortices, rotating tanks, flows in a circular cross-section." },
  { id: "annulus", name: "Annulus", code: "cn.Annulus(1.0, 2.0)",
    coords: "phi, r", walls: "inner, outer, walls",
    text: "The gap between two circles.",
    uses: "Taylor–Couette flow in a cross-section, rotating shear flows." },
  { id: "sphere", name: "Sphere", code: "cn.Sphere(1.0)",
    coords: "phi, theta", walls: "none",
    text: "The surface of a sphere, regular at both poles.",
    uses: "Shallow water, planetary waves, atmospheres." },
  { id: "shell", name: "Spherical shell", code: "cn.Shell(0.35, 1.0)",
    coords: "phi, theta, r", walls: "inner, outer, walls",
    text: "The space between two spheres, where planets and stars convect. Simulations on a CPU or a GPU, adjoints and optimal perturbations, and a ready rotating-convection problem. The wall conditions use a formulation designed to stay well conditioned as the radial resolution grows, and they are checked before the first step; the user guide lists the tested sizes.",
    uses: "Rotating convection, dynamos, mantle and core flows, rotating shells.",
    guide: ["guide/shells.html", "Spherical shells in the user guide"] },
  { id: "ball", name: "Ball", code: "cn.Ball(1.0)",
    coords: "phi, theta, r", walls: "wall",
    text: "The full sphere. The basis is regular at the centre.",
    uses: "Flows in a spherical container, stellar interiors." },
];

const EXAMPLE = `import conatus as cn

disk = cn.Disk(1.0)                       # coordinates (phi, r), boundary "wall"
m = cn.Model(disk, name="heat in a disk", kappa=1.0)
T = m.scalar("T")
m.equations("dt(T) - kappa*lap(T) = 0")
m.at("wall", T=0)
print(m.explain())                        # bases, conditions and their placement`;

export default function Geometries() {
  return (
    <>
      <PageHead eyebrow="Geometries" title="From the channel to the ball">
        <p>In every geometry you name the domain, write the equations and name the boundaries where conditions hold.
          Conatus chooses the bases of the geometry, regular at the axis and at the centre, and prints them with
          <code>explain()</code>.</p>
      </PageHead>
      <Section id="example" title="Declaring a domain" intro={<p>A geometry is one line. The rest of the model does not
        depend on it: the same equation in a box or in a ball differs only by this line and by the boundary names.</p>}>
        <Code title="a model on a disk">{EXAMPLE}</Code>
      </Section>
      <Section id="domains" title="The geometries" className="geo-section">
        <dl className="geo-list">
          {GEOS.map((g) => (
            <div key={g.id} id={g.id} className="geo-row">
              <dt><h3 className="geo-name">{g.name}</h3><code className="geo-code">{g.code}</code></dt>
              <dd>
                <p>{g.text}</p>
                <p className="geo-meta"><span>Coordinates</span> {g.coords}<br /><span>Boundaries</span> {g.walls}</p>
                <p className="geo-uses">{g.uses}</p>
                {g.guide && <p className="geo-more"><DocLink to={g.guide[0]}>{g.guide[1]} →</DocLink></p>}
              </dd>
            </div>
          ))}
        </dl>
        <p>Every wall takes rigid or stress-free conditions (<code>u=cn.free_slip</code>), curved walls included. Mapped bases,
          other polynomial families and hand-chosen bases are available in the expert layer;
          see <DocLink to="guide/bases.html">the user guide</DocLink>. The gap between two rotating cylinders is <code>cn.TaylorCouette(ri, ro, length)</code>; pipe flow and Taylor–Couette flow
          are also ready problems (<code>cn.problems</code>), as native models in <code>cn.problems.native</code>. To see the geometries at work, open
          the <Link to="/gallery">gallery</Link>.</p>
      </Section>
    </>
  );
}
