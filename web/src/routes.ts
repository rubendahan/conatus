// Route table of the website: one entry per page. Used by the router (client and server),
// the prerender script (one static HTML file per path), the navigation and the sitemap.
import type { ComponentType } from "react";

export type NavGroup = "primary" | "more" | "none";

export interface RouteDef {
  path: string;
  title: string;          // <title> and navigation label (label may be shorter: `nav`)
  nav?: string;
  group: NavGroup;
  description: string;    // meta description, also shown in search results
  load: () => Promise<{ default: ComponentType }>;
  src: string;            // module path, used by the prerender to preload the page's JS and CSS
}

export const routes: RouteDef[] = [
  { path: "/", title: "Conatus: one PDE model for simulation, stability and invariant solutions", nav: "Home", group: "none",
    description: "Write a PDE model once, then simulate it, compute its stability and find its invariant solutions from the same object. Conatus is experimental research software for fluid instability and dynamical systems.",
    load: () => import("./pages/Home"), src: "src/pages/Home.tsx" },
  { path: "/tutorials/getting-started", title: "First steps: the Burgers equation in ten minutes", nav: "Getting started", group: "primary",
    description: "Getting started with Conatus in ten minutes: one equation written as on paper, a simulation, a plot and one physical number checked against the exact solution, then four guided changes.",
    load: () => import("./pages/Tutorial"), src: "src/pages/Tutorial.tsx" },
  { path: "/tutorials", title: "Tutorials", group: "primary",
    description: "Fifteen executed tutorials, from getting started to bifurcations and a GPU run followed live, each ending on a published value it reproduces.",
    load: () => import("./pages/Tutorials"), src: "src/pages/Tutorials.tsx" },
  { path: "/documentation", title: "Documentation", nav: "Docs", group: "primary",
    description: "The map of the Conatus documentation: getting started, the tutorials, the user guide, the API reference and the mathematics behind each method.",
    load: () => import("./pages/Documentation"), src: "src/pages/Documentation.tsx" },
  { path: "/install", title: "Install", group: "primary",
    description: "Install Conatus with pip: CPU wheel, CUDA 12 wheel (beta), or from source.",
    load: () => import("./pages/Install"), src: "src/pages/Install.tsx" },
  { path: "/gallery", title: "Gallery", group: "primary",
    description: "Flows computed with Conatus: Kelvin–Helmholtz, Rayleigh–Bénard, Kuramoto–Sivashinsky, Taylor–Couette, Swift–Hohenberg and the sphere, each with its script and validated value.",
    load: () => import("./pages/Gallery"), src: "src/pages/Gallery.tsx" },
  { path: "/capabilities", title: "Capabilities", group: "more",
    description: "What Conatus does: equations as on paper, simulation on CPU and GPU, stability, orbits, bifurcations and run tracking.",
    load: () => import("./pages/Capabilities"), src: "src/pages/Capabilities.tsx" },
  { path: "/dynamics", title: "Dynamics", group: "more",
    description: "Eigenproblems, bifurcations, periodic orbits, Floquet–Hill analysis, adjoints and non-modal growth.",
    load: () => import("./pages/Dynamics"), src: "src/pages/Dynamics.tsx" },
  { path: "/geometries", title: "Geometries", group: "more",
    description: "From the channel to the ball: Fourier, Chebyshev, Zernike and spin-weighted bases, regular by construction.",
    load: () => import("./pages/Geometries"), src: "src/pages/Geometries.tsx" },
  { path: "/validation", title: "Validation", group: "more",
    description: "How Conatus compares with published results and independent codes, topic by topic.",
    load: () => import("./pages/Validation"), src: "src/pages/Validation.tsx" },
  { path: "/about", title: "About", group: "primary",
    description: "What Conatus is for, its status and maintenance, how to report a problem, the optional tools, the plans and the origin of the name.",
    load: () => import("./pages/About"), src: "src/pages/About.tsx" },
  { path: "/coming-next", title: "Coming next", group: "more",
    description: "Planned features (multi-GPU runs, the Studio from a remote browser, resolvent analysis) and the experimental tools of this release.",
    load: () => import("./pages/ComingNext"), src: "src/pages/ComingNext.tsx" },
  { path: "/citing", title: "Citing", group: "more",
    description: "How to cite Conatus: CITATION.cff and BibTeX.",
    load: () => import("./pages/Citing"), src: "src/pages/Citing.tsx" },
];

// Old addresses that moved (W1: one tutorials index under one name). A static host has no rewrite
// rules here: the prerender writes a small page at each old path that sends the browser on
// (meta refresh + canonical link + noindex), and the client router redirects in place.
export const redirects: { from: string; to: string }[] = [
  { from: "/tutorial", to: "/tutorials/getting-started" },
  { from: "/lessons", to: "/tutorials" },
];

export const notFound: RouteDef = {
  path: "/404", title: "Page not found", group: "none",
  description: "This page does not exist.",
  load: () => import("./pages/NotFound"),
  src: "src/pages/NotFound.tsx",
};

export const allRoutes = [...routes, notFound];

export function findRoute(pathname: string): RouteDef {
  const p = pathname.replace(/\/+$/, "") || "/";
  return routes.find((r) => r.path === p) ?? notFound;
}
