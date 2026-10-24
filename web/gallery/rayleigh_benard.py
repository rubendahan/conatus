"""Website gallery: 2D Rayleigh–Bénard convection at high Rayleigh number (nonlinear IVP, GPU).

Equations exactly as tutorial 2 (docs/site/tutorials/code/t02b_rayleigh_benard_gpu.py), in
thermal-diffusion units (length H, time H^2/kappa, temperature Delta T), rigid isothermal walls
at z = -1/2 (T = 1/2) and z = +1/2 (T = -1/2), periodic in x with period Lx:

    div u + tau_p = 0
    dt T - lap T                     = -u.grad T
    dt u - Pr lap u + grad p - Pr Ra T e_z = -u.grad u          (first-order tau lifts, as in t02)

Resolution (declared): the thermal boundary layer is ~ 1/(2 Nu) thick (Nu ~ 15-20 at Ra = 1e7 in
2D, Pr = 1), i.e. ~0.03, covered by > 20 Chebyshev nodes at Nz = 256; the bulk Kolmogorov (=
Batchelor at Pr = 1) scale (Pr^2 / (Ra Nu))^(1/4) ~ 5e-3 is above the horizontal spacing Lx/Nx.
dt is fixed and explicit; the advective CFL number max|u| dt / dx is printed (an indicator, not
an error control). The run goes to --tend (several tens of free-fall times 1/sqrt(Ra Pr)).

Diagnostics: Nu = 1 + <w T> (volume mean, Chebyshev quadrature in z) and the wall Nusselt numbers
-dT/dz, time-averaged over the last --avg-window. Writes $CN_OUT/rb_fields.npz and the image
`rayleigh-benard` (total temperature, "smoke" scale) through web/gallery/brand_cmaps.py.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as Ch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand_cmaps as C  # noqa: E402

from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, IVPSolver,  # noqa: E402
                         Parameter, RealFourier, VectorField, discretize, div, dot, dt, grad,
                         integ, interp, lap, lift, unit_vector)


def rayleigh_benard(Nx, Nz, Ra, Pr, Lx, dealias):
    """The IVP of tutorial 2 (same equations, same declared numerics)."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=dealias)
    bz = Chebyshev(z, Nz, (-0.5, 0.5), dealias=dealias)
    D = Domain(cs, [bx, bz])
    u, T, p = VectorField("u", D), Field("T", D), Field("p", D)
    tp = Field("tp", Domain(cs, []))
    tT1, tT2 = Field("tT1", D.without(z)), Field("tT2", D.without(z))
    tu1, tu2 = VectorField("tu1", D.without(z)), VectorField("tu2", D.without(z))
    Ra_, Pr_ = Parameter("Ra", Ra), Parameter("Pr", Pr)
    ez = unit_vector(z)
    pb = IVP([u, T, p, tp, tT1, tT2, tu1, tu2], name="Rayleigh-Benard 2D")
    pb.add_equation((div(u) + dot(ez, lift(tu2, bz, -1)) + tp, 0))
    pb.add_equation((dt(T) - lap(T) + lift(tT1, bz, -1) + lift(tT2, bz, -2), -dot(u, grad(T))))
    pb.add_equation((dt(u) - Pr_ * lap(u) + grad(p) - Pr_ * Ra_ * T * ez
                     + lift(tu1, bz, -1) + lift(tu2, bz, -2), -dot(u, grad(u))))
    pb.add_equation((interp(T, z, "left"), 0.5))
    pb.add_equation((interp(T, z, "right"), -0.5))
    pb.add_equation((interp(u, z, "left"), 0))
    pb.add_equation((interp(u, z, "right"), 0))
    pb.add_equation((integ(p), 0))
    return discretize(pb), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ra", type=float, default=1e7)
    ap.add_argument("--Pr", type=float, default=1.0)
    ap.add_argument("--Lx", type=float, default=4.0)
    ap.add_argument("--Nx", type=int, default=1024)
    ap.add_argument("--Nz", type=int, default=256)
    ap.add_argument("--dealias", type=float, default=1.5)
    ap.add_argument("--scheme", default="RK443")
    ap.add_argument("--dt", type=float, default=1e-6)
    ap.add_argument("--tend", type=float, default=0.035)
    ap.add_argument("--sample", type=float, default=5e-4)
    ap.add_argument("--avg-window", type=float, default=0.01)
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--budget", type=float, default=1500.0, help="wall seconds before rendering")
    ap.add_argument("--preview", type=float, nargs="*", default=[], help="extra preview times")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--clip", type=float, default=99.0, help="grey range: [-A, A], A = clip-percentile of |T| in |z| < 0.4")
    ap.add_argument("--render-only", action="store_true",
                    help="re-render the images from $CN_OUT/rb_fields.npz (no time stepping)")
    a = ap.parse_args()
    out = Path(os.environ.get("CN_OUT", "."))
    if a.render_only:
        return render_only(out, a.clip)
    t0 = time.perf_counter()
    plan, T = rayleigh_benard(a.Nx, a.Nz, a.Ra, a.Pr, a.Lx, a.dealias)
    s = IVPSolver(plan, a.scheme, a.dt, backend=a.backend, f_backend="fused")
    rng = np.random.default_rng(a.seed)
    modes = [(2 * np.pi * j / a.Lx, rng.uniform(0, 2 * np.pi), rng.normal()) for j in range(1, 33)]

    def T_init(X, Z):             # conduction profile + seeded noise vanishing at the walls
        n = sum(amp * np.cos(k * X + ph) for k, ph, amp in modes) / np.sqrt(len(modes))
        return -Z + a.amp * n * np.cos(np.pi * Z)
    s.set_state({T: T_init})
    print(s.explain(), flush=True)
    build_s = time.perf_counter() - t0
    Zg = plan._as_grid(T, lambda X, Z: Z + 0 * X)[0]           # Chebyshev nodes as stored
    order = np.argsort(Zg)
    zg = Zg[order]
    xg = np.arange(a.Nx) * a.Lx / a.Nx
    dx = a.Lx / a.Nx

    def z_mean_and_walls(f):
        """Chebyshev fit in z (exact on the Nz nodes) -> mean over z and d/dz at both walls."""
        c = Ch.chebfit(2 * zg, f[..., order].T, a.Nz - 1)          # xi = 2 z in [-1, 1]
        ci = Ch.chebint(c)
        mean = (Ch.chebval(1.0, ci) - Ch.chebval(-1.0, ci)) / 2
        dc = Ch.chebder(c)
        return mean, 2 * Ch.chebval(-1.0, dc), 2 * Ch.chebval(1.0, dc)

    run = (lambda n: s.run_graph(n)) if a.backend == "cuda" else (lambda n: s.run(n))
    block = int(round(a.sample / a.dt))
    series, previews = [], {}
    tr0 = time.perf_counter()
    stop = "tend"
    while s.t < a.tend - 1e-12:
        run(block)
        Tg, ug = s.grid("T"), s.grid("u")
        if not (np.all(np.isfinite(Tg)) and np.all(np.isfinite(ug))):
            raise FloatingPointError(f"non-finite state at t = {s.t:g}")
        wT, _, _ = z_mean_and_walls(ug[1] * Tg)
        Tm = Tg.mean(axis=0)                                       # horizontal mean profile
        _, dTb, dTt = z_mean_and_walls(Tm[None, :])
        umax = float(np.sqrt(ug[0] ** 2 + ug[1] ** 2).max())
        rec = {"t": s.t, "Nu_vol": float(1 + wT.mean()), "Nu_bottom": float(-dTb[0]),
               "Nu_top": float(-dTt[0]), "umax": umax, "cfl": umax * a.dt / dx,
               "wall_s": time.perf_counter() - tr0}
        series.append(rec)
        if len(series) % 5 == 0:
            print(json.dumps(rec), flush=True)
        for tp in a.preview:
            if abs(s.t - tp) < 0.5 * a.sample and tp not in previews:
                previews[tp] = (Tg.copy(), float(s.t))
        if rec["wall_s"] > a.budget:
            stop = "wall budget"
            break
    wall = time.perf_counter() - tr0
    ts = np.array([r["t"] for r in series])
    late = ts >= ts[-1] - a.avg_window
    diag = {k: float(np.mean([r[k] for r, l in zip(series, late) if l]))
            for k in ("Nu_vol", "Nu_bottom", "Nu_top")}
    diag.update({"avg_window": [float(ts[late].min()), float(ts[-1])],
                 "free_fall_times": float(ts[-1] * np.sqrt(a.Ra * a.Pr)), "stop": stop,
                 "note": "time averages over a short window of a single realization "
                         "(indicative, not a converged Nu)"})
    print("diagnostics:", json.dumps(diag), flush=True)
    Tg = s.grid("T")
    (out / "rb_series.json").write_text(json.dumps({"series": series, "diagnostics": diag}))
    np.savez_compressed(out / "rb_fields.npz", T=Tg, u=s.grid("u"), x=xg, z=Zg, t=s.t)

    meta = {"title": "Rayleigh–Bénard convection",
            "script": "web/gallery/rayleigh_benard.py",
            "example_script": "docs/site/tutorials/code/t02b_rayleigh_benard_gpu.py",
            "geometry": "cartesian", "domain": "convection",
            "parameters": {"Ra": a.Ra, "Pr": a.Pr, "Lx": a.Lx, "amp": a.amp, "seed": a.seed},
            "numerics": {"Nx": a.Nx, "Nz": a.Nz, "bases": "RealFourier x Chebyshev",
                         "dealias": a.dealias, "scheme": a.scheme, "dt": a.dt,
                         "backend": a.backend, "f_backend": "fused", "steps": s.iteration,
                         "wall_s": wall, "build_s": build_s,
                         "cfl_max": max(r["cfl"] for r in series)},
            "diagnostics": diag}
    (out / "rb_meta.json").write_text(json.dumps(meta, indent=1))
    render(out, meta, Tg, Zg, xg, float(s.t), previews, a.clip)


def render(out, meta, Tg, Zg, xg, t_final, previews, clip):
    """Images of the total temperature ("smoke"): 4:1 full box (published) and a 2:1 crop preview.
    The grey range is symmetric, [-A, A] with A the clip-percentile of |T| in the interior
    |z| < 0.4: the well-mixed bulk sits near T = 0 and the boundary layers reach +-1/2, so the
    walls saturate (white bottom, black top) and the grey levels are spent on the plumes."""
    P, N = meta["parameters"], meta["numerics"]
    Lx = P["Lx"]
    order = np.argsort(Zg)
    zg = Zg[order]
    W, H = 1600, int(round(1600 / Lx))
    frames = [("", Tg, t_final)] + [(f"-preview-t{tp:g}", g, tt) for tp, (g, tt) in previews.items()]
    hw = "one L40S GPU" if N["backend"] == "cuda" else "4 CPU cores"
    for tag, G, tt in frames:
        A = float(np.percentile(np.abs(G[:, np.abs(Zg) < 0.4]), clip))
        px = C.resample(G[:, order].T, xg, zg, W, H, periodic_x=Lx, y_range=(-0.5, 0.5))
        rgb, sc = C.field_image(px, "smoke", vmin=-A, vmax=A)
        m = dict(meta, colormap="smoke", t=tt, scale=sc,
                 alt="A long horizontal slab of convecting fluid on black: bright hot plumes rise "
                     "from the bottom wall and dark cold plumes sink from the top, carried by "
                     "large convection rolls across a nearly uniform grey interior.",
                 caption=f"Temperature in 2D Rayleigh–Bénard convection at Ra = {P['Ra']:.0e}".replace("e+0", "e") + f", Pr = "
                         f"{P['Pr']:g}, aspect ratio {Lx:g}, rigid walls, t = {tt:.4f} thermal times "
                         f"(≈ {tt * np.sqrt(P['Ra'] * P['Pr']):.0f} free-fall times); {N['Nx']} × "
                         f"{N['Nz']} RealFourier × Chebyshev, 3/2 dealiasing, {N['scheme']}, dt = "
                         f"{N['dt']:g}, {hw}.")
        C.save(rgb, f"rayleigh-benard{tag}", out, m)
        if tag == "":                              # 2:1 crop preview (half the box)
            px2 = C.resample(G[:, order].T, xg, zg, W, W // 2, periodic_x=Lx,
                             x_range=(0, 2.0), y_range=(-0.5, 0.5))
            rgb2, sc2 = C.field_image(px2, "smoke", vmin=-A, vmax=A)
            C.save(rgb2, "rayleigh-benard-preview-crop", out, dict(m, scale=sc2))


def render_only(out, clip):
    """Re-render from the fields saved by this script in the run directory (no time stepping):
    plotting from scratch/output/<run_id>, allowed on the login node."""
    d = np.load(out / "rb_fields.npz")
    mp = out / "rb_meta.json"
    if mp.exists():
        meta = json.loads(mp.read_text())
    else:                                          # runs made before rb_meta.json existed
        meta = json.loads((out / "rayleigh-benard.meta.json").read_text())
        for k in ("slug", "width", "height", "files", "run_id", "t", "scale", "colormap", "alt",
                  "caption"):
            meta.pop(k, None)
    render(out, meta, d["T"], d["z"], d["x"], float(d["t"]), {}, clip)

if __name__ == "__main__":
    main()
