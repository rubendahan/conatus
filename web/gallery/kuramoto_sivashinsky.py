"""Gallery image: space-time diagram of Kuramoto--Sivashinsky chaos, L = 22 (signed colormap).

u_t + u u_x + u_xx + u_xxxx = 0 on (0, 22), periodic; same problem, basis (RealFourier N = 32,
dealias 3/2) and scheme (RK443, dt = 0.01) as docs/site/tutorials/code/t05_ks_orbits.py and the
Studio run. A transient of ``--transient`` time units is discarded,
then u(x, t) is sampled every ``--sample`` over ``--window``; x is synthesized exactly from the
Fourier coefficients (no interpolation in x), t is linearly resampled to the pixel columns.
Image: time runs left to right, x upwards; red u < 0, glass u > 0, black u = 0.

"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand_cmaps as C  # noqa: E402

from conatus.expert import (IVP, CartesianCoordinates, Domain, Field, IVPSolver,  # noqa: E402
                         RealFourier, diff, discretize, dt, lap)

SLUG = "kuramoto-sivashinsky-l22"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=22.0)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--transient", type=float, default=300.0)
    ap.add_argument("--window", type=float, default=250.0)
    ap.add_argument("--sample", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=400)
    a = ap.parse_args()
    out = Path(os.environ.get("CN_OUT", "."))
    L, N = a.L, a.N

    cs = CartesianCoordinates("x")
    x = cs["x"]
    u = Field("u", Domain(cs, [RealFourier(x, N, (0, L), dealias=1.5)]))
    pb = IVP([u], name=f"Kuramoto-Sivashinsky L={L:g}")
    pb.add_equation((dt(u) + lap(u) + lap(lap(u)), -u * diff(u, x)))
    s = IVPSolver(discretize(pb), "RK443", a.dt)
    print(s.explain())

    rng = np.random.default_rng(a.seed)                 # small random low modes, zero mean
    amp, ph = 0.1 * rng.normal(size=6), 2 * np.pi * rng.random(6)
    s.set_state({u: lambda X: sum(amp[j] * np.cos(2 * np.pi * (j + 1) * X / L + ph[j]) for j in range(6))})
    t0 = time.perf_counter()
    s.run(int(round(a.transient / a.dt)))
    per = int(round(a.sample / a.dt))
    nsamp = int(round(a.window / a.sample)) + 1
    coeffs, times = [], []
    for i in range(nsamp):
        if i:
            s.run(per)
        coeffs.append(s.coeffs("u").copy())
        times.append(s.t)
    wall = time.perf_counter() - t0
    Cf = np.array(coeffs)                               # (nt, N): (cos, -sin) pairs
    times = np.array(times)
    mean = Cf[:, 0]
    E = 0.5 * (Cf[:, 0] ** 2 + 0.5 * (Cf[:, 2:] ** 2).sum(1))   # (1/2L) int u^2 dx
    print(f"integrated to t = {s.t:g} in {wall:.1f} s; |mean u| max {np.abs(mean).max():.1e}; "
          f"energy (1/2L)int u^2 in window: mean {E.mean():.4f}, min {E.min():.4f}, max {E.max():.4f}")
    if not np.all(np.isfinite(Cf)):
        raise FloatingPointError("non-finite state")

    # exact Fourier synthesis in x on the pixel rows (row 0 = top = x -> L)
    H, W = a.height, a.width
    xp = L - (np.arange(H) + 0.5) * L / H
    k = 2 * np.pi * np.arange(N // 2) / L
    cosm, sinm = np.cos(np.outer(xp, k)), np.sin(np.outer(xp, k))
    field = cosm @ Cf[:, 0::2].T - sinm @ Cf[:, 1::2].T  # (H, nt)
    tp = times[0] + (np.arange(W) + 0.5) * (times[-1] - times[0]) / W
    img = np.stack([np.interp(tp, times, row) for row in field])
    np.savez(out / f"{SLUG}.npz", coeffs=Cf, t=times, L=L, N=N, dt=a.dt)
    rgb, scale = C.field_image(img, "signed")
    meta = {
        "title": "Kuramoto–Sivashinsky chaos, L = 22",
        "alt": ("Space-time diagram of a chaotic Kuramoto–Sivashinsky solution: time runs left to "
                "right, space vertically; wavy red and green stripes of negative and positive u "
                "merge, split and drift, separated by black zero lines."),
        "caption": (f"u(x, t) of u_t + u u_x + u_xx + u_xxxx = 0 on a periodic domain L = 22, over "
                    f"{a.window:g} time units after a transient of {a.transient:g}. RealFourier N = {N} "
                    f"(dealias 3/2), RK443, dt = {a.dt:g}."),
        "colormap": "signed",
        "script": "web/gallery/kuramoto_sivashinsky.py",
        "example_script": "docs/site/tutorials/code/t05_ks_orbits.py",
        "validated": {
            "value": ("travelling wave TW2 seeded from a live L = 22 run: c = −0.349541, E = 0.604820, "
                      "Newton residual 4.9e-16"),
            "reference": ("Cvitanović, Davidchack & Siminos 2010, SIAM J. Appl. Dyn. Syst. 9, 1–33 "
                          "(velocity 0.34954, energy 0.6048)"),
            "source": "paper/conatus.tex §Live monitoring and control (Studio)",
            "run_id": "",
        },
        "geometry": "cartesian",
        "domain": "dynamics",
        "parameters": {"L": L, "N": N, "dealias": 1.5, "scheme": "RK443", "dt": a.dt,
                       "transient": a.transient, "window": a.window, "sample": a.sample,
                       "seed": a.seed},
        "diagnostics": {"energy_mean": float(E.mean()), "energy_min": float(E.min()),
                        "energy_max": float(E.max()), "mean_u_max": float(np.abs(mean).max()),
                        "t_end": float(s.t), "wall_s": wall},
        "scale": scale,
    }
    C.save(rgb, SLUG, out, meta)


if __name__ == "__main__":
    main()
