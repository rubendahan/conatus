"""Gallery image: Swift--Hohenberg rolls growing from noise in a large periodic box (smoke).

    du/dt = r u - (1 + lap)^2 u - u^3            (Swift & Hohenberg 1977), r = 0.1

Same equation, bases and dealiasing as part (c) of examples/patterns/swift_hohenberg.py
(RealFourier x RealFourier, dealias = 2 declared for the cubic term, SBDF2), in a box of
``--nx`` x ``--ny`` roll wavelengths (2 pi each) started from seeded random plane waves of
|k| close to 1: rolls of every orientation grow and form a labyrinth with defects that
coarsens slowly. The field u is drawn in the smoke colormap (dark troughs, light crests); the
pixels are an exact Fourier synthesis on the pixel grid (FFT zero-padding).
The r.m.s. of u is compared with that of the straight rolls of the validated branch,
A / sqrt(2) with A(r = 0.1) = 0.3652433 (diagnostic only: the labyrinth is not straight rolls).

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

from conatus.expert import (IVP, CartesianCoordinates, Domain, Field, IVPSolver, Parameter,  # noqa: E402
                         RealFourier, discretize, dt, lap)

SLUG = "swift-hohenberg"
A_BRANCH = 0.3652433015                  # roll amplitude at r = 0.1 (paper, gallery table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--nx", type=int, default=24, help="box length in roll wavelengths (x)")
    ap.add_argument("--ny", type=int, default=12, help="box length in roll wavelengths (y)")
    ap.add_argument("--ppw", type=int, default=8, help="Fourier modes per wavelength")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--tend", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=34)
    a = ap.parse_args()
    out = Path(os.environ.get("CN_OUT", "."))
    Lx, Ly = 2 * np.pi * a.nx, 2 * np.pi * a.ny
    Nx, Ny = a.ppw * a.nx, a.ppw * a.ny
    cs = CartesianCoordinates("x", "y")
    x, y = cs.coords
    D = Domain(cs, [RealFourier(x, Nx, (0, Lx), dealias=2), RealFourier(y, Ny, (0, Ly), dealias=2)])
    u, r = Field("u", D), Parameter("r", a.r)
    ivp = IVP([u], name="Swift-Hohenberg 2D")
    ivp.add_equation((dt(u) - r * u + (u + lap(u)) + lap(u + lap(u)), -u * u * u))
    s = IVPSolver(discretize(ivp, params={"r": a.r}), "SBDF2", a.dt)
    print(s.explain())

    rng = np.random.default_rng(a.seed)                 # random plane waves, lattice |k| in [0.9, 1.1]
    nxs, nys = np.meshgrid(np.arange(-a.nx * 2, a.nx * 2 + 1), np.arange(0, a.ny * 2 + 1))
    kx, ky = 2 * np.pi * nxs / Lx, 2 * np.pi * nys / Ly
    sel = np.abs(np.hypot(kx, ky) - 1.0) < 0.1
    kx, ky = kx[sel], ky[sel]
    amp, ph = 1e-2 * rng.normal(size=kx.size), 2 * np.pi * rng.random(kx.size)
    s.set_state({u: lambda X, Y: sum(amp[j] * np.cos(kx[j] * X + ky[j] * Y + ph[j])
                                     for j in range(kx.size))})
    t0 = time.perf_counter()
    nsteps = int(round(a.tend / a.dt))
    s.run(nsteps - 1)
    before = s.grid("u")
    s.run(1)
    g = s.grid("u")
    wall = time.perf_counter() - t0
    rate = float(np.abs(g - before).max() / a.dt)
    rms = float(np.sqrt(np.mean(g ** 2)))
    print(f"t = {s.t:g} in {wall:.1f} s: rms u = {rms:.6f} (straight rolls: A/sqrt2 = "
          f"{A_BRANCH / np.sqrt(2):.6f}), max|du/dt| = {rate:.1e}, max|u| = {np.abs(g).max():.4f}")
    if not np.all(np.isfinite(g)):
        raise FloatingPointError("non-finite state")
    np.savez(out / f"{SLUG}.npz", u=g, t=s.t, Lx=Lx, Ly=Ly)

    W = 1600
    H = int(round(W * Ly / Lx))
    G = np.fft.rfft2(g.T)                               # (Ny, Nx//2+1), y first (image rows)
    P = np.zeros((H, W // 2 + 1), complex)
    hy = Ny // 2
    P[:hy, :G.shape[1] - 1] = G[:hy, :-1]
    P[-hy + 1:, :G.shape[1] - 1] = G[-hy + 1:, :-1]
    img = np.fft.irfft2(P, s=(H, W)) * (H * W) / (Nx * Ny)
    img = img[::-1]                                     # row 0 = top = y -> Ly
    lim = float(np.abs(img).max())
    rgb, sc = C.field_image(img, "smoke", vmin=-lim, vmax=lim)
    meta = {
        "title": "Swift–Hohenberg labyrinth",
        "alt": ("A dense labyrinth of light and dark parallel stripes of fixed width, curving in "
                "patches of different orientations that meet at defects and grain boundaries."),
        "caption": (f"Rolls of du/dt = r u − (1 + ∇²)² u − u³ at r = {a.r:g} grown from noise in a periodic "
                    f"box of {a.nx} × {a.ny} roll wavelengths, t = {s.t:g}. RealFourier × RealFourier "
                    f"{Nx} × {Ny} (dealias 2 for the cubic term), SBDF2, dt = {a.dt:g}."),
        "colormap": "smoke",
        "script": "web/gallery/swift_hohenberg.py",
        "example_script": "examples/patterns/swift_hohenberg.py",
        "validated": {
            "value": ("roll amplitude A(r = 0.1) = 0.3652433015 (N-converged, Newton residual 8e-14); "
                      "(A² − 4r/3)/r² → 1/144 along the continued branch; 2D IVP on the branch to 1e-9"),
            "reference": "Swift & Hohenberg 1977, Phys. Rev. A 15, 319; Cross & Hohenberg 1993, Rev. Mod. Phys. 65, 851",
            "source": "paper/conatus.tex §Gallery; examples/README.md",
            "run_id": None,
        },
        "geometry": "cartesian",
        "domain": "patterns",
        "parameters": {"r": a.r, "Lx": Lx, "Ly": Ly, "Nx": Nx, "Ny": Ny, "dealias": 2,
                       "scheme": "SBDF2", "dt": a.dt, "tend": a.tend, "seed": a.seed},
        "diagnostics": {"rms_u": rms, "rolls_rms": A_BRANCH / np.sqrt(2), "max_dudt": rate,
                        "wall_s": wall},
        "scale": sc,
    }
    C.save(rgb, SLUG, out, meta)


if __name__ == "__main__":
    main()
