"""Gallery: Rossby-Haurwitz wave (wavenumber 4) in shallow water on the rotating sphere.

The problem is the one of benchmarks/spherical/rossby_haurwitz.py (Williamson et al. 1992, case 6;
initial data, constants and equations imported or repeated verbatim): inviscid shallow water,
Coriolis implicit, RK443, T63 and dt = 0.03 / Omega as in the validation run
. It is integrated for 14 days, then the relative vorticity
zeta = curl(u) is evaluated spectrally and synthesized exactly on a fine latitude-longitude grid
(2 Re sum_{m >= 0} a_m(cos theta) e^{i m phi} - a_0; this synthesis matches the closed-form initial
vorticity of the wave to 1e-14, checked at t = 0 below), and shown in orthographic projection
seen from 35 deg N. Signed scale: cyclonic (northern) positive, zero black.

    python web/gallery/rossby_haurwitz.py [--L 63 --dt 0.03 --days 14]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import brand_cmaps as C  # noqa: E402

import conatus.expert as CN  # noqa: E402
from conatus.discretize import discretize  # noqa: E402

SLUG = "rossby-haurwitz"


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "rossby_haurwitz_bench", ROOT / "benchmarks" / "spherical" / "rossby_haurwitz.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synth(S, z: np.ndarray, colat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Exact real synthesis of a scalar (m, l) coefficient array: (len(colat), len(lon))."""
    x = np.cos(colat)
    a = np.array([S.H.values(int(m), 0, x) @ z[i] for i, m in enumerate(S.ms)])   # (nm, nx)
    return (2 * np.real(np.exp(1j * np.outer(lon, S.ms)) @ a) - np.real(a[0])[None, :]).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=63)
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--view-lat", type=float, default=35.0)
    a = ap.parse_args()
    out = Path(os.environ["CN_OUT"])
    rh = load_driver()
    s2 = CN.S2Coordinates("phi", "theta")
    bt = CN.SphereBasis(s2["theta"], a.L + 1, radius=1.0, dealias=3 / 2)
    Np = 2 * (a.L + 1)
    D = CN.Domain(s2, [CN.RealFourier(s2["phi"], Np, (0, 2 * np.pi), dealias=3 / 2), bt])
    u, h = CN.VectorField("u", D), CN.Field("h", D)
    Cz = CN.TensorField("Cz", CN.Domain(s2, [bt]), 2)
    pb = CN.IVP([u, h])                                       # same equations as the benchmark
    pb.add_equation((CN.dt(u) + CN.grad(h) + CN.dot(Cz, u), -CN.dot(u, CN.grad(u))))
    pb.add_equation((CN.dt(h) + rh.H0 * CN.div(u), -CN.div((h - rh.H0) * u)))

    def cz(P, T):
        c = 2 * np.cos(T) + 0 * P
        return np.array([[0 * c, c], [-c, 0 * c]])
    plan = discretize(pb, ncc={Cz: cz})
    s = CN.IVPSolver(plan, "RK443", a.dt)
    s.set_state({u: lambda P, T: np.array(rh.initial(P, T)[:2]), h: lambda P, T: rh.initial(P, T)[2]})
    S = plan.space
    # t = 0 check of the synthesis against the closed-form vorticity of the wave
    th_c = np.linspace(0.05, np.pi - 0.05, 37)
    ph_c = np.linspace(0, 2 * np.pi, 49)[:-1]
    z0 = synth(S, plan.evaluator.evaluate(CN.curl(u), {u: s.coeffs("u")}, {}), th_c, ph_c)
    T, P = np.meshgrid(th_c, ph_c, indexing="ij")
    lat, R = np.pi / 2 - T, rh.RW
    exact = 2 * rh.W * np.sin(lat) - rh.K * np.sin(lat) * np.cos(lat) ** R * (R + 1) * (R + 2) * np.cos(R * P)
    err0 = float(np.abs(z0 - exact).max() / np.abs(exact).max())
    nsteps = int(round(a.days * 86400 * rh.OMEGA / a.dt))
    t0 = time.perf_counter()
    s.run(nsteps)
    wall = time.perf_counter() - t0
    z = plan.evaluator.evaluate(CN.curl(u), {u: s.coeffs("u")}, {})
    ncol, nlon = 721, 1440
    colat = np.linspace(0, np.pi, ncol)
    lon = 2 * np.pi * np.arange(nlon) / nlon
    Z = synth(S, z, colat, lon)                               # (ncol, nlon)
    Hh = synth(S, s.coeffs("h"), colat, lon)                  # height, same exact synthesis
    np.savez_compressed(out / f"{SLUG}_field.npz", zeta=Z.astype(np.float32), h=Hh.astype(np.float32),
                        colat=colat, lon=lon, t=s.t, days=s.t / (rh.OMEGA * 86400),
                        u_coeffs=s.coeffs("u"), h_coeffs=s.coeffs("h"))
    # orthographic projection seen from (view_lat, 0)
    W, H = a.width, a.height
    rad = 0.46 * H
    X = (np.arange(W) + 0.5 - W / 2) / rad
    Y = (H / 2 - np.arange(H) - 0.5) / rad
    XX, YY = np.meshgrid(X, Y)
    rho2 = XX ** 2 + YY ** 2
    inside = rho2 <= 1.0
    ZZ = np.sqrt(np.clip(1 - rho2, 0, None))                  # towards the viewer
    p0 = np.radians(a.view_lat)
    # rotate the view frame back to Earth frame (tilt about the x axis of the image)
    ex = ZZ * np.cos(p0) - YY * np.sin(p0)                    # Earth x (lon 0, equator)
    ez = ZZ * np.sin(p0) + YY * np.cos(p0)                    # Earth z (north)
    ey = XX                                                   # Earth y (lon 90 E)
    c_lat = np.arccos(np.clip(ez, -1, 1))
    c_lon = np.mod(np.arctan2(ey, ex), 2 * np.pi)
    from scipy.ndimage import map_coordinates
    def project(F):
        wrap = np.concatenate([F, F[:, :1]], axis=1)
        f = map_coordinates(wrap, [(c_lat / np.pi * (ncol - 1)).ravel(),
                                   (c_lon / (2 * np.pi) * nlon).ravel()],
                            order=3, mode="nearest").reshape(H, W)
        f[~inside] = np.nan
        return f
    limb = np.clip(1.0 - np.abs(np.sqrt(rho2) - 1.0) * rad / 0.8, 0, 1)
    rgb, scale = C.field_image(project(Z), "signed", vmax=float(np.abs(Z).max()))
    rgb = C.overlay_line(rgb, limb, color="#3A3A38", alpha=0.9)
    days = s.t / (rh.OMEGA * 86400)
    diag = dict(t=float(s.t), days=float(days), steps=nsteps, wall_s=wall,
                synthesis_error_t0=err0, max_abs_zeta=float(np.abs(Z).max()), scale=scale)
    print(json.dumps(diag, indent=1))
    meta = dict(
        title="Rossby–Haurwitz wave on the sphere",
        alt=("A globe seen from mid-northern latitudes on a black background, covered by large "
             "alternating green and red vorticity lobes: four around each hemisphere, the "
             "northern and southern lobes of opposite sign across the equator, with faint "
             "concentric ripples near the pole."),
        caption=(f"Relative vorticity of the wavenumber-4 Rossby–Haurwitz wave after {days:.0f} days "
                 f"(Williamson et al. 1992, test case 6), inviscid shallow water on the rotating sphere, "
                 f"seen from {a.view_lat:g}° N (no hyperdiffusion damps the faint small-scale ripples). "
                 f"Fourier × spin-weighted spherical harmonics, "
                 f"T{a.L} ({Np} × {a.L + 1}), implicit Coriolis, RK443, dt = {a.dt:g}/Ω."),
        colormap="signed", script="web/gallery/rossby_haurwitz.py",
        example_script="benchmarks/spherical/rossby_haurwitz.py",
        validated=dict(value=("14 days: mass exact, energy conserved to 1.4e-7; pattern 7.7% slower "
                              "than the non-divergent Haurwitz speed, converged (T42/T63 within 0.13%)"),
                       reference="Williamson et al. 1992, J. Comput. Phys. 102, 211–224, test case 6; Haurwitz 1940",
                       source="paper/conatus.tex §Verification", run_id=""),
        geometry="sphere", domain="rotating",
        parameters=dict(Lmax=a.L, Nphi=Np, dealias=1.5, scheme="RK443", dt=a.dt, days=a.days, R=rh.RW),
        diagnostics=diag)
    C.save(rgb, SLUG, out, meta)


if __name__ == "__main__":
    main()
