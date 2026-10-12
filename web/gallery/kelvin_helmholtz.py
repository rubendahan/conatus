"""Website gallery: Kelvin–Helmholtz roll-up of two tanh shear layers (nonlinear IVP, GPU).

Doubly periodic box x in [0, Lx), z in [-1, 1), streamfunction form as in
python/conatus/ecs/kolmogorov.py (u = psi_z, w = -psi_x, vorticity omega = -lap(psi)):

    dt q - nu lap q + tau = -psi_z q_x + psi_x q_z,     q = lap(psi),   int psi = 0 (gauge tau)
    dt c - kappa lap c    = -psi_z c_x + psi_x c_z                       (passive dye)

Base flow U(z) = tanh((z - 1/2)/delta) - tanh((z + 1/2)/delta) + 1 (U = +1 outside, -1 in the
central band: two layers of half-thickness delta, velocity jump 2, Re = U delta / nu). The dye
c = 1 marks the central band. Perturbation: the x-mode closest to the most unstable inviscid
wavenumber of Michalke (1964), alpha delta = 0.4449, localized on both layers, plus a small seeded
low-mode noise so that the two layers do not stay mirror images.

Every numerical choice is an argument (resolution, dealiasing, scheme, dt, t_end). Writes the
fields at the image time in $CN_OUT/kh_fields.npz and two images (signed vorticity; dye in smoke
with one fluorescein iso-line c = 1/2) through web/gallery/brand_cmaps.py.

"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand_cmaps as C  # noqa: E402

from conatus.expert import (IVP, CartesianCoordinates, Domain, Field, IVPSolver, Parameter,  # noqa: E402
                         RealFourier, diff, discretize, dt, integ, lap)

MICHALKE_ALPHA_DELTA = 0.44492          # examples/README.md (Conatus, L = 20, N = 384)


def lncosh(a):
    """log cosh(a) without overflow."""
    return np.logaddexp(a, -a) - np.log(2.0)


def build(Nx: int, Nz: int, Lx: float, delta: float, nu: float, kappa: float, dealias: float):
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    D = Domain(cs, [RealFourier(x, Nx, (0, Lx), dealias=dealias),
                    RealFourier(z, Nz, (-1, 1), dealias=dealias)])
    psi, c, tau = Field("psi", D), Field("c", D), Field("tau", Domain(cs, []))
    nu_, ka_ = Parameter("nu", nu), Parameter("kappa", kappa)
    q = lap(psi)
    pb = IVP([psi, c, tau], name="Kelvin-Helmholtz (two tanh layers, streamfunction)")
    pb.add_equation((dt(q) - nu_ * lap(q) + tau, -diff(psi, z) * diff(q, x) + diff(psi, x) * diff(q, z)))
    pb.add_equation((dt(c) - ka_ * lap(c), -diff(psi, z) * diff(c, x) + diff(psi, x) * diff(c, z)))
    pb.add_equation((integ(psi), 0))
    return discretize(pb), psi, c


def initial(Lx, delta, n_mode, amp, noise, seed):
    """psi and c as callables of the grids (X, Z)."""
    rng = np.random.default_rng(seed)
    kx = 2 * np.pi * n_mode / Lx
    # seeded noise: low x-modes 1..2 n_mode with random phases, on both layers
    modes = [(2 * np.pi * j / Lx, rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi),
              rng.normal()) for j in range(1, 2 * n_mode + 1)]

    def env(Z):
        return np.exp(-((Z - 0.5) / (2 * delta)) ** 2), np.exp(-((Z + 0.5) / (2 * delta)) ** 2)

    def psi0(X, Z):
        base = delta * lncosh((Z - 0.5) / delta) - delta * lncosh((Z + 0.5) / delta) + Z
        eu, el = env(Z)
        pert = amp * delta * np.cos(kx * X) * (eu + el)
        for k, pu, pl, a in modes:
            pert = pert + noise * delta * a * (np.cos(k * X + pu) * eu + np.cos(k * X + pl) * el)
        return base + pert

    def c0(X, Z):
        return 0.5 * (np.tanh((Z + 0.5) / delta) - np.tanh((Z - 0.5) / delta)) + 0 * X
    return psi0, c0, kx


def spectral_fields(psi_g, Lx, Lz):
    """Vorticity -lap(psi) and velocity (psi_z, -psi_x) on the uniform periodic grid (FFT)."""
    Nx, Nz = psi_g.shape
    kx = 2 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)[:, None]
    kz = 2 * np.pi * np.fft.fftfreq(Nz, Lz / Nz)[None, :]
    P = np.fft.fft2(psi_g)
    om = np.real(np.fft.ifft2((kx ** 2 + kz ** 2) * P))
    u = np.real(np.fft.ifft2(1j * kz * P))
    w = np.real(np.fft.ifft2(-1j * kx * P))
    return om, u, w


RK443_IMAG_LIMIT = 1.57   # |R(iy)| <= 1 for y <= 1.5699 (explicit stages of conatus RK443)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=1024)
    ap.add_argument("--Nz", type=int, default=512)
    ap.add_argument("--Lx", type=float, default=4.0)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--Re", type=float, default=500.0, help="U delta / nu")
    ap.add_argument("--Sc", type=float, default=1.0, help="nu / kappa")
    ap.add_argument("--n-mode", type=int, default=6, help="x-mode of the imposed perturbation")
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=34)
    ap.add_argument("--dealias", type=float, default=1.5)
    ap.add_argument("--scheme", default="RK443")
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--t-image", type=float, nargs="+", default=[6.0],
                    help="times rendered; the first is the published image, others are previews")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--sample", type=float, default=0.1, help="growth-rate sampling interval")
    a = ap.parse_args()
    out = Path(os.environ.get("CN_OUT", "."))
    Lz = 2.0
    nu = a.delta / a.Re
    kappa = nu / a.Sc
    t0 = time.perf_counter()
    plan, psi, c = build(a.Nx, a.Nz, a.Lx, a.delta, nu, kappa, a.dealias)
    kw = dict(backend=a.backend, f_backend="fused")
    s = IVPSolver(plan, a.scheme, a.dt, **kw)
    psi0, c0, kx = initial(a.Lx, a.delta, a.n_mode, a.amp, a.noise, a.seed)
    s.set_state({psi: psi0, c: c0})
    print(s.explain(), flush=True)
    build_s = time.perf_counter() - t0
    xg = np.arange(a.Nx) * a.Lx / a.Nx
    zg = -1 + np.arange(a.Nz) * Lz / a.Nz
    dx = a.Lx / a.Nx

    run = (lambda n: s.run_graph(n)) if a.backend == "cuda" else (lambda n: s.run(n))
    block = int(round(a.sample / a.dt))
    times = sorted(a.t_image)
    series, snaps = [], {}
    warned = False
    t_run0 = time.perf_counter()
    while s.t < times[-1] - 1e-9:
        run(block)
        P = s.grid("psi")
        if not np.all(np.isfinite(P)):
            raise FloatingPointError(f"non-finite psi at t = {s.t:g}")
        Ph = np.fft.rfft(P, axis=0)
        amp_mode = float(np.abs(Ph[a.n_mode]).max() / a.Nx)
        om, u, w = spectral_fields(P, a.Lx, Lz)
        umax = float(np.sqrt(u ** 2 + w ** 2).max())
        # largest advective eigenvalue of the explicit part, |lambda| dt <= pi dt (|u|/dx +
        # |w|/dz) (Fourier, k_max = pi/dx); RK443's explicit stages are stable on the imaginary axis
        # up to 1.57 only (computed from the tableau). Reported, never acted on (dt is the user's).
        adv = float(np.pi * a.dt * (np.abs(u).max() / dx + np.abs(w).max() / (Lz / a.Nz)))
        series.append({"t": s.t, "psi_mode_amp": amp_mode, "umax": umax,
                       "cfl": umax * a.dt / dx, "adv_lambda_dt": adv})
        if a.scheme == "RK443" and adv > RK443_IMAG_LIMIT and not warned:
            warned = True
            print(f"WARNING t = {s.t:g}: advective |lambda| dt = {adv:.2f} > {RK443_IMAG_LIMIT} "
                  "(RK443 explicit stability on the imaginary axis): expect divergence unless "
                  "viscosity damps the highest modes; reduce dt", flush=True)
        if len(series) % 10 == 0:
            print(json.dumps(series[-1]), flush=True)
        for ti in times:
            if abs(s.t - ti) < 0.5 * a.dt * block and ti not in snaps:
                snaps[ti] = (om, s.grid("c").copy(), P.copy(), float(s.t))
    wall = time.perf_counter() - t_run0
    steps = s.iteration
    # linear growth rate of the imposed mode: fit log amplitude over the window where it grows
    # exponentially (amplitude between 3x and 300x its initial value)
    ts = np.array([r["t"] for r in series])
    A = np.array([r["psi_mode_amp"] for r in series])
    win = (A > 3 * A[0]) & (A < 300 * A[0])
    first_stop = np.argmax(A > 300 * A[0]) if np.any(A > 300 * A[0]) else len(A)
    win[first_stop:] = False
    growth = float(np.polyfit(ts[win], np.log(A[win]), 1)[0]) if win.sum() >= 3 else None
    diag = {"growth_rate_mode": growth, "growth_rate_times_delta": growth * a.delta if growth else None,
            "alpha_delta_imposed": kx * a.delta, "growth_window": [float(ts[win].min()), float(ts[win].max())]
            if win.sum() else None,
            "note": "viscous (Re = %g), two interacting layers, finite amplitude window: a check of "
                    "order of magnitude against the inviscid single-layer value, not a validation" % a.Re}
    print("diagnostics:", json.dumps(diag), flush=True)
    (out / "kh_series.json").write_text(json.dumps({"series": series, "diagnostics": diag}))

    params = {"Lx": a.Lx, "Lz": Lz, "delta": a.delta, "Re": a.Re, "nu": nu, "kappa": kappa, "Sc": a.Sc,
              "n_mode": a.n_mode, "amp": a.amp, "noise": a.noise, "seed": a.seed}
    numerics = {"Nx": a.Nx, "Nz": a.Nz, "bases": "RealFourier x RealFourier", "dealias": a.dealias,
                "scheme": a.scheme, "dt": a.dt, "backend": a.backend, "f_backend": "fused",
                "steps": steps, "wall_s": wall, "build_s": build_s,
                "cfl_max": max(r["cfl"] for r in series)}
    validated = {"value": "ω_i max = 0.189702 at α = 0.44492 (inviscid tanh layer, L = 20, N = 384)",
                 "reference": "Michalke 1964, JFM 19, 543–556, Table 1 (0.1897 at α = 0.4446)",
                 "source": "paper/conatus.tex §Gallery; examples/README.md", "run_id": None}
    common = {"script": "web/gallery/kelvin_helmholtz.py",
              "example_script": "examples/shear/kelvin_helmholtz.py", "validated": validated,
              "geometry": "cartesian", "domain": "shear", "parameters": params, "numerics": numerics,
              "diagnostics": diag}
    W, H = 1600, int(round(1600 * Lz / a.Lx))
    for i, ti in enumerate(times):
        om, cg, P, t_snap = snaps[ti]
        np.savez_compressed(out / f"kh_fields_t{ti:g}.npz", omega=om, c=cg, psi=P, x=xg, z=zg, t=t_snap)
        om_px = C.resample(om.T, xg, zg, W, H, periodic_x=a.Lx, periodic_y=Lz, y_range=(-1, 1))
        c_px = C.resample(cg.T, xg, zg, W, H, periodic_x=a.Lx, periodic_y=Lz, y_range=(-1, 1))
        rgb_o, sc_o = C.field_image(om_px, "signed")
        rgb_c, sc_c = C.field_image(c_px, "smoke", vmin=0.0, vmax=1.0)
        rgb_c = C.overlay_line(rgb_c, C.contour_mask(c_px, 0.5, 1.2), alpha=0.85)
        tag = "" if ti == a.t_image[0] else f"-preview-t{ti:g}"
        res = f"{a.Nx} × {a.Nz} RealFourier modes (3/2 dealiasing)"
        C.save(rgb_o, f"kelvin-helmholtz-vorticity{tag}", out, dict(
            common, title="Kelvin–Helmholtz billows",
            alt="Two horizontal shear layers rolled up into rows of spiral vortices on a black "
                "background: the upper row in red (negative vorticity), the lower row in green "
                "(positive vorticity), thin filaments wrapping around each core.",
            caption=f"Vorticity of two tanh shear layers (half-thickness δ = {a.delta:g}, Re = Uδ/ν = "
                    f"{a.Re:g}) at t = {t_snap:.1f}, nonlinear IVP seeded by the box mode "
                    f"nearest the most unstable one (αδ = {kx * a.delta:.3f}); {res}, streamfunction form, {a.scheme}, dt = {a.dt:g}, "
                    f"{'one L40S GPU' if a.backend == 'cuda' else '4 CPU cores'}.",
            colormap="signed", t=t_snap, scale=sc_o))
        C.save(rgb_c, f"kelvin-helmholtz-dye{tag}", out, dict(
            common, title="Kelvin–Helmholtz dye",
            alt="A pale horizontal band of passive dye between two shear layers, its edges curled "
                "into rows of spirals that entrain dark fluid, outlined by one thin fluorescent "
                "yellow-green line at half concentration.",
            caption=f"Passive dye marking the central band (Schmidt number {a.Sc:g}) in the same "
                    f"Kelvin–Helmholtz run at t = {t_snap:.1f}; the line is the iso-concentration "
                    f"c = 1/2. {res}, {a.scheme}, dt = {a.dt:g}.",
            colormap="smoke", t=t_snap, scale=sc_c))


if __name__ == "__main__":
    main()
