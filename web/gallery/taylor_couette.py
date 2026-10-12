"""Gallery image: the critical Taylor-vortex mode between rotating cylinders, eta = 0.5 (signed).

Circular Couette flow V(r) = A r + B / r (inner cylinder rotating, outer at rest, gap d = 1,
inner wall speed 1), axisymmetric (m = 0) perturbations of axial wavenumber k, linearised
Navier--Stokes in primitive variables on an AnnulusBasis, as in
docs/site/tutorials/code/t08_polar.py. The neutral Reynolds number at k = k_c is found by Brent's
method on the growth rate (xtol declared), and the mode at that Re is drawn in the meridional
(z, r) plane over two axial wavelengths at true aspect ratio: azimuthal velocity u_phi (signed
colormap, z to the right, inner cylinder at the bottom) with six silver streamlines of the
meridional circulation, psi = i r u_r / k (u_r = -(1/r) d psi / dz). Radial profiles come from
the Chebyshev coefficients (AnnulusOps.point_row) on 801 radii, interpolated to the pixel rows.

"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand_cmaps as C  # noqa: E402

from conatus.expert import (EVP, AnnulusBasis, ComplexFourier, CylindricalCoordinates,  # noqa: E402
                         Domain, Field, Parameter, VectorField, discretize, div, dot, eigenvalues,
                         grad, integ, interp, lap, lift)

SLUG = "taylor-couette-mode"
ETA = 0.5
RI, RO = ETA / (1 - ETA), 1 / (1 - ETA)
B = 1.0 / (1.0 / RI - RI / RO ** 2)
A = -B / RO ** 2
cc = CylindricalCoordinates("phi", "r", "z")
phi, r, z = cc["phi"], cc["r"], cc["z"]
s = Parameter("s")


def couette(Re, k, N):
    """Tutorial 8's Taylor--Couette EVP; returns (plan, group, result)."""
    br = AnnulusBasis(r, N, (RI, RO), dealias=1)
    D = Domain(cc, [ComplexFourier(phi, 4, (0, 2 * np.pi), dealias=1), br,
                    ComplexFourier(z, 4, (0, 2 * np.pi / k), dealias=1)])
    Db = D.without(r)
    u, p = VectorField("u", D, dtype=complex), Field("p", D, dtype=complex)
    t1, t2 = VectorField("t1", Db, dtype=complex), VectorField("t2", Db, dtype=complex)
    tp = Field("tp", Domain(cc, []), dtype=complex)
    U0 = VectorField("U0", Domain(cc, [br]))
    pb = EVP([u, p, t1, t2, tp], eigenvalue=s, name="Taylor-Couette")
    pb.add_equation((s * u + dot(U0, grad(u)) + dot(u, grad(U0)) + grad(p) - lap(u) / Re
                     + lift(t1, br, -1, lam=2) + lift(t2, br, -2, lam=2), 0))
    pb.add_equation((div(u) + tp, 0))
    pb.add_equation((interp(u, r, "left"), 0))
    pb.add_equation((interp(u, r, "right"), 0))
    pb.add_equation((integ(p), 0))

    def base(P, R, Z):
        zero = 0 * (P + R + Z)
        return np.array([A * R + B / R + zero, zero, zero])
    plan = discretize(pb, ncc={U0: base})
    g = plan.group(0, k)
    return plan, g, eigenvalues(plan, g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--k", type=float, default=3.1625)
    ap.add_argument("--xtol", type=float, default=1e-9)
    a = ap.parse_args()
    out = Path(os.environ.get("CN_OUT", "."))
    grow = lambda Re, N=a.N: couette(Re, a.k, N)[2].values.real.max()  # noqa: E731
    Re_c = brentq(grow, 60.0, 80.0, xtol=a.xtol)
    Re_c_fine = brentq(lambda Re: grow(Re, a.N * 3 // 2), 60.0, 80.0, xtol=a.xtol)
    plan, g, res = couette(Re_c, a.k, a.N)
    i = int(np.argmax(res.values.real))
    lam = res.values[i]
    print(f"Re_c(k = {a.k}) = {Re_c:.6f} (N = {a.N}), {Re_c_fine:.6f} (N = {a.N * 3 // 2}); "
          f"s = {lam:.3e}, residual {res.residuals[i]:.1e}")
    v, ops = res.vectors[:, i], plan.space.ops
    rr = np.linspace(RI, RO, 801)

    def comp(c, kk):
        (off, sz), = [(o, n) for vi, cm, o, n in plan.var_slots(g) if vi == 0 and cm == (c,)]
        return np.array([ops.point_row(kk, 0, 0, x, 0, sz) for x in rr]) @ v[off:off + sz]
    um, up = comp(0, -1), comp(1, 1)                   # spin components u_-+ = u_r -+ i u_phi
    uphi, ur = (up - um) / 2j, (up + um) / 2
    scale = uphi[np.argmax(np.abs(uphi))]
    uphi, ur = uphi / scale, ur / scale
    psi = 1j * rr * ur / a.k
    np.savez(out / f"{SLUG}.npz", r=rr, u_phi=uphi, u_r=ur, psi=psi, k=a.k, Re_c=Re_c)

    lam_z = 2 * np.pi / a.k
    W = 1600
    H = int(round(W * (RO - RI) / (2 * lam_z)))
    zz = (np.arange(W) + 0.5) / W * 2 * lam_z
    rp = RO - (np.arange(H) + 0.5) / H * (RO - RI)       # row 0 = outer cylinder (top)

    def grid(f):
        fr = np.interp(rp, rr, f.real) + 1j * np.interp(rp, rr, f.imag)
        return np.real(fr[:, None] * np.exp(1j * a.k * zz[None, :]))
    F, P = grid(uphi), grid(psi)
    rgb, sc = C.field_image(F, "signed", vmax=1.0, grain_sigma=0)
    Pm = np.abs(P).max()
    lines = sum(C.contour_mask(P, lv * Pm, width_px=1.0) for lv in (-0.75, -0.45, -0.15, 0.15, 0.45, 0.75))
    rgb = C.grain(C.overlay_line(rgb, np.clip(lines, 0, 1), color="#EDEBE6", alpha=0.35), 1.6)
    meta = {
        "title": "Taylor–Couette: the first Taylor vortices",
        "alt": ("A horizontal band of alternating red and green vertical stripes fading to black at "
                "the top and bottom walls: the azimuthal velocity of Taylor vortices between two "
                "cylinders, with thin silver closed streamlines circling the dark vortex centres."),
        "caption": (f"Critical axisymmetric mode of circular Couette flow, radius ratio η = 0.5, outer "
                    f"cylinder at rest: azimuthal velocity and meridional streamlines over two "
                    f"wavelengths at Re_c = {Re_c:.4f}, k = {a.k}. Fourier × Chebyshev AnnulusBasis "
                    f"N = {a.N}, dense QZ, Brent on the growth rate."),
        "colormap": "signed",
        "script": "web/gallery/taylor_couette.py",
        "example_script": "docs/site/tutorials/code/t08_polar.py",
        "validated": {
            "value": "η = 0.5, axisymmetric: Re_c = 68.1863 at k_c = 3.1625",
            "reference": "Recktenwald, Lücke & Müller 1993, Phys. Rev. E 48, 4444, Table I (68.19, 3.16)",
            "source": "paper/conatus.tex §Verification",
            "run_id": None,
        },
        "geometry": "cylinder",
        "domain": "rotating",
        "parameters": {"eta": ETA, "N": a.N, "k": a.k, "brent_xtol": a.xtol},
        "diagnostics": {"Re_c": Re_c, "Re_c_N_1.5": Re_c_fine, "growth_at_Re_c": [lam.real, lam.imag],
                        "residual": float(res.residuals[i])},
        "scale": sc,
    }
    C.save(rgb, SLUG, out, meta)


if __name__ == "__main__":
    main()
