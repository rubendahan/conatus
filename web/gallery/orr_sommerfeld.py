"""Gallery image: the Tollmien--Schlichting mode of plane Poiseuille flow, Re = 1e4, alpha = 1 (signed).

Orr--Sommerfeld EVP of docs/site/tutorials/code/t01_orr_sommerfeld.py (Chebyshev N = 64, tau
modes rejected by the N = 96 drift test). The least stable mode v(y) (Chebyshev coefficients of
the eigenvector, checked against the four wall conditions) gives the streamwise velocity perturbation from continuity
(i alpha u + v' = 0):  u_hat = i v' / alpha,
drawn as Re(u_hat(y) e^{i alpha x}) over one wavelength 2 pi / alpha in x (to the right) and
the channel -1 <= y <= 1 (upwards), at true aspect ratio. v and its derivatives are evaluated
exactly (Clenshaw) on the pixel rows. Normalisation: max |u_hat| real and positive.
(The vorticity was tried first: its wall layers hide the interior at a linear scale.)

"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as Ch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brand_cmaps as C  # noqa: E402

from conatus.expert import EVP, Parameter, channel, diff, discretize, eigenvalues, interp  # noqa: E402

SLUG = "orr-sommerfeld-mode"


def orr_sommerfeld(N, Re, alpha):
    """Tutorial 1's Orr--Sommerfeld EVP (v(y), U = 1 - y^2, clamped walls)."""
    c = Parameter("c", dtype=complex)
    ch = channel(N, base="poiseuille")
    v, y, U = ch.field("v"), ch.y, ch.U
    Re_, a = Parameter("Re", Re), Parameter("alpha", alpha)

    def L2(f):
        return diff(diff(f, y), y) - a ** 2 * f
    pb = EVP([v], eigenvalue=c, name="Orr-Sommerfeld")
    pb.add_equation((L2(L2(v)) / (1j * a * Re_) - U * L2(v) + diff(diff(U, y), y) * v + c * L2(v), 0))
    pb.add_bc(interp(v, y, -1), interp(v, y, 1), interp(diff(v, y), y, -1), interp(diff(v, y), y, 1))
    return discretize(pb, ncc=ch.ncc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--Re", type=float, default=1e4)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(os.environ.get("CN_OUT", "."))
    res = eigenvalues(orr_sommerfeld(a.N, a.Re, a.alpha),
                      drift_plan=orr_sommerfeld(a.N * 3 // 2, a.Re, a.alpha), drift_tol=1e-6)
    i = int(res.values.imag.argmax())
    cval = res.values[i]
    print(res.explain(key="imag"))
    cf = res.vectors[:a.N, i]
    walls = np.abs(np.concatenate([Ch.chebval([-1, 1], cf), Ch.chebval([-1, 1], Ch.chebder(cf))])).max()
    print(f"c = {cval:.10f}, backward residual {res.residuals[i]:.1e}, "
          f"max wall defect |v|, |v'| = {walls / np.abs(cf).max():.1e} (relative)")

    W = 1600
    lam = 2 * np.pi / a.alpha
    H = int(round(W * 2.0 / lam))
    yp = 1.0 - (np.arange(H) + 0.5) * 2.0 / H            # row 0 = upper wall
    xp = (np.arange(W) + 0.5) * lam / W
    uh = 1j * Ch.chebval(yp, Ch.chebder(cf)) / a.alpha
    uh = uh / uh[np.argmax(np.abs(uh))]
    F = np.real(uh[:, None] * np.exp(1j * a.alpha * xp[None, :]))
    np.savez(out / f"{SLUG}.npz", y=yp, u_hat=uh, v_cheb=cf, c=cval)
    rgb, sc = C.field_image(F, "signed", vmax=1.0)
    meta = {
        "title": "Tollmien–Schlichting wave in a channel",
        "alt": ("A wide band of red and green lobes on black: the streamwise velocity of the least "
                "stable wave of plane channel flow, strongest in bright layers just inside the top and "
                "bottom walls, the upper and lower halves of opposite sign, with a sharp phase kink "
                "near each wall."),
        "caption": (f"Streamwise velocity perturbation of the Orr–Sommerfeld mode of plane Poiseuille flow at "
                    f"Re = {a.Re:g}, α = {a.alpha:g}, one wavelength: c = {cval.real:.8f} + "
                    f"{cval.imag:.8f}i. Chebyshev N = {a.N}, dense QZ, tau modes removed by an "
                    f"N = {a.N * 3 // 2} drift test."),
        "colormap": "signed",
        "script": "web/gallery/orr_sommerfeld.py",
        "example_script": "docs/site/tutorials/code/t01_orr_sommerfeld.py",
        "geometry": "cartesian",
        "domain": "shear",
        "parameters": {"N": a.N, "Re": a.Re, "alpha": a.alpha},
        "diagnostics": {"c": [float(cval.real), float(cval.imag)], "residual": float(res.residuals[i]),
                        "wall_defect_rel": float(walls / np.abs(cf).max())},
        "scale": sc,
    }
    C.save(rgb, SLUG, out, meta)


if __name__ == "__main__":
    main()
