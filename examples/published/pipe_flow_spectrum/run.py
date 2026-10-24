"""Rightmost eigenvalues of linearised pipe flow from Re = 10^2 to 10^7 (Meseguer & Trefethen 2003).

Reference: A. Meseguer & L. N. Trefethen (2003), Linearized pipe flow to Reynolds number 10^7,
J. Comput. Phys. 186, 178-197, doi:10.1016/S0021-9991(03)00029-9. Table 1 lists, to ten decimals,
the rightmost eigenvalue of Hagen-Poiseuille flow for the azimuthal and axial wavenumbers
(n, k) = (1, 0), and for (1, 1) the rightmost centre mode and the rightmost wall mode, at
R = 10^2, ..., 10^7 (17 values; no centre mode at R = 10^2). The authors expect all but possibly
the last of the ten digits to be correct. Units: pipe radius, centreline velocity.

What is computed: the eigenvalues s of the linearised Navier-Stokes equations about U = 1 - r^2
(primitive variables, no-slip wall, regular at the axis) for one Fourier pair (n, k), at two radial
resolutions N1 < N2 that grow with Re. A computed eigenvalue is kept only if the other resolution
finds it again to 1e-8 (this removes spurious modes of the discretisation), and the value at N2 is
reported. Conatus writes modes as exp(i n phi + i k z + s t); Meseguer & Trefethen's Table 1 lists
the complex conjugate, which is the value compared here. Branches: the rightmost (1, 1) mode with
|Im s| > 0.75 is the centre mode, the rightmost one with |Im s| < 0.6 the wall mode.
Gap: |s_Conatus - conj(lambda_MT)|; "within rounding" when the real and imaginary parts are both
within 5e-11 (half a unit of the 10th decimal) of the printed value.

Run: python examples/published/pipe_flow_spectrum/run.py   (writes result.json)
"""
import json

import numpy as np

import conatus as cn
from conatus.expert import eigenvalues

# Meseguer & Trefethen (2003), Table 1: R -> ((1, 0), (1, 1) centre mode, (1, 1) wall mode)
TABLE_1 = {
    1e2: ("-0.1468197064", None, ("-0.1471366653", "0.5725629712")),
    1e3: ("-0.0146819706", ("-0.0708640053", "0.8467498288"), ("-0.0911426036", "0.4691428789")),
    1e4: ("-0.0014681971", ("-0.0227049146", "0.9514811947"), ("-0.0472321996", "0.2737887093")),
    1e5: ("-0.0001468197", ("-0.0072023080", "0.9846498286"), ("-0.0292364601", "0.1372143077")),
    1e6: ("-0.0000146820", ("-0.0022796480", "0.9951451356"), ("-0.0153380019", "0.0649631475")),
    1e7: ("-0.0000014682", ("-0.0007210913", "0.9984646856"), ("-0.0074895687", "0.0303389812")),
}
RESOLUTION = {1e2: (32, 48), 1e3: (48, 64), 1e4: (64, 96), 1e5: (96, 128), 1e6: (160, 192), 1e7: (256, 320)}
HALF = 5e-11


def spectrum(Re, N, k):
    hp = cn.problems.pipe_evp(N=N, Re=Re, m=1, k=1.0)          # (n, k) = (1, 1) and (1, 0) groups
    plan = cn.discretize(hp.problem, ncc=hp.ncc)
    return eigenvalues(plan, plan.group(1, k)).values


def confirmed(Re, k):
    """Eigenvalues at N2 found again at N1 to 1e-8, and the largest such difference among the kept ones."""
    n1, n2 = RESOLUTION[Re]
    a, b = spectrum(Re, n1, k), spectrum(Re, n2, k)
    d = np.array([np.min(abs(a - z)) for z in b])
    keep = d < 1e-8 * np.maximum(1.0, abs(b))
    return b[keep], d[keep]


def rightmost(vals, diffs, lo, hi):
    sel = (abs(vals.imag) >= lo) & (abs(vals.imag) < hi)
    i = int(np.argmax(np.where(sel, vals.real, -np.inf)))
    return vals[i], diffs[i]


out = {"case": "pipe_flow_spectrum", "rows": []}
n_in = n_one = n_all = 0
gaps = []
print("R      mode     Meseguer & Trefethen (conj.)       Conatus (N2)                         gap      N1-N2   rounding")
for Re, (k0, centre, wall) in TABLE_1.items():
    v0, d0 = confirmed(Re, 0.0)
    v1, d1 = confirmed(Re, 1.0)
    picks = [("(1,0)", (k0, "0"), rightmost(v0, d0, 0.0, np.inf))]
    if centre:
        picks.append(("centre", centre, rightmost(v1, d1, 0.75, np.inf)))
    picks.append(("wall", wall, rightmost(v1, d1, 0.0, 0.6)))
    for name, (re_t, im_t), (s, dres) in picks:
        ref = complex(float(re_t), -float(im_t))                 # conjugate of the tabulated value
        gap = abs(s - ref)
        ok = abs(s.real - ref.real) <= HALF and abs(s.imag - ref.imag) <= HALF
        one = abs(s.real - ref.real) <= 2 * HALF and abs(s.imag - ref.imag) <= 2 * HALF
        n_in += ok
        n_one += one
        n_all += 1
        gaps.append(gap)
        out["rows"].append({"Re": Re, "mode": name, "ref": [ref.real, ref.imag], "ref_text": [re_t, im_t],
                            "conatus": [s.real, s.imag], "gap": gap, "resolution_diff": dres,
                            "N": RESOLUTION[Re], "within_rounding": bool(ok)})
        print(f"{Re:7.0e} {name:7s} {ref.real:+.10f} {ref.imag:+.10f}i   {s.real:+.12f} {s.imag:+.12f}i   "
              f"{gap:.1e}  {dres:.0e}  {'yes' if ok else ('last digit' if one else 'no')}")
print(f"tabulated eigenvalues: {n_all}; within rounding: {n_in} of {n_all}; "
      f"within one unit of the last digit: {n_one} of {n_all}")
low = [g for r, g in zip(out["rows"], gaps, strict=True) if r["Re"] <= 1e4]
print(f"largest gap for R = 10^2 to 10^4: {max(low):.1e}; for R = 10^2 to 10^7: {max(gaps):.1e}")
out.update({"within_rounding": n_in, "within_last_digit": n_one, "n_ref": n_all, "max_gap": max(gaps)})
if __name__ == "__main__":
    # what this case establishes: every tabulated value to 10^-9 (below 10^-10 up to R = 10^5), each
    # converged in N to 10^-11
    assert max(gaps) < 1e-9 and max(r["resolution_diff"] for r in out["rows"]) < 1e-11
    assert max(g for r, g in zip(out["rows"], gaps, strict=True) if r["Re"] <= 1e5) < 1e-10
    with open("result.json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item())
