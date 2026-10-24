"""Orr-Sommerfeld spectrum of plane Poiseuille flow at Re = 10^4, alpha = 1 (Orszag 1971).

Reference: S. A. Orszag (1971), Accurate solution of the Orr-Sommerfeld stability equation,
J. Fluid Mech. 50, 689-703, doi:10.1017/S0022112071002842. The 32 eigenvalues c below are copied
from the column "lambda_n ([23])" of Table 1 of S. Gana (2026), Advancements in spectral collocation
methods for high-order eigenvalue problems, arXiv:2608.29932, which reproduces Orszag's values next
to its own computation. Re_c = 5772.22 is in the abstract of Orszag (1971); alpha_c = 1.02056 is
the value later papers quote from it.

What is computed: every eigenvalue c of the Orr-Sommerfeld equation for U = 1 - y^2 between rigid
walls at y = -1, 1, at N = 128 Chebyshev modes; a mode is kept only if it is found again at N = 160
(drift test, 1e-6: the modes where the branches of the spectrum meet are sensitive to rounding at
the 1e-8 level). Each published eigenvalue is matched to the nearest computed one not already
matched. The gap of a point is |c_Conatus - c_Orszag|; it is "within rounding" when both |dc_r| and
|dc_i| are at most half a unit of the last digit Orszag prints. Then the critical Reynolds number.

Run: python examples/published/orr_sommerfeld_spectrum/run.py   (writes result.json)
"""
import json
from decimal import Decimal

import numpy as np

import conatus as cn
from conatus.expert import critical

# Orszag (1971), as tabulated in Gana (2026), Table 1, third column; index n of that table.
ORSZAG = [
    (1, "0.23752649", "0.00373967"), (2, "0.1900592", "-0.1828219"), (3, "0.27720434", "-0.05089873"),
    (5, "0.34910682", "-0.12450198"), (6, "0.41635102", "-0.13822652"), (7, "0.36850", "-0.23882"),
    (8, "0.38399", "-0.26511"), (9, "0.474901", "-0.208731"), (10, "0.532045", "-0.206466"),
    (11, "0.51292", "-0.28663"), (13, "0.58721", "-0.26716"), (14, "0.63672", "-0.25988"),
    (20, "0.70887", "-0.28765"), (21, "0.71232", "-0.28551"), (24, "0.73741", "-0.25872"),
    (25, "0.73812", "-0.25969"), (27, "0.76588", "-0.23119"), (28, "0.76649", "-0.23159"),
    (31, "0.794388", "-0.203221"), (32, "0.794818", "-0.203529"), (34, "0.8228350", "-0.1752287"),
    (35, "0.8231370", "-0.1754781"), (38, "0.8512458", "-0.1472339"), (39, "0.8514494", "-0.1474256"),
    (41, "0.87962729", "-0.11923285"), (42, "0.87975570", "-0.11937073"), (45, "0.90798305", "-0.09122274"),
    (46, "0.90805633", "-0.09131286"), (47, "0.93631654", "-0.06320150"), (48, "0.93635178", "-0.06325157"),
    (49, "0.96463092", "-0.03516728"), (50, "0.96464251", "-0.03518658"),
]
RE_C, ALPHA_C = "5772.22", "1.02056"          # Orszag (1971), text


def half_unit(s: str) -> float:
    """Half a unit of the last printed digit of s."""
    return 0.5 * 10.0 ** Decimal(s).as_tuple().exponent


m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="plane Poiseuille", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")

res = m.eigenvalues(N=128, drift=160, drift_tol=1e-6)
c = res.values
print(f"N = 128 (drift test at N = 160, 1e-6): {c.size} modes kept, max backward residual {res.residuals.max():.1e}")
print(" n   Orszag (1971)               Conatus                  gap      within rounding")
rows, inside, one, free = [], 0, 0, list(range(c.size))
for n, cr, ci in ORSZAG:
    ref = complex(float(cr), float(ci))
    j = min(free, key=lambda i: abs(c[i] - ref))
    free.remove(j)
    gap = abs(c[j] - ref)
    ok = abs(c[j].real - ref.real) <= half_unit(cr) * (1 + 1e-9) and abs(c[j].imag - ref.imag) <= half_unit(ci) * (1 + 1e-9)
    unit = abs(c[j].real - ref.real) <= 2 * half_unit(cr) * (1 + 1e-9) and abs(c[j].imag - ref.imag) <= 2 * half_unit(ci) * (1 + 1e-9)
    inside += ok
    one += unit
    rows.append({"n": n, "ref": [float(cr), float(ci)], "ref_text": [cr, ci],
                 "conatus": [c[j].real, c[j].imag], "gap": gap, "within_rounding": bool(ok)})
    print(f"{n:2d}   {cr:>11} {ci:>12}i   {c[j].real:.8f} {c[j].imag:+.8f}i   {gap:.1e}  {'yes' if ok else ('last digit' if unit else 'no')}")
gaps = np.array([r["gap"] for r in rows])
worst = rows[int(np.argmax(gaps))]
print(f"published eigenvalues: {len(rows)}; within rounding of the printed digits: {inside} of {len(rows)}; "
      f"within one unit of the last digit: {one} of {len(rows)}")
print(f"largest gap |c_Conatus - c_Orszag| = {gaps.max():.1e} (n = {worst['n']}); median {np.median(gaps):.1e}")

P = m.parametric(N=64, over=("Re", "alpha"), drift=96, drift_tol=1e-6)
crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-10)
print(f"Re_c = {crit.value:.4f} (Orszag {RE_C}), alpha_c = {crit.over['alpha']:.6f} (Orszag {ALPHA_C})")
print(f"gap: Re_c {abs(crit.value - float(RE_C)):.1e}, alpha_c {abs(crit.over['alpha'] - float(ALPHA_C)):.1e}")

if __name__ == "__main__":
    # what this case establishes: every listed eigenvalue within one unit of Orszag's last digit,
    # resolved modes, and Re_c to Orszag's printed digits
    assert one == len(rows) and res.residuals.max() < 1e-12
    assert abs(crit.value - float(RE_C)) <= half_unit(RE_C)
    with open("result.json", "w") as f:
        json.dump({"case": "orr_sommerfeld_spectrum", "points": rows,
                   "spectrum": [[z.real, z.imag] for z in c if z.imag > -1.0],
                   "within_rounding": inside, "within_last_digit": one, "n_ref": len(rows), "max_gap": gaps.max(),
                   "Re_c": crit.value, "alpha_c": crit.over["alpha"]}, f, indent=1, default=lambda o: o.item())
