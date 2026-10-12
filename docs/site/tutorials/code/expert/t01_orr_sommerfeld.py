"""Tutorial 1 -- a first eigenvalue problem: Orr--Sommerfeld stability of plane Poiseuille flow.

Expert layer: the same tutorial written directly in the IR (bases, tau variables and lifts,
LHS/RHS by hand). The native version one directory up gives bit-identical matrices (tests/model).

Expected values (checked at the end of the script):
  c(Re = 1e4, alpha = 1) = 0.23752649 + 0.00373967i   Orszag (1971), J. Fluid Mech. 50, 689
  Re_c = 5772.22, alpha_c = 1.02056                    Orszag (1971)
Run: python t01_orr_sommerfeld.py   (a few seconds of CPU)
"""
# [imports]
from conatus.expert import (EVP, Parameter, ParametricEVP, channel, critical, diff, discretize,
                            eigenvalues, interp, verify)
# [/imports]

# [problem]
c = Parameter("c", dtype=complex)            # the eigenvalue: complex wave speed c
Re, a = Parameter("Re", 1e4), Parameter("alpha", 1.0)


def orr_sommerfeld(N):
    """Orr--Sommerfeld EVP for v(y) on N Chebyshev modes, and the channel it lives on."""
    ch = channel(N, base="poiseuille")       # y in [-1, 1], U(y) = 1 - y^2 (an NCC)
    v, y, U = ch.field("v"), ch.y, ch.U

    def L2(f):                               # (D^2 - alpha^2) f for exp(i alpha (x - c t))
        return diff(diff(f, y), y) - a ** 2 * f

    pb = EVP([v], eigenvalue=c, name="Orr-Sommerfeld")
    pb.add_equation((L2(L2(v)) / (1j * a * Re) - U * L2(v) + diff(diff(U, y), y) * v
                     + c * L2(v), 0))
    pb.add_bc(interp(v, y, -1), interp(v, y, 1),                    # v = 0 at the walls
              interp(diff(v, y), y, -1), interp(diff(v, y), y, 1))  # v' = 0 at the walls
    return pb, ch


pb, ch = orr_sommerfeld(64)
print(pb.explain())
# [/problem]

# [solve]
plan = discretize(pb, ncc=ch.ncc)            # frozen discrete plan; its matrices are public
pb96, ch96 = orr_sommerfeld(96)              # same problem, finer: used to reject tau modes
res = eigenvalues(plan, drift_plan=discretize(pb96, ncc=ch96.ncc), drift_tol=1e-6)
i = int(res.values.imag.argmax())            # least stable surviving mode
print(res.explain(key="imag"))
print(f"c = {res.values[i]:.10f}   backward residual {res.residuals[i]:.1e}")
# [/solve]

# [critical]
P = ParametricEVP(pb, [Re, a], ncc=ch.ncc)   # L(Re, alpha) + c M(Re, alpha), exact in p
opts = dict(growth="imag", select=lambda s: abs(s) < 2)   # declared filter on tau modes
mode = P.least_stable(Re=1e4, alpha=1.0, **opts)
crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-9, **opts)
print(f"Re_c = {crit.value:.4f}   alpha_c = {crit.over['alpha']:.6f}")
print(f"dc/dRe = {P.sensitivity(mode, 'Re'):.3e}")
print(verify(P, crit.mode, crit).explain())  # residuals, spectral tail, Taylor tests
# [/critical]

# [check]
C_ORSZAG = 0.23752649 + 0.00373967j          # Orszag (1971), Re = 1e4, alpha = 1
RE_C, ALPHA_C = 5772.22, 1.02056             # Orszag (1971); accurate value 1.020547
assert abs(res.values[i] - C_ORSZAG) < 1e-8, res.values[i]
assert res.residuals[i] < 1e-12
assert abs(crit.value - RE_C) < 0.01 and abs(crit.over["alpha"] - ALPHA_C) < 2e-5
print("OK: c and Re_c agree with Orszag (1971) to the published digits; alpha_c within 2e-5")
# [/check]
