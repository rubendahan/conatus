"""Tutorial 1 -- a first eigenvalue problem: Orr--Sommerfeld stability of plane Poiseuille flow.

Written with the native layer (cn.Model): the equation as on paper, conditions by boundary name;
the tau terms are placed automatically and listed by explain(). The expert version (IR with
add_bc) is expert/t01_orr_sommerfeld.py; both give bit-identical matrices (tests/model).

Expected values (checked at the end of the script):
  c(Re = 1e4, alpha = 1) = 0.23752649 + 0.00373967i   Orszag (1971), J. Fluid Mech. 50, 689
  Re_c = 5772.22, alpha_c = 1.02056                    Orszag (1971)
Run: python t01_orr_sommerfeld.py   (under a minute of CPU)
"""
# [imports]
import conatus as cn
from conatus.expert import critical, verify
# [/imports]

# [problem]
line = cn.Box(y=cn.walls(-1, 1))               # wall-normal coordinate y in [-1, 1]
m = cn.Model(line, name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
m.scalar("v")                                  # v(y) exp(i alpha (x - c t)), wall-normal velocity
m.given(U=lambda y: 1 - y**2)                  # plane Poiseuille base flow (depends on y)
m.eigenvalue("c", growth="imag")               # wave speed c: a mode grows with Im c
m.define("L2(f) = dy(dy(f)) - alpha**2*f")     # D^2 - alpha^2
m.equations("Orr_Sommerfeld: L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")            # rigid walls at y = -1 and y = 1
print(m.explain(full=True))                    # every automatic decision (taus, modes)
# [/problem]

# [solve]
res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)   # keep modes found again at N = 96
i = int(res.values.imag.argmax())                     # least stable surviving mode
print(res.explain(key="imag"))
print(f"c = {res.values[i]:.10f}   backward residual {res.residuals[i]:.1e}")
# [/solve]

# [mode]
mode = res.mode(i)                             # the eigenfunction on the grid, taus left out
print(mode.explain())                          # unknowns, grid, declared normalisation
y, v = mode.coordinate("y"), mode["v"]         # Chebyshev points, complex v(y)
# [/mode]

# [figure]
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
ax.plot(y, v.real, label="Re v")
ax.plot(y, v.imag, label="Im v")
ax.plot(y, abs(v), "k--", lw=0.8, label="|v|")
ax.set_xlabel("y")
ax.set_title("Least stable Orr-Sommerfeld mode, Re = 10⁴, α = 1")
ax.legend()
fig.savefig("mode.png", bbox_inches="tight")
# [/figure]

# [critical]
# L(Re, alpha) + c M(Re, alpha), exact in the parameters; the same pencil at N = 96 rejects the
# spurious tau modes (drift test, as for eigenvalues above); growth = Im c (the phase speed c)
P = m.parametric(N=64, over=("Re", "alpha"), drift=96, drift_tol=1e-6)
least = P.least_stable(Re=1e4, alpha=1.0)
crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-9)
print(crit.explain())                          # growth convention, drift counts, verification
print(f"Re_c = {crit.value:.4f}   alpha_c = {crit.over['alpha']:.6f}")
print(f"dc/dRe = {P.sensitivity(least, 'Re'):.3e}")
print(verify(P, crit.mode, crit).explain())  # residuals, spectral tail, Taylor tests
# [/critical]

# [check]
C_ORSZAG = 0.23752649 + 0.00373967j          # Orszag (1971), Re = 1e4, alpha = 1
RE_C, ALPHA_C = 5772.22, 1.02056             # Orszag (1971); accurate value 1.020547
assert abs(res.values[i] - C_ORSZAG) < 1e-8, res.values[i]
assert res.residuals[i] < 1e-12
assert abs(abs(v).max() - 1) < 1e-14 and abs(v[0]) < 1e-3 and abs(v[-1]) < 1e-3   # normalised, v = 0 at the walls
assert abs(crit.value - RE_C) < 0.01 and abs(crit.over["alpha"] - ALPHA_C) < 2e-5
print("OK: c and Re_c agree with Orszag (1971) to the published digits; alpha_c within 2e-5")
# [/check]
