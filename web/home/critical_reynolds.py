# Linear stability of plane Poiseuille flow U(y) = 1 - y^2 between walls at y = -1 and y = 1.
# Orr-Sommerfeld equation for v(y) exp(i alpha (x - c t)): a mode grows when Im(c) > 0.
import conatus as cn
from conatus.expert import critical

m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="plane Poiseuille", Re=1e4, alpha=1.0)
m.scalar("v")
m.given(U=lambda y: 1 - y**2)
m.eigenvalue("c", growth="imag")
m.define("L2(f) = dy(dy(f)) - alpha**2*f")
m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
m.at("walls", "v = 0", "dy(v) = 0")

# Least stable mode at Re = 10000, alpha = 1 (modes kept only if found again at N = 96)
res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)
i = res.values.imag.argmax()
print(f"c = {res.values[i]:.8f}   residual {res.residuals[i]:.0e}")

# Critical Reynolds number: the lowest Re at which Im(c) = 0 for some alpha
P = m.parametric(N=64, over=("Re", "alpha"), drift=96, drift_tol=1e-6)
crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-9)
print(f"Re_c = {crit.value:.2f}   alpha_c = {crit.over['alpha']:.5f}")
