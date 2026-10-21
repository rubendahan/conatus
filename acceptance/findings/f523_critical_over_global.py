"""F523 (external review of 0.1.1, F6): critical(over=) finds the lower of two interior minima.

The neutral curve r_c(q) = (q+0.6)^2 (q-0.5)^2 + 0.01 (q+0.6) has local minima near q = -0.604
(r_c = -2.05e-5) and q = 0.496 (r_c = 0.01098). One bounded Brent on (-1, 1) returned the second,
verified. Expected: the first, within the declared tolerances (oracle: a NumPy grid of 2e6 points).
"""
import numpy as np
from _acceptance import check, passed, start

start(523, "critical(over=) returns the smallest neutral value of the interval")
import conatus as cn  # noqa: E402
from conatus.expert import critical  # noqa: E402

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), r=0.1, q=0.1)
m.scalar("u")
m.eigenvalue("s", growth="real")
m.equations("s*u=(r-((q+0.6)**2*(q-0.5)**2+0.01*(q+0.6)))*u")
P = m.parametric(8, over=("r", "q"))
c = critical(P, "r", bracket=(-0.5, 2), over={"q": (-1, 1)}, xtol=1e-9)
q = np.linspace(-1, 1, 2_000_001)
f = (q + 0.6) ** 2 * (q - 0.5) ** 2 + 0.01 * (q + 0.6)
k = int(np.argmin(f))
print(f"  critical r_c = {c.value:.10g} at q = {c.over['q']:.8g}; grid {f[k]:.10g} at {q[k]:.8g}")
check(abs(c.value - f[k]) < 1e-8, f"r_c = {c.value:.6g}, the minimum over (-1, 1) is {f[k]:.6g}")
check(abs(c.over["q"] - q[k]) < 1e-4, f"q = {c.over['q']:.6g}, expected {q[k]:.6g}")
passed(f"r_c = {c.value:.4e} at q = {c.over['q']:.5f}")
