"""F950 (hunt 5C, M2): EmbeddedError measures the unknowns, not the tau variables.

Oracle: u_t = c u_zz on [0, 1], u = 0 at both walls, u(0) = sin(pi z): u = exp(-c pi^2 t) sin(pi z)
in closed form, and the rescaling t -> c t leaves the problem identical. At dt c = 1e-6 one RK443
step has a true error below 1e-9 relative, so the estimate must be small too (it was 9e-2 with
c = 1e4). An adaptive run (tol 1e-7) must take the same number of steps for c = 1 and c = 1e4
(it raised ConvergenceError at c = 1e4) and match the closed form.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(950, "EmbeddedError: estimate of the unknowns, invariant under t -> c t")
import conatus as cn  # noqa: E402
from conatus.expert import EmbeddedError  # noqa: E402

warnings.simplefilter("ignore")


def sim(c, dt):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="h", c=c)
    m.scalar("u")
    m.equations("dt(u) - c*dz(dz(u)) = 0")
    m.at("bottom", u=0)
    m.at("top", u=0)
    s = m.simulate(N=(24,), scheme="RK443", dt=dt)
    s.initial(u=lambda z: np.sin(np.pi * z))
    return s


for c in (1.0, 1e4):
    s = sim(c, 1e-6 / c)
    est = EmbeddedError(s.solver).step()
    z = np.asarray(s.coordinate("z"))
    true = np.max(np.abs(s.grid("u") - np.exp(-c * np.pi ** 2 * s.t) * np.sin(np.pi * z)))
    check(est < 1e-6, f"c = {c:g}, dt c = 1e-6: estimate {est:.2e}, true error {true:.1e}")

steps = {}
for c in (1.0, 1e4):
    s = sim(c, 1e-3 / c)
    e = EmbeddedError(s.solver, adapt=True, tol=1e-7, safety=0.9)
    n = 0
    while s.t * c < 0.05:
        e.step()
        n += 1
    z = np.asarray(s.coordinate("z"))
    err = np.max(np.abs(s.grid("u") - np.exp(-c * np.pi ** 2 * s.t) * np.sin(np.pi * z)))
    check(err < 1e-5, f"c = {c:g}: error {err:.1e} against the closed form")
    steps[c] = (n, e.rejected)
check(steps[1.0][0] == steps[1e4][0], f"steps differ under t -> c t: {steps}")
passed(f"estimate small at dt c = 1e-6; adaptive steps {steps} for c = 1 and 1e4")
