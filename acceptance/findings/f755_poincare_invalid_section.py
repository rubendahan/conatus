"""F755 (audit of 0.2.1, N60): an invalid section is not an absence of crossings.

Oracle: g given by hand; g = y has upward crossings at atan(0.1) and atan(0.1) + 2 pi on the rotation x_t = -y, y_t = x from (1, -0.1); a finite constant gives none.
"""
from _acceptance import check, passed, start

start(755, "Poincare section with non-finite values")
import numpy as np  # noqa: E402


from conatus.chaos.poincare import poincare_section  # noqa: E402
from conatus.errors import ConatusError  # noqa: E402
from conatus.bifurcation import ode_problem  # noqa: E402
from conatus.discretize import discretize  # noqa: E402
from conatus.timestep import IVPSolver  # noqa: E402


def solver():
    pb = ode_problem(["x", "y"], {}, lambda X, P: [-X["y"], X["x"]], name="unit oscillator")
    return IVPSolver(discretize(pb), "RK443", .05)


kw = dict(steps=80, g_tol=1e-11, max_iterations=40)
for value in (np.nan, np.inf, -np.inf):
    for direction in (-1, 0, 1):
        try:
            poincare_section(solver(), [1., -.1], lambda X, v=value: np.full(X.shape[0], v),
                             direction=direction, **kw)
            check(False, f"{value}, {direction}: accepted")
        except ConatusError:
            pass
kw["steps"] = 160
r = poincare_section(solver(), [1., -.1], lambda X: X[:, 1], direction=1, **kw)
t = r.times[0]
check(np.allclose(t, [np.arctan(.1), np.arctan(.1) + 2 * np.pi], atol=5e-6), f"times {t}")
r = poincare_section(solver(), [1., -.1], lambda X: -1. - 0 * X[:, 1], direction=0, **kw)
check(r.times[0].size == 0, "constant section")
passed("refused, valid section exact")
