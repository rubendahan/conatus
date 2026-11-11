"""F910 (hunt 5A, K3): poincare_section finds both crossings when two fall within one step, or
says it may have missed some.

Oracle: Stuart-Landau x' = x - w y - x r^2, y' = w x + y - y r^2, w = 2, from (1, 0): the unit
circle, theta = w t. The section y = 0.99 is crossed upwards at theta = asin 0.99 and downwards
at pi - asin 0.99 (0.1415 apart in t), each once per period T = pi. Over two periods with
dt = 0.2 (0.3.0: 2 of 4 crossings, return time 2T, no word): 4 crossings at the closed-form
times (to 1e-3 at dt = 0.01; to 0.05, the scheme's phase error at dt = 0.2 and well below the
0.14 between the two crossings), or a CorrectnessWarning. Control: dt = 0.01.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(910, "poincare_section: two crossings in one step found or reported")
from conatus.chaos import poincare_section  # noqa: E402
from conatus.bifurcation import ode_problem  # noqa: E402
from conatus.expert import IVPSolver, discretize  # noqa: E402
from conatus.errors import CorrectnessWarning  # noqa: E402

W, H = 2.0, 0.99


def solver(dt):
    pb = ode_problem(["x", "y"], {"w": W}, lambda X, P: [
        X["x"] - P["w"] * X["y"] - X["x"] * (X["x"] * X["x"] + X["y"] * X["y"]),
        P["w"] * X["x"] + X["y"] - X["y"] * (X["x"] * X["x"] + X["y"] * X["y"])])
    return IVPSolver(discretize(pb), "RK443", dt)


a = np.arcsin(H)
T = 2 * np.pi / W
exact = {1: [a / W, a / W + T], -1: [(np.pi - a) / W, (np.pi - a) / W + T]}
exact[0] = sorted(exact[1] + exact[-1])
for dt in (0.01, 0.2):
    for direction in (0, 1, -1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = poincare_section(solver(dt), np.array([1.0, 0.0]), lambda X: X[:, 1] - H,
                                 direction=direction, steps=int(round(2 * T / dt)) + 1,
                                 g_tol=1e-12, max_iterations=60)
        warned = any(issubclass(w.category, CorrectnessWarning) for w in caught)
        t = np.asarray(r.times[0])
        where = f"dt {dt}, direction {direction:+d}: times {np.round(t, 5)}, warned {warned}"
        tol = 1e-3 if dt < 0.1 else 0.05
        good = t.size == len(exact[direction]) and np.allclose(t, exact[direction], atol=tol)
        check(good or warned, where)
        if direction and good:
            check(np.allclose(r.return_times(), T, atol=tol), f"{where}: return times {r.return_times()}")
passed("4 crossings in two periods at the closed-form times for dt = 0.01 and 0.2; return time T")
