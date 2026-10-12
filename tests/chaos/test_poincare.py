"""Poincaré sections with the crossing located on the scheme.

Stuart–Landau oscillator x' = x - w y - x r^2, y' = w x + y - y r^2 (r' = r - r^3, theta' = w):
section y = 0 crossed upwards (theta = 0 mod 2 pi). Closed forms: return time 2 pi / w for every
orbit, first-return map r_{k+1} = (1 + (r_k^-2 - 1) e^{-2T})^{-1/2} with T = 2 pi / w. Several
initial radii in one ensemble. The located points carry only the scheme's error: the errors of the
return times and of the map decrease with the order of RK443 (3) when dt is halved, and the
residual |g| at the points is below the declared tolerance."""
import numpy as np
import pytest

from conatus.bifurcation import ode_problem
from conatus.chaos import poincare_section
from conatus.discretize import discretize
from conatus.errors import ConvergenceError, OptionError
from conatus.timestep import IVPSolver

W = 2.0


def _solver(dt=0.01):
    pb = ode_problem(["x", "y"], {"w": W}, lambda X, P: [
        X["x"] - P["w"] * X["y"] - X["x"] * (X["x"] * X["x"] + X["y"] * X["y"]),
        P["w"] * X["x"] + X["y"] - X["y"] * (X["x"] * X["x"] + X["y"] * X["y"])], "Stuart-Landau")
    return IVPSolver(discretize(pb), "RK443", dt)


@pytest.mark.slow                   # 5.7 s in the fast suite (qual_durations0)
def test_stuart_landau_return_map_and_times():
    X0 = np.array([[0.3, -0.01], [2.0, -0.01], [0.8, -0.3]])
    T = 2 * np.pi / W
    errs = []
    for dt in (0.01, 0.005):
        s = _solver(dt)
        res = poincare_section(s, X0, lambda X: X[:, 1], direction=1, steps=int(round(20 / dt)),
                               g_tol=1e-13, max_iter=30)
        assert all(len(p) >= 5 for p in res.points)
        r0, r1 = res.return_pairs(lambda p: p[0])
        exact = 1 / np.sqrt(1 + (1 / r0**2 - 1) * np.exp(-2 * T))
        errs.append((np.abs(res.return_times() - T).max(), np.abs(r1 - exact).max()))
        assert max(max(r) for r in res.residuals) <= 1e-13
        # the factors made for the located sub-steps are discarded (the single solver was only
        # used for the location: nothing is left in its cache)
        assert not s.system._factors
    (et1, em1), (et2, em2) = errs
    assert et1 < 3e-6 and em1 < 3e-6
    assert 6 < et1 / et2 < 10 and 6 < em1 / em2 < 10          # third order: scheme error only
    assert "regula falsi" in res.explain()


def test_poincare_refusals():
    s = _solver()
    with pytest.raises(OptionError, match="direction"):
        poincare_section(s, np.array([0.5, -0.1]), lambda X: X[:, 1], direction=2, steps=10,
                         g_tol=1e-12, max_iter=5)
    with pytest.raises(ConvergenceError, match="not located"):
        poincare_section(s, np.array([0.5, -0.1]), lambda X: X[:, 1], direction=1, steps=400,
                         g_tol=1e-30, max_iter=2)
