#!/usr/bin/env python3
"""F794: a linear boundary-value solve states whether its solution is resolved, and a
vanishing leading coefficient is refused because the order drops. Oracle: the exact solution of
nu u'' - u = 1, u(0) = u(1) = 0, u = -1 + cosh((z - 1/2)/sqrt(nu)) / cosh(1/(2 sqrt(nu))), which
lies in [-1, 0]."""
from _acceptance import check, passed, start

start(794, "BVP resolution statement and order drop")
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.errors import BoundaryConditionError, UnderResolvedWarning  # noqa: E402


def layer(nu, N):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="layer", nu=nu)
    u = m.scalar("u")  # noqa: F841
    m.equations("nu*dz(dz(u)) - u = 1")
    m.at("walls", u=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = m.solve(N)
    return sol, [x for x in w if issubclass(x.category, UnderResolvedWarning)]


sol, w = layer(1e-10, 32)
z = sol.coordinate("z")
exact = -1 + np.exp(-z * 1e5) + np.exp((z - 1) * 1e5)
check(np.abs(sol.grid("u") - exact).max() > 0.5, "the N = 32 case is really wrong at the grid points")
check(w, "nu = 1e-10, N = 32 warns")
check("UNDER-RESOLVED" in sol.explain(), "explain() states it")

sol, w = layer(0.01, 64)
z = sol.coordinate("z")
r = 1 / (2 * np.sqrt(0.01))
exact = -1 + np.cosh((z - 0.5) / np.sqrt(0.01)) / np.cosh(r)
check(np.abs(sol.grid("u") - exact).max() < 1e-10, "nu = 0.01, N = 64 matches the exact solution")
check(not w, "a resolved solve does not warn")
check("largest spectral tail" in sol.explain(), "explain() gives the tail of a resolved solve")

# Thresholds in the middle: the warning follows the error of the solution (nu = 1e-5).
for N, should in ((32, True), (512, False)):
    sol, w = layer(1e-5, N)
    z = sol.coordinate("z")
    s = np.sqrt(1e-5)
    err = np.abs(sol.grid("u") - (-1 + np.exp(-z / s) + np.exp((z - 1) / s))).max()
    check(bool(w) == should, f"nu = 1e-5, N = {N}: error {err:.1e}, warned {bool(w)}")


def refused(eq, **params):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="drop", **params)
    m.scalar("u")
    m.equations(eq)
    m.at("walls", u=0)
    try:
        m.solve(32)
    except BoundaryConditionError as err:
        return "drops" in str(err)
    return False


check(refused("nu*dz(dz(u)) - u = 1", nu=0.0), "nu = 0 with two wall conditions is refused")
check(refused("kappa*dz(dz(u)) - u = 1", kappa=0.0), "the name of the parameter does not matter")
check(refused("a*dz(dz(u)) + b*dz(dz(u)) - u = 1", a=2.0, b=-2.0), "terms that cancel are read")
check(refused("a*dz(dz(u)) + dz(u) = 1", a=0.0), "an order drop from 2 to 1 is refused")
check(not refused("a*dz(dz(u)) + b*dz(dz(u)) - u = 1", a=2.0, b=1.0), "a non-zero sum is accepted")
passed("resolution stated on LBVP; order drop refused")
