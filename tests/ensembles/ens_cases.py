"""Small problems shared by tests/ensembles and tests/chaos (and the benchmarks)."""
from __future__ import annotations

import numpy as np

from conatus.bifurcation import ode_problem
from conatus.discretize import discretize
from conatus.timestep import IVPSolver

SIGMA, BETA = 10.0, 8.0 / 3.0


def lorenz_problem(r: float = 28.0, sigma: float = SIGMA, b: float = BETA):
    """Lorenz (1963) in the IR (0-D: one constant Fourier mode per variable)."""
    return ode_problem(["x", "y", "z"], {"sigma": sigma, "r": r, "b": b},
                       lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                     P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                     X["x"] * X["y"] - P["b"] * X["z"]], "Lorenz")


def lorenz_solver(dt: float, r: float = 28.0, scheme: str = "RK443", backend: str = "cpu"):
    return IVPSolver(discretize(lorenz_problem(r)), scheme, dt, backend=backend)


def lorenz_rhs(X, r: float = 28.0, sigma: float = SIGMA, b: float = BETA):
    """Closed-form vector field f(x) (for the CLV check: the neutral CLV is parallel to f)."""
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    return np.stack([sigma * (y - x), r * x - y - x * z, x * y - b * z], axis=-1)


def run_single(solver, X0, nsteps: int):
    """End state of one ordinary IVPSolver run (the reference for member-by-member checks)."""
    solver.X = solver.xp.asarray(X0, dtype=solver.system.dtype).copy()
    solver.t, solver.iteration, solver._t_origin, solver._it_origin = 0.0, 0, 0.0, 0
    solver._MXh.clear()
    solver._Fh.clear()
    solver.run(nsteps)
    X = solver.X
    return X.copy() if solver.xp is np else solver.xp.asnumpy(X)
