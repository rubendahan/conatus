"""Fixtures of the run-registry tests: a small Kuramoto-Sivashinsky IVP and a private registry root.

Every test gets ``CN_RUNS`` pointed at its own temporary directory, so no test ever writes to the
project's registry (``scratch/runs``)."""
import numpy as np
import pytest

from conatus.expert import (IVP, CartesianCoordinates, Domain, Field, IVPSolver, RealFourier, diff,
                            discretize, dt, lap)

L = 22.0


@pytest.fixture(autouse=True)
def private_registry(tmp_path, monkeypatch):
    root = tmp_path / "registry"
    monkeypatch.setenv("CN_RUNS", str(root))
    return root


@pytest.fixture(scope="session")
def ks():
    """(plan, u, u0, make_solver): KS L = 22, N = 32, RK443, dt = 0.05."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 32, (0, L), dealias=1.5)])
    u = Field("u", D)
    pb = IVP([u], name="KS L=22")
    pb.add_equation((dt(u) + lap(u) + lap(lap(u)), -u * diff(u, x)))
    plan = discretize(pb)

    def u0(X):
        return np.cos(2 * np.pi * X / L) * (1 + np.sin(2 * np.pi * X / L))

    def make(dt_=0.05, scheme="RK443"):
        s = IVPSolver(plan, scheme, dt_)
        s.set_state({u: u0})
        return s
    return plan, u, u0, make
