"""0.3.1 review: the primitive-period check of collocation orbits does not depend on where the
origin of the state is. Oracle: the Hopf normal form's limit cycle of radius 1/2, one lap, moved
to u0 + (x, y); its primitive period is its period whatever u0."""
import numpy as np
import pytest

import conatus.bifurcation.orbits as O
from conatus.bifurcation import DynamicalSystem


def _system(u0):
    def G(u, par):
        x, y = u[0] - u0, u[1] - u0
        r2 = x * x + y * y
        return np.array([x * (par["p"] - r2) - y, y * (par["p"] - r2) + x])

    def J(u, par):
        x, y = u[0] - u0, u[1] - u0
        r2 = x * x + y * y
        return np.array([[par["p"] - r2 - 2 * x * x, -2 * x * y - 1],
                         [-2 * x * y + 1, par["p"] - r2 - 2 * y * y]])

    return DynamicalSystem.from_callables(G, 2, {"p": 0.25}, jac=J,
                                          dfdp=lambda u, p, nm: np.array(u) - u0, fd_x=1e-4,
                                          name="hopf")


@pytest.mark.parametrize("u0, laps", [(0.0, 1), (1e9, 1), (-1e9, 1), (1e9, 2)])
def test_repetitions_do_not_depend_on_the_origin(u0, laps):
    pb = O.PeriodicOrbitProblem(_system(u0), "p", O.CollocationMesh.uniform(40, 4))
    t = pb.mesh.times()
    U = u0 + 0.5 * np.stack([np.cos(2 * np.pi * laps * t), np.sin(2 * np.pi * laps * t)], 1)
    pb.set_reference(U)
    out = O.floquet_from_collocation(pb, np.concatenate([U.ravel(), [2 * np.pi * laps, 0.25]]))
    assert out["repetitions"] == laps
