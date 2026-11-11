"""0.3.1 review: an event is located at the zero of its crossing factor, not where that factor
enters the rounding band set by a stiff spectator. Oracle: x' = p x - x^2 (branch point at p = 0)
and the Hopf normal form (Hopf at p = 0), each with a decoupled spectator z' = -mu z. With
mu = 1e8 the branch point was placed at p = -6.7e-7 and the Hopf point at -3.3e-7 (10 n eps mu),
with a bracket of 1e-11 of the step; with mu = 1e10, 100 times further."""
import numpy as np
import pytest

import conatus.expert as C
from conatus.bifurcation import continue_equilibria

CFG = dict(ds=0.05, ds_min=1e-9, ds_max=0.1, tolerance=1e-10, max_points=60)


def _problem(kind, mu):
    if kind == "branch point":
        names = ["x", "z"]

        def rhs(X, P):
            return [P["p"] * X["x"] - X["x"] ** 2, -mu * X["z"]]
    else:
        names = ["a", "b", "z"]

        def rhs(X, P):
            a, b, p = X["a"], X["b"], P["p"]
            r2 = a * a + b * b
            return [p * a - b - a * r2, a + p * b - b * r2, -mu * X["z"]]
    return C.ode_problem(names, {"p": -0.5}, rhs), len(names)


@pytest.mark.parametrize("kind", ["branch point", "hopf"])
@pytest.mark.parametrize("mu", [1.0, 1e8, 1e10])
def test_event_at_the_zero_of_its_factor_beside_a_stiff_spectator(kind, mu):
    pb, n = _problem(kind, mu)
    cv = continue_equilibria(C.ir_system(pb, ["p"]), np.zeros(n), "p",
                             C.ContinuationSettings(**CFG), marginal_tol=1e-9, bounds=(-0.5, 1.5))
    ev = cv.events_of(kind)
    assert len(ev) == 1 and ev[0].converged
    assert abs(float(ev[0].point.z[-1])) <= 1e-9, ev[0].explain()


@pytest.mark.parametrize("mu", [1.0, 1e6, 1e8])
def test_fold_beside_a_stiff_spectator(mu):
    """x' = p - x^2 (fold at p = 0) and z' = -mu z: the fold was placed at p = -6.3e-6 for
    mu = 1e8, the corrector's one scale for all rows letting the stiff row loosen the slow one."""
    pb = C.ode_problem(["x", "z"], {"p": 1.0}, lambda X, P: [P["p"] - X["x"] ** 2, -mu * X["z"]])
    cv = continue_equilibria(C.ir_system(pb, ["p"]), np.array([1.0, 0.0]), "p",
                             C.ContinuationSettings(**CFG), marginal_tol=1e-9, bounds=(-0.5, 1.5),
                             direction=-1.0)
    ev = cv.events_of("fold")
    assert len(ev) == 1 and abs(float(ev[0].point.z[-1])) <= 1e-9, ev[0].explain()
