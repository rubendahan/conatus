"""Audit (params keys never checked) and the proposed NCC-tail decision.

Written before the fixes; each failed on the code it guards.
"""
import warnings

import numpy as np
import pytest

from conatus.discretize import DiscretizationError, discretize
from conatus.ir import (LBVP, NLBVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter,
                        RealFourier, diff, interp, lift)

def _poisson_with_k():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, 16, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, t1, t2, f = (Field("u", D), Field("t1", Domain(cs, [])), Field("t2", Domain(cs, [])),
                    Field("f", D))
    k = Parameter("k", 2.0)
    pb = LBVP([u, t1, t2])
    pb.add_equation((diff(diff(u, x), x) - k * u + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    return pb


def test_cartesian_plan_refuses_stray_params():
    """R1 (high): discretize(pb, params={"kk": 3.0}) used the declared k = 2 without a word. The
    message lists the valid keys and suggests the closest one (never applied)."""
    pb = _poisson_with_k()
    discretize(pb, params={"k": 3.0})                          # a real parameter: accepted
    with pytest.raises(DiscretizationError, match=r"'kk'") as e:
        discretize(pb, params={"kk": 3.0})
    msg = str(e.value)
    assert "valid keys: ['k']" in msg and "did you mean 'k'" in msg, msg


def test_cartesian_plan_stray_params_change_the_solution_otherwise():
    """Why R1 is high: the typo'd plan solved with k = 2 (declared), not 3 (intended)."""
    pb = _poisson_with_k()
    p3 = discretize(pb, params={"k": 3.0})
    assert p3.params == {"k": 3.0}
    with pytest.raises(DiscretizationError):
        discretize(pb, params={"K": 3.0})                      # case matters too


def test_periodic_newton_refuses_stray_params():
    """R11: PeriodicNewtonSolver keeps its own params (it never goes through discretize), so R1
    does not cover it: same check there."""
    from conatus.solvers import PeriodicNewtonSolver, PhaseCondition
    cs = CartesianCoordinates("x")
    x = cs["x"]
    u = Field("u", Domain(cs, [RealFourier(x, 12, (0, 2 * np.pi), dealias=2)]))
    r = Parameter("r", 0.1)
    pb = NLBVP([u])
    pb.add_equation((r * u - u - 2 * diff(diff(u, x), x) - diff(diff(diff(diff(u, x), x), x), x)
                     - u * u * u, 0))
    kw = dict(tolerance=1e-12, damping=1.0, max_iterations=5, phase=PhaseCondition(u, x, kind="sin1"))
    PeriodicNewtonSolver(pb, params={"r": 0.2}, **kw)
    with pytest.raises(DiscretizationError, match=r"'rr'.*did you mean 'r'"):
        PeriodicNewtonSolver(pb, params={"rr": 0.2}, **kw)

def test_under_resolved_ncc_is_refused():
    """Audit §3: an NCC whose Chebyshev tail exceeded 1e-12 was used as is (then, provisionally,
    with a NCCResolutionWarning). decided: refused, naming the coefficient,
    its tail and the resolution or declaration to write instead."""
    from conatus.errors import DiscretizationError
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, 16, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, t1, t2, f, U = (Field("u", D), Field("t1", Domain(cs, [])), Field("t2", Domain(cs, [])),
                       Field("f", D), Field("U", D))
    pb = LBVP([u, t1, t2])
    pb.add_equation((diff(diff(u, x), x) - U * u + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    with pytest.raises(DiscretizationError, match=r"NCC U.*tail"):
        discretize(pb, ncc={U: lambda X: np.tanh(20 * X)}).matrices()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        discretize(pb, ncc={U: lambda X: 1 - X ** 2}).matrices()          # exact polynomial
