"""Time-step halving: the observed global order equals the order the solver states.

The relation: for every scheme and start-up, halving dt divides the error at a fixed time by
2^p with p = ``solver.effective_order`` (the order explain() and the Methods paragraph state),
within ORDER_TOL. Problems:
  logistic   u' = -2u - u^2 on a constant field: implicit linear part, explicit nonlinear part
             (exact solution 2 / (3 e^{2t} - 1); with u' = -u - u^2 the leading errors of the
             implicit and explicit Euler parts nearly cancel and the first-order schemes look
             pre-asymptotic at these steps);
  walls_t    u_t = u_yy / 2 between walls moving as cos(t) (exact e^{-y} cos(t - y)): the
             boundary data are a row without time derivative, imposed per ``bc_time``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn
from conatus._tolerances_internal import INTERNAL
from conatus.timestep.integrator import SCHEMES, scheme_family

ORDER_TOL = INTERNAL["INVARIANCE_ORDER"].value


def _logistic_error(scheme: str, startup: str, n: int) -> tuple[float, object]:
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="logistic")
    m.scalar("u")
    m.equations("dt(u) + 2*u = -u*u")
    sim = m.simulate(4, scheme=scheme, dt=1.0 / n, startup=startup).initial(u=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run(steps=n)
    exact = 2.0 / (3.0 * np.exp(2.0 * sim.t) - 1.0)
    return float(np.max(np.abs(sim.grid("u") - exact))), sim.solver


def _walls_error(scheme: str, bc_time: str, n: int, startup: str = "rk443"):
    m = cn.Model(cn.Box(y=cn.walls(0, 5)), name="moving walls")
    m.scalar("u")
    m.equations("dt(u) - dy(dy(u))/2 = 0")
    m.at("bottom", u="cos(t)")
    m.at("top", u=lambda t: np.exp(-5) * np.cos(t - 5))
    kw = {} if scheme_family(scheme) == "rk" else {"startup": startup}
    sim = m.simulate(N=40, scheme=scheme, dt=1.0 / n, dealias=1, bc_time=bc_time, **kw)
    sim.initial(u=lambda y: np.exp(-y) * np.cos(-y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run(steps=n)
    y = sim.coordinate("y")
    return float(np.max(np.abs(sim.grid("u") - np.exp(-y) * np.cos(sim.t - y)))), sim.solver


def _observed(errs: list[float]) -> float:
    return float(np.log2(errs[-2] / errs[-1]))


STARTUPS = [(s, st) for s in SCHEMES for st in (("lower_order", "rk443")
                                                 if scheme_family(s) != "rk" else ("lower_order",))]


@pytest.mark.parametrize("scheme,startup", STARTUPS, ids=[f"{s}-{st}" for s, st in STARTUPS])
def test_logistic_observed_order_is_the_stated_order(scheme, startup):
    out = [_logistic_error(scheme, startup, n) for n in (40, 80, 160)]
    p = _observed([e for e, _ in out])
    stated = out[-1][1].effective_order
    # never stated above what is achieved (SBDF3/4 with the lower-order start-up are observed at
    # 3 here: the start-up's O(dt^2) term is small on this damped problem at these steps; f017 and
    # tests/ivp/test_review_numerics.py show its order-2 regime)
    assert p > stated - ORDER_TOL, f"{scheme}/{startup}: observed {p:.2f}, stated {stated}"


BC_CASES = [(s, b) for s in SCHEMES for b in ("auto", "stage", "step")]


@pytest.mark.slow
@pytest.mark.parametrize("scheme,bc_time", BC_CASES, ids=[f"{s}-{b}" for s, b in BC_CASES])
def test_moving_walls_observed_order_is_the_stated_order(scheme, bc_time):
    out = [_walls_error(scheme, bc_time, n) for n in (80, 160, 320)]
    p = _observed([e for e, _ in out])
    stated = out[-1][1].effective_order
    # no explicit term here: a Crank-Nicolson scheme may do better than its explicit part's order
    # (CNAB1 is second order on it); the relation is that the stated order is never above the
    # observed one
    assert p > stated - ORDER_TOL, (f"{scheme}/bc_time={bc_time}: observed {p:.2f}, "
                                    f"stated {stated}; errors {[e for e, _ in out]}")


FAST_BC = [("RK443", "step"), ("RK443", "stage"), ("RK222", "step"), ("RKSMR", "step"),
           ("RK443", "auto")]


@pytest.mark.parametrize("scheme,bc_time", FAST_BC, ids=[f"{s}-{b}" for s, b in FAST_BC])
def test_moving_walls_stated_order_is_achieved(scheme, bc_time):
    """The cases where the convention lowers the order (fast subset of the matrix above): the
    stated order is the observed one, and explain() says why it is below the scheme's."""
    out = [_walls_error(scheme, bc_time, n) for n in (80, 160)]
    p = _observed([e for e, _ in out])
    solver = out[-1][1]
    assert abs(p - solver.effective_order) < ORDER_TOL, (scheme, bc_time, p, solver.effective_order)
    if bc_time != "auto":
        assert f"effective global order {solver.effective_order}" in solver.explain()


def _mapped_error(scheme: str, n: int):
    h = lambda x: 0.1 * np.cos(x)                                                    # noqa: E731
    ue = lambda x, z, t: np.exp(-t) * np.sin(x) * np.cos(z) + z                      # noqa: E731
    m = cn.Model(cn.MappedBox(x=cn.periodic(2 * np.pi), z=cn.between(bottom=h, top=1.0)))
    m.scalar("u")
    m.given("g", lambda x, z: np.sin(x) * np.cos(z))
    m.given("gb", lambda x: np.sin(x) * np.cos(h(x)))
    m.given("hb", lambda x: h(x))
    m.given("gt", lambda x: np.sin(x) * np.cos(1.0))
    m.given("zz", lambda x, z: z)
    m.equations("dt(u) - lap(u) = -u*u + exp(-t)*g + (exp(-t)*g + zz)**2")
    m.at("bottom", u="exp(-t)*gb + hb")
    m.at("top", u="exp(-t)*gt + 1")
    sim = m.simulate((20, 20), scheme=scheme, dt=0.4 / n)
    sim.initial(u=lambda x, z: ue(x, z, 0.0))
    sim.run(steps=n)
    err = np.abs(sim.grid("u") - ue(sim.coordinate("x"), sim.coordinate("z"), sim.t)).max()
    return float(err), sim


@pytest.mark.slow
@pytest.mark.parametrize("scheme", ["SBDF2", "SBDF3", "SBDF4", "RK222", "RK443"])
def test_mapped_box_observed_order_is_the_stated_order(scheme):
    out = [_mapped_error(scheme, n) for n in (20, 40, 80)]
    p = _observed([e for e, _ in out])
    sim = out[-1][1]
    assert abs(p - sim.effective_order) < ORDER_TOL, (scheme, p, sim.effective_order)
    if sim.effective_order < int(scheme[-1]):
        assert f"effective global order {sim.effective_order}" in sim.explain()
