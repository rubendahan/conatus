"""Time-dependent boundary values in the native layer.

``t`` is the time of a native IVP: boundary values as text (``m.at("top", u="cos(t)")``, or the
condition text ``"u = cos(t)"``) or as ``lambda t: ...`` (numpy functions build the expression)
reach the solver as expressions of the time parameter, so the default ``bc_time="auto"`` resolves
to "rk" (exact derivative): the native model has the order of the expert one (RK443 2.76), with
the same numbers. Refusals: t in an implicit term, t in a model without dt(), a lambda of t and of
coordinates (the separable form to write is given). With "rk" the wall value of the state at t_n is
g(t_n) up to the local error of the RK quadrature of g' (not accumulated); with "stage" it is exact.

Problem: u_t = u_yy / 2 on [0, 5], u = exp(-y) cos(t - y) exact (as tests/ivp/test_bc_time.py).
"""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import BoundaryConditionError, EquationError

Y = 5.0


def _model(how):
    m = cn.Model(cn.Box(y=cn.walls(0, Y)), name="moving walls")
    m.scalar("u")
    m.equations("dt(u) - dy(dy(u))/2 = 0")
    if how == "text":
        m.at("bottom", u="cos(t)")
        m.at("top", f"u = {float(np.exp(-Y))!r}*cos(t - {Y})")
    else:
        m.at("bottom", u=lambda t: np.cos(t))
        m.at("top", u=lambda t: np.exp(-Y) * np.cos(t - Y))
    return m


def _error(how, dt, **kw):
    sim = _model(how).simulate(N=32, scheme="RK443", dt=dt, dealias=1, **kw)
    sim.initial(u=lambda y: np.exp(-y) * np.cos(y))
    sim.run(until=1.0)
    y = np.asarray(sim.coordinate("y"))
    return float(np.abs(sim.solver.grid("u") - np.exp(-y) * np.cos(sim.t - y)).max()), sim


@pytest.mark.parametrize("how", ["text", "lambda"])
def test_native_moving_walls_reach_the_rk_order(how):
    e1, sim = _error(how, 0.02)
    e2, _ = _error(how, 0.01)
    assert np.log2(e1 / e2) > 2.6, (e1, e2)                 # 2.76, as the expert layer
    assert sim.solver.time_parameter == "t" and sim.solver.bc_time == "rk"
    txt = sim.explain(full=True)
    assert "Time: t appears" in txt and "bc_time='rk', default 'auto'" in txt
    e_stage, s2 = _error(how, 0.01, bc_time="stage")
    assert s2.solver.bc_time == "stage" and e_stage > 10 * e2


def test_native_forcing_in_t_and_refusals():
    m = cn.Model(cn.Box(y=cn.walls(0, 1)), name="forced")
    m.scalar("u")
    m.equations("dt(u) - dy(dy(u)) = sin(t)")
    m.at("walls", u=0)
    sim = m.simulate(N=12, scheme="SBDF2", dt=1e-2)
    assert sim.solver.time_parameter == "t"
    sim.run(steps=3)
    m2 = cn.Model(cn.Box(y=cn.walls(0, 1)), name="implicit t")
    m2.scalar("u")
    m2.equations("dt(u) - dy(dy(u)) + t*u = 0")
    m2.at("walls", u=0)
    with pytest.raises(EquationError, match="multiplies an unknown in an implicit term") as ei:
        m2.simulate(N=12, scheme="SBDF2", dt=1e-2)
    assert "explicit(" in str(ei.value)
    m3 = cn.Model(cn.Box(y=cn.walls(0, 1)), name="no time")
    m3.scalar("u")
    m3.equations("dy(dy(u)) = 0")
    m3.at("bottom", u="cos(t)")
    m3.at("top", u=0)
    with pytest.raises(EquationError, match="the time t appears in a LBVP"):
        m3.solve(N=12)


def test_native_lambda_of_t_and_coordinates_is_refused_with_the_separable_form():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="travelling wall")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = 0")
    m.at("bottom", u=lambda x, t: np.sin(x - t))
    m.at("top", u=0)
    with pytest.raises(BoundaryConditionError, match="a function of t and of") as ei:
        m.simulate(N=(16, 16), scheme="RK222", dt=1e-2)
    assert "S*cos(t) - C*sin(t)" in str(ei.value)
    # the separable form, with declared data: exact trace sin(x - t) at the wall
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, 1)), name="travelling wall")
    m.scalar("u")
    m.given(S=lambda x: np.sin(x), C=lambda x: np.cos(x))
    m.equations("dt(u) - lap(u) = 0")
    m.at("bottom", u="S*cos(t) - C*sin(t)")
    m.at("top", u=0)
    for bc_time, tol in (("auto", 1e-7), ("stage", 1e-12)):
        sim = m.simulate(N=(16, 16), scheme="RK222", dt=1e-2, bc_time=bc_time)
        sim.run(steps=30)
        x, z = (np.asarray(sim.coordinate(c)).ravel() for c in ("x", "z"))
        c = np.polynomial.chebyshev.chebfit(2 * z - 1, sim.solver.grid("u").T, 15)
        trace = np.polynomial.chebyshev.chebval(-1, c)
        # "rk": the wall value of the state at t_n is g(t_n) up to the local error of the RK
        # quadrature of g' (1e-8 here, not accumulated: each step starts from g(t_n));
        # "stage": exact
        assert np.abs(trace - np.sin(x - sim.t)).max() < tol, bc_time
