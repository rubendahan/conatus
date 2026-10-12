"""Time-dependent boundary data. ``bc_time="rk"`` (stage values from the scheme's
implicit tableau applied to the exact g'(t)), ``"stage"`` (g at the stage times), ``"step"`` (carried
by F with the explicit weights, the Dedalus convention); default ``"auto"`` = "rk" when g'(t) is
exact, else "stage" (declared).

Problem: u_t = u_yy / 2 on [0, 5], u = exp(-y) cos(t - y) exact, both boundary values moving
(Chebyshev N = 32: spatial error < 1e-13), errors at t = 1. Checks: observed orders of all
thirteen schemes in the three modes, bit identity of "step" with no imposition and of the two modes for
constant data, explain(), refusals, checkpoint mismatch, and the exact tangent/adjoint of FlowMap
with the boundary rows imposed at the stage times (dot product, Taylor).
"""
import numpy as np
import pytest

from conatus.adjoint import FlowMap
from conatus.discretize import discretize
from conatus.errors import OptionError
from conatus.ir import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, cos, diff,
                        dt, interp)
from conatus.reference import grid
from conatus.timestep import IVPSolver

Y = 5.0


def _problem(moving=True, state_dependent=False):
    cs = CartesianCoordinates("y")
    y = cs["y"]
    by = Chebyshev(y, 32, (0, Y), dealias=1)
    u, t = Field("u", Domain(cs, [by])), Parameter("t", 0.0)
    pb = IVP([u])
    pb.add_equation((dt(u) - diff(diff(u, y), y) / 2, 0))
    left = cos(t) if moving else 1.0
    if state_dependent:
        left = cos(t) + 1e-3 * interp(u, y, Y)
    pb.add_bc((interp(u, y, 0), left), (interp(u, y, Y), np.exp(-Y) * cos(t - Y)))
    return discretize(pb), u, by


def _error(scheme, step, bc_time, startup="lower_order"):
    plan, u, by = _problem()
    s = IVPSolver(plan, scheme, step, time_parameter="t", bc_time=bc_time, startup=startup)
    s.set_state({u: lambda Yg: np.exp(-Yg) * np.cos(Yg)})
    s.run(int(round(1 / step)))
    Yg = grid(by, dealias=False)
    return float(np.abs(s.grid("u") - np.exp(-Yg) * np.cos(s.t - Yg)).max())


# scheme -> (lowest order accepted with "rk", lowest with "stage", highest with "step" or None).
# Measured (dt = 0.02 -> 0.01 -> 0.005):
#   rk    RK111 1.02/1.01, RK222 2.00/2.00, RK443 2.76/2.81, RKSMR 2.00/2.00, RKGFY 2.00/2.00;
#         multistep schemes identical to "stage" (SBDF3/4: RK443 startup substeps differ, same order)
#   stage RK111 1.00, RK222 1.99/1.98, RK443 2.02/1.97 (stage order of its DIRK part), RKSMR
#         1.94/1.90, RKGFY 2.00, SBDF1-4 1.00/2.01/2.98/3.96, CNAB1 2.00 (L only here),
#         CNAB2/MCNAB2/CNLF2 2.00
#   step  RK222/RK443/RKSMR 1.05-1.2, CNAB1 1.00, CNAB2 1.32/0.74, CNLF2 2.83/1.42
ORDERS = {"RK111": (0.95, 0.95, None), "RK222": (1.95, 1.9, 1.3), "RK443": (2.6, 1.9, 1.3),
          "RKSMR": (1.95, 1.85, 1.3), "RKGFY": (1.95, 1.95, None), "SBDF1": (0.95, 0.95, None),
          "SBDF2": (1.95, 1.95, None), "SBDF3": (2.9, 2.9, None), "SBDF4": (3.85, 3.85, None),
          "CNAB1": (1.9, 1.9, 1.1), "CNAB2": (1.9, 1.9, 1.5), "MCNAB2": (1.9, 1.9, None),
          "CNLF2": (1.9, 1.9, None)}


def _orders(scheme, bc_time):
    startup = "rk443" if scheme in ("SBDF3", "SBDF4") else "lower_order"
    e = [_error(scheme, h, bc_time, startup) for h in (0.02, 0.01)]
    return float(np.log2(e[0] / e[1])), e


@pytest.mark.parametrize("scheme", sorted(ORDERS))
def test_orders_in_the_three_conventions(scheme):
    lo_rk, lo_stage, hi_step = ORDERS[scheme]
    p, e = _orders(scheme, "rk")
    assert p > lo_rk, (p, e)
    p_auto, e_auto = _orders(scheme, "auto")              # default: "rk" (g' exact here)
    assert e_auto == e, (e_auto, e)
    p, e = _orders(scheme, "stage")
    assert p > lo_stage, (p, e)
    if hi_step is not None:                               # the Dedalus convention loses order
        p, e = _orders(scheme, "step")
        assert p < hi_step, (p, e)


@pytest.mark.parametrize("scheme", ["RK443", "SBDF2", "CNAB2"])
def test_step_convention_is_untouched_and_constant_data_identical(scheme, monkeypatch):
    plan, u, _ = _problem()
    init = {u: lambda Yg: np.exp(-Yg) * np.cos(Yg)}

    def run(bc_time, plan=plan, patch=False):
        s = IVPSolver(plan, scheme, 1e-2, time_parameter="t", bc_time=bc_time)
        if patch:
            monkeypatch.setattr(s, "impose_bc", lambda rhs, t, beta, G=None: rhs)
        s.set_state(init)
        s.run(7)
        return s.X.copy()
    assert np.array_equal(run("step"), run("stage", patch=True))      # step = no row touched
    assert np.array_equal(run("step"), run("rk", patch=True))
    assert not np.array_equal(run("step"), run("stage"))
    if scheme == "RK443":                                  # rk differs from stage on RK only
        assert not np.array_equal(run("rk"), run("stage"))
    else:                                                  # multistep: g(t_n+1) in both
        assert np.array_equal(run("rk"), run("stage"))
    plan_c, u, _ = _problem(moving=False)                              # constant left value
    init = {u: lambda Yg: np.exp(-Yg) * np.cos(Yg)}
    s1 = IVPSolver(plan_c, scheme, 1e-2, bc_time="step")
    s2 = IVPSolver(plan_c, scheme, 1e-2)
    s3 = IVPSolver(plan_c, scheme, 1e-2, bc_time="stage")
    for s in (s1, s2, s3):
        s.set_state(init)
        s.run(7)
    assert np.array_equal(s1.X, s2.X) and np.array_equal(s1.X, s3.X)
    assert s2.bc_rows() is None                                      # no time parameter: no-op


def test_explain_refusals_and_checkpoint(tmp_path):
    plan, u, _ = _problem()
    s = IVPSolver(plan, "RK443", 1e-2, time_parameter="t")
    txt = s.explain(full=True)
    assert s.bc_time == "rk" and "equation(s) 1, 2" in txt
    assert "applied to the exact d/dt of the data (bc_time='rk', default 'auto'" in txt
    st = IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time="stage")
    assert st.bc_time == "stage" and "imposed at every stage time (bc_time='stage';" in st.explain(full=True)
    assert "Dedalus convention (bc_time='step')" in IVPSolver(plan, "RK443", 1e-2, time_parameter="t",
                                                              bc_time="step").explain(full=True)
    with pytest.raises(OptionError, match="bc_time='sub'"):
        IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time="sub")
    plan_m, _, _ = _problem(state_dependent=True)
    for mode in ("auto", "rk", "stage"):
        with pytest.raises(OptionError, match="depends on both t and the unknowns"):
            IVPSolver(plan_m, "RK443", 1e-2, time_parameter="t", bc_time=mode).step()
    IVPSolver(plan_m, "RK443", 1e-2, time_parameter="t", bc_time="step").step()   # Dedalus: allowed
    s.set_state({u: lambda Yg: np.exp(-Yg) * np.cos(Yg)})
    s.run(2)
    ck = s.save_checkpoint(tmp_path / "ck")
    for other in ("step", "stage"):                   # saved with "rk": another mode is refused
        s2 = IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time=other)
        with pytest.raises(ValueError, match="bc_time"):
            s2.load_checkpoint(ck)
    s3 = IVPSolver(plan, "RK443", 1e-2, time_parameter="t")          # same mode: bit for bit
    s3.load_checkpoint(ck)
    s.run(3)
    s3.run(3)
    assert np.array_equal(s.X, s3.X)


@pytest.mark.parametrize("scheme,bc_time", [("RK443", "auto"), ("RK443", "stage"),
                                            ("RKSMR", "rk"), ("SBDF2", "auto"), ("CNAB2", "auto")])
def test_flowmap_exact_with_boundary_data_at_stage_times(scheme, bc_time):
    """The boundary rows hold beta g(t_stage): zero rows in the tangent's right-hand side and in
    the adjoint's multiplier. Dot product to rounding, Taylor remainder of order 2 (the map is
    affine in X here: the remainder is rounding; checked on the nonlinear case by the order
    tests of the forward map)."""
    plan, u, _ = _problem()
    s = IVPSolver(plan, scheme, 1e-2, time_parameter="t", bc_time=bc_time)
    fm = FlowMap(s, 5, t0=0.2)
    assert f"boundary data imposed at the stage times (bc_time={s.bc_time!r})" in fm.explain()
    rng = np.random.default_rng(3)
    n = fm.n
    X0, V, W = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    JV, JtW = fm.jvp(X0, V), fm.vjp(X0, W)
    a, b = float(W @ JV), float(JtW @ V)
    assert abs(a - b) <= 1e-12 * np.linalg.norm(W) * np.linalg.norm(JV)
    # the map is affine: Phi(X0 + h V) - Phi(X0) = h J V exactly up to rounding
    P0 = fm(X0)
    for h in (1e-1, 1.0):
        assert np.linalg.norm(fm(X0 + h * V) - P0 - h * JV) <= 1e-10 * h * np.linalg.norm(JV)


def test_rk_needs_an_exact_derivative_and_auto_says_why():
    """'auto' falls back to 'stage' when g'(t) is not exact (here sign(t - 0.5), no
    derivative), and says why; an explicit 'rk' is refused with what to write instead."""
    from conatus.ir import sign
    cs = CartesianCoordinates("y")
    y = cs["y"]
    by = Chebyshev(y, 16, (0, Y), dealias=1)
    u, t = Field("u", Domain(cs, [by])), Parameter("t", 0.0)
    pb = IVP([u])
    pb.add_equation((dt(u) - diff(diff(u, y), y) / 2, 0))
    pb.add_bc((interp(u, y, 0), sign(t - 0.5)), (interp(u, y, Y), 0))
    plan = discretize(pb)
    s = IVPSolver(plan, "RK443", 1e-2, time_parameter="t")
    assert s.bc_time == "stage"
    txt = s.explain(full=True)
    assert "'auto' fallback: no exact time derivative of the right-hand side of equation 1" in txt
    assert "sign()" in txt
    with pytest.raises(OptionError, match="bc_time='rk': no exact time derivative") as ei:
        IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time="rk").step()
    assert "bc_time='stage'" in str(ei.value)
    s.set_state({})
    s.run(3)                                              # the fallback runs as 'stage'
    s2 = IVPSolver(plan, "RK443", 1e-2, time_parameter="t", bc_time="stage")
    s2.set_state({})
    s2.run(3)
    assert np.array_equal(s.X, s2.X)


def test_stage_values_follow_the_implicit_tableau():
    """bc_time='rk': the boundary value carried by stage i is g(t) + dt sum_j H_ij g'(t + c_j dt),
    checked on the stage states of one RK222 step (row of u(0))."""
    from conatus.timestep.integrator import rk_tableau
    plan, u, by = _problem()
    s = IVPSolver(plan, "RK222", 0.1, time_parameter="t")
    s.set_state({u: lambda Yg: np.exp(-Yg) * np.cos(Yg)}, t=0.3)
    A, H, c = rk_tableau("RK222")
    X = s.rk_stages(s.X, 0.3, 0.1)
    for i in (1, 2):
        s.X = X[i]
        u0 = float(np.polynomial.chebyshev.chebval(-1.0, s.coeffs("u").real))
        want = np.cos(0.3) + 0.1 * sum(H[i][j] * -np.sin(0.3 + c[j] * 0.1) for j in range(i + 1))
        assert abs(u0 - want) < 1e-12, (i, u0, want)
