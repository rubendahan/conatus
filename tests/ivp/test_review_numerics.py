"""Regression tests of the external review, lot B "numerics": time stepping.

Each test reproduces the reviewer's script (finding number in the
test name) and fails on v0.1.0.
"""
import warnings

import numpy as np
import pytest

from conatus.discretize import discretize
from conatus.errors import DtypeError
from conatus.ir import IVP, CartesianCoordinates, Chebyshev, Domain, Field, dt
from conatus.timestep import IVPSolver


def _cheb_field(dtype):
    cs = CartesianCoordinates("x")
    b = Chebyshev(cs["x"], 8, (-1, 1), dealias=1)
    return Field("u", Domain(cs, [b]), dtype=dtype)


def test_f2_complex_field_keeps_imaginary_part():
    """Finding 2: a field declared complex is solved in a complex system (was float64, zeros)."""
    u = _cheb_field(complex)
    pb = IVP([u])
    pb.add_equation((dt(u), 1j))
    s = IVPSolver(discretize(pb), "RK222", .1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        s.set_state({u: lambda x: 1j})
        s.run(1)
    assert np.dtype(s.system.dtype).kind == "c"
    assert np.allclose(s.grid("u"), 1.1j, atol=1e-13)


def test_f2_complex_equation_on_real_field_refused():
    """Finding 2: complex coefficients on a field declared real are refused at build time."""
    u = _cheb_field(np.float64)
    pb = IVP([u])
    pb.add_equation((dt(u), 1j))
    with pytest.raises(DtypeError, match="declared real"):
        IVPSolver(discretize(pb), "RK222", .1)


def test_f2_complex_initial_data_on_real_field_refused():
    """Finding 2: initial data with an imaginary part is refused for a real field (lossy cast);
    data that is complex-typed but exactly real is accepted without a ComplexWarning."""
    u = _cheb_field(np.float64)
    pb = IVP([u])
    pb.add_equation((dt(u), 0))
    s = IVPSolver(discretize(pb), "RK222", .1)
    with pytest.raises(DtypeError, match="imaginary part"):
        s.set_state({u: lambda x: 1j + 0 * x})
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        s.set_state({u: lambda x: (1 + 0j) + 0 * x})
    assert np.allclose(s.grid("u"), 1.0)


# ---------------------------------------------------------------------------------- finding 10
def _riccati(dt):
    import conatus as cn
    m = cn.Model(cn.Box(x=cn.walls(-1, 1)))
    m.scalar("u")
    m.equations("dt(u) = -u*u")
    return m.simulate(4, scheme="RK222", dt=dt).initial(u=1)


def test_f10_adaptive_step_above_tol_is_rejected_and_retried():
    """Finding 10: the controller kept a step 3e7 times above tol; it now rejects it, restores
    the state and retries with a smaller dt until the estimate is below tol."""
    from conatus.timestep import EmbeddedError
    s = _riccati(.5)
    e = EmbeddedError(s.solver, adapt=True, tol=1e-8, safety=.9)
    error = e.step()
    assert error <= e.tol
    assert e.rejected > 0 and s.solver.iteration == 1
    assert 0 < s.t < .5
    assert abs(s.grid("u")[0] - 1 / (1 + s.t)) < 1e-7
    for _ in range(5):                                      # later steps are accepted too
        assert e.step() <= e.tol
    assert abs(s.grid("u")[0] - 1 / (1 + s.t)) < 1e-6


def test_f10_tol_unreachable_at_dt_min_is_refused():
    """Finding 10: never accept a step above tol: ConvergenceError at dt_min, state restored."""
    from conatus.errors import ConvergenceError
    from conatus.timestep import EmbeddedError
    s = _riccati(.5)
    e = EmbeddedError(s.solver, adapt=True, tol=1e-8, safety=.9, dt_min=.25)
    with pytest.raises(ConvergenceError, match="dt_min"):
        e.step()
    assert s.t == 0 and s.solver.iteration == 0 and np.allclose(s.grid("u"), 1.0)


@pytest.mark.parametrize("kw", [dict(tol=0.0), dict(tol=np.nan), dict(tol=1e-6, dt_min=1., dt_max=.5),
                                dict(tol=1e-6, dt_min=-1.)])
def test_f10_tol_and_bounds_validated_first(kw):
    """Finding 10: tol and the dt bounds are validated before any step."""
    from conatus.errors import OptionError
    from conatus.timestep import EmbeddedError
    s = _riccati(.1)
    with pytest.raises(OptionError):
        EmbeddedError(s.solver, adapt=True, safety=.9, **kw)
    assert s.solver.iteration == 0


# ---------------------------------------------------------------------------------- finding 17
def _heat_burgers(scheme, dt, startup=None):
    import conatus as cn
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u) - dx(dx(u)) = -0.5*u*u")
    kw = {} if startup is None else {"startup": startup}
    s = m.simulate(16, scheme=scheme, dt=dt, **kw).initial(u=lambda x: 1 + np.sin(x))
    s.run(steps=round(1 / dt))
    return s


@pytest.mark.parametrize("scheme, startup, order", [("SBDF3", "lower_order", 2),
                                                    ("SBDF4", "lower_order", 2),
                                                    ("SBDF3", "rk443", 3)])
def test_f17_effective_order_is_stated_and_observed(scheme, startup, order):
    """Finding 17: with the default lower-order startup SBDF3/4 converge at order 2 (the
    first-order first step); the solver states the effective order (property and explain())
    and it matches the observed order (u_t = u_xx - u^2/2, reference RK443 at dt = 1e-3)."""
    ref = _heat_burgers("RK443", 1e-3).grid("u")
    runs = [_heat_burgers(scheme, dt, startup) for dt in (0.02, 0.01)]
    e1, e2 = (np.max(np.abs(r.grid("u") - ref)) for r in runs)
    observed = np.log2(e1 / e2)
    assert runs[0].solver.effective_order == order
    assert abs(observed - order) < 0.3, observed
    text = runs[0].solver.explain()
    if order < int(scheme[4:]):
        assert f"effective global order {order}" in text
    else:
        assert "effective global order" not in text


def test_f17_default_startup_is_lower_order_and_said():
    """Finding 17: the default stays the declared Dedalus convention; explain() says its cost."""
    s = _heat_burgers("SBDF3", 0.1)
    assert s.solver.startup == "lower_order" and s.solver.effective_order == 2
    assert "effective global order 2" in s.solver.explain(full=True)


# ------------------------------------------------------------------------------------ F101
def test_f101_constant_field_is_not_under_resolved():
    """F101 (-10-16): u = 1 on 4 Fourier modes is exact, yet the run check warned
    "under-resolved, tail 1.0" (its tail window covered the mean); the window is not judged."""
    import conatus as cn
    from conatus.errors import UnderResolvedWarning
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations("dt(u) + u = 0")
    s = m.simulate(4, scheme="RK222", dt=0.1).initial(u=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnderResolvedWarning)
        s.run(steps=3)
    assert "under-resolved" not in s.explain()
