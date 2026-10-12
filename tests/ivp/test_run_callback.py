"""``IVPSolver.run(n, callback=cb)`` without ``every`` used to never call ``cb`` (every=0).

Decision (safest, no hidden choice): a callback without an explicit ``every`` is refused with an
``OptionError`` naming the fix, and so is a non-positive ``every`` or an ``every`` without a
callback. Nothing is stepped before the refusal."""
import numpy as np
import pytest

from ivp_cases import burgers_1d, rayleigh_benard_2d

from conatus.errors import OptionError
from conatus.timestep import IVPSolver, TauWatch


def _solver():
    plan, f = rayleigh_benard_2d(16, 8, Ra=1e5, Lx=2.0)
    s = IVPSolver(plan, "RK222", 1e-3)
    s.set_state({f["T"]: lambda x, z: 1e-2 * np.cos(np.pi * z) * np.cos(np.pi * x)})
    return s


def test_callback_without_every_is_refused_before_stepping():
    s = _solver()
    tw = TauWatch(s, growth_limit=1e6)
    with pytest.raises(OptionError, match="every"):
        s.run(5, callback=tw)
    assert s.iteration == 0                      # refused up front, nothing advanced
    assert all(len(h) == 0 for h in tw.history.values())


@pytest.mark.parametrize("every", [0, -1])
def test_non_positive_every_is_refused(every):
    s = _solver()
    with pytest.raises(OptionError, match="every"):
        s.run(2, callback=lambda _s: None, every=every)


def test_every_without_callback_is_refused():
    s = _solver()
    with pytest.raises(OptionError, match="callback"):
        s.run(2, every=1)


def test_callback_called_every_k_steps():
    s = _solver()
    seen = []
    s.run(6, callback=lambda sv: seen.append(sv.iteration), every=2)
    assert seen == [2, 4, 6]
    s.run(2)                                     # no callback, no every: plain stepping
    assert s.iteration == 8


def test_every_solver_checks_dt_and_backend_alike():
    """The Cartesian solver refused a bad backend with an OptionError and its
    fix, the polar and spherical ones with a bare ValueError (no fix), the distributed one with
    another text; dt = 0, nan or a string reached float() or a comparison. One check now."""
    import conatus as cn
    from conatus.errors import InputTypeError, OptionError, UnsupportedError
    from conatus.timestep import IVPSolver
    disk = cn.Model(cn.Disk(1.0))
    disk.scalar("T")
    disk.equations("dt(T) - lap(T) = 0")
    disk.at("wall", "T = 0")
    plans = [burgers_1d(N=16)[0], disk.discretize((8, 8))]
    for plan in plans:
        for dt in (0, -1e-3, float("nan"), "1e-3", True):
            with pytest.raises(OptionError, match="dt=") as err:
                IVPSolver(plan, "SBDF2", dt)
            assert err.value.fix
        with pytest.raises(OptionError, match=r"spelled 'cuda' here \(not 'gpu'\)"):
            IVPSolver(plan, "SBDF2", 1e-3, backend="gpu")
    with pytest.raises(UnsupportedError, match="f_backend"):
        IVPSolver(plans[1], "SBDF2", 1e-3, f_backend="fused")
    with pytest.raises(InputTypeError, match="unexpected arguments"):
        IVPSolver(plans[1], "SBDF2", 1e-3, decouple=True)
    s = IVPSolver(plans[0], "RK443", 1e-3)
    with pytest.raises(OptionError, match="dt="):
        s.set_dt(0.0)


def test_changing_plan_params_after_construction_is_refused():
    """``solver.plan.params['Ra'] = 4000`` after construction was ignored by
    the factored L, M (5 steps: bit-identical to Ra = 2000) while F reads plan.params at every
    stage; a parameter on both sides would be half updated. Refused at the next step."""
    from conatus.discretize import discretize
    plan, f = rayleigh_benard_2d(8, 8, 2000.0)
    s = IVPSolver(discretize(plan.problem, params={"Ra": 2000.0}), "SBDF1", 1e-3)
    s.run(1)
    s.plan.params["Ra"] = 4000.0
    with pytest.raises(OptionError, match=r"Ra: 2000.0 -> 4000.0"):
        s.run(1)
    s.plan.params["Ra"] = 2000.0
    s.run(1)
    assert s.iteration == 2


def test_ivpsolver_signature_is_public():
    """inspect.signature(IVPSolver) gives the real arguments (it showed the dispatching
    __new__'s (plan=None, *args, **kwargs): API reference, BACKLOG UX)."""
    import inspect
    from conatus.timestep import IVPSolver
    params = list(inspect.signature(IVPSolver).parameters)
    assert params[:3] == ["plan", "scheme", "dt"] and "time_parameter" in params
