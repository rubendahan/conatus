"""Exact checkpoint/restart, CFL vs embedded error, agreement with the C++ ImexIntegrator."""
import numpy as np
import pytest

from ivp_cases import burgers_1d, heat_1d, rayleigh_benard_2d
from conatus.reference import transforms as T
from conatus.runtime import ImexIntegrator
from conatus.timestep import CFL, EmbeddedError, IVPSolver


def _rb_t0(X, Z):
    return -Z + 1e-2 * np.cos(np.pi * X) * np.cos(np.pi * Z)


@pytest.mark.parametrize("scheme,startup", [("SBDF3", "rk443"), ("SBDF4", "lower_order"),
                                            ("RK443", "lower_order")])
def test_restart_is_bitwise(tmp_path, scheme, startup):
    plan, f = rayleigh_benard_2d(16, 12, Ra=1e5)
    a = IVPSolver(plan, scheme, 2e-4, startup=startup)
    a.set_state({f["T"]: _rb_t0})
    a.run(12)
    b = IVPSolver(plan, scheme, 2e-4, startup=startup)
    b.set_state({f["T"]: _rb_t0})
    b.run(5)                                     # interrupted inside / after the startup
    ck = b.save_checkpoint(tmp_path / "ck")
    c = IVPSolver(plan, scheme, 2e-4, startup=startup)
    c.load_checkpoint(ck)
    c.run(7)
    assert np.array_equal(a.X, c.X)
    assert a.t == c.t and a.iteration == c.iteration


def test_restart_refuses_changed_numerics(tmp_path):
    plan, u, ex = heat_1d()
    a = IVPSolver(plan, "SBDF2", 1e-3)
    a.set_state({u: lambda X: ex(X, 0)})
    a.run(3)
    ck = a.save_checkpoint(tmp_path / "ck.npz")
    with pytest.raises(ValueError, match="dt"):
        IVPSolver(plan, "SBDF2", 2e-3).load_checkpoint(ck)


def test_matches_cpp_integrator():
    """Same global M, L and F: the Python step loop and the C++ ImexIntegrator agree."""
    plan, u, ex = heat_1d(N=20)
    for scheme in ("SBDF3", "RK443"):
        s = IVPSolver(plan, scheme, 0.01, ordering="variable-major")
        s.set_state({u: lambda X: ex(X, 0)})
        x0 = s.X.copy()
        s.run(15)
        border = [i for i, k in enumerate(s.system.slots[0].row_kind) if k != "bulk"]
        core = ImexIntegrator(s.system.M, s.system.L, border, scheme, 0.01,
                              F=lambda x, t: s.F(np.asarray(x), t))
        x, t = core.run(x0, 0.0, 15)
        assert np.max(np.abs(x - s.X)) < 1e-12 * np.max(np.abs(x))


def test_cfl_is_not_an_error_estimate():
    plan, f = rayleigh_benard_2d(32, 16, Ra=1e5)
    s = IVPSolver(plan, "RK443", 2e-4)
    s.set_state({f["T"]: _rb_t0})
    s.run(200)
    cfl = CFL(s, "u", safety=0.5)
    dt_cfl = cfl.compute()
    assert np.isfinite(dt_cfl) and dt_cfl > 0
    assert s.dt == 2e-4                          # never changes dt
    assert "not an error estimate" in cfl.explain()


def test_embedded_error_scales_with_order():
    plan, u, ex = burgers_1d(N=64)
    errs = []
    for dt in (0.02, 0.01):
        s = IVPSolver(plan, "RK443", dt)
        s.set_state({u: lambda X: ex(X, 0)})
        e = EmbeddedError(s)
        errs.append(e.step())
        assert s.dt == dt                        # report only
    # companion RK222: local error O(dt^3)
    assert 2.5 < np.log2(errs[0] / errs[1]) < 3.5


def test_adaptive_only_on_request():
    plan, u, ex = burgers_1d(N=64)
    s = IVPSolver(plan, "RK222", 0.05)
    s.set_state({u: lambda X: ex(X, 0)})
    e = EmbeddedError(s, adapt=True, tol=1e-8, safety=0.9)
    for _ in range(5):
        e.step()
    assert s.dt < 0.05
    X = T.grid(plan.domain.bases[0], dealias=False)
    assert np.max(np.abs(s.grid("u") - ex(X, s.t))) < 1e-6
    sb = IVPSolver(plan, "SBDF2", 0.05)
    with pytest.raises(ValueError):
        EmbeddedError(sb, adapt=True, tol=1e-6, safety=0.9)
    with pytest.raises(ValueError):
        EmbeddedError(s, adapt=True, safety=0.9)  # tolerance must be explicit
