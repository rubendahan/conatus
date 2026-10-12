"""Root and Newton results say whether they are roots.

* Y8: ``secant_root`` returns a SecantRoot (converged, residual, iterations, reasons) and raises
  ConvergenceError when unconverged unless ``allow_unconverged=True``; the v0.1 tuple unpacking
  keeps working with a DeprecationWarning. ``find_root`` (Hill, cascade) raises the same way.
* Y9: ``NewtonSolver`` landing on u = 0 flags ``trivial=True``; ``find_travelling_wave`` from
  c0 = 0 onto an equilibrium labels the result an equilibrium.
"""
import numpy as np
import pytest

import conatus as cn
from conatus.errors import ConvergenceError, OptionError
from conatus.floquet.hill_general import HillBlocks, secant_root
from conatus.floquet.roots import SecantRoot


def _mathieu(a, q, Nh=12):
    """Mathieu equation x'' + (a - 2 q cos 2t) x = 0 as a first-order Hill system."""
    def diag(mu, n):
        s = mu + 2j * n
        return np.array([[s, -1.0], [a, s]], dtype=complex)
    C = np.array([[0.0, 0.0], [-q, 0.0]], dtype=complex)
    return HillBlocks(diag=diag, couplings={1: C, -1: C}, Nh=Nh, size=2, label="Mathieu",
                      omega=2.0, notes=("test",))


def test_secant_root_result_object_and_unconverged_raise():
    H = _mathieu(-0.4, 0.5)
    r = secant_root(H, 0.3 + 0.1j, tol=1e-12, max_iterations=60)
    assert isinstance(r, SecantRoot) and r.converged and r.iterations >= 1
    assert r.residual < 1e-6 and not r.reasons and "converged" in r.explain()
    with pytest.raises(ConvergenceError, match="not converged") as e:
        secant_root(H, 0.3 + 0.1j, tol=1e-13, max_iterations=2)
    assert isinstance(e.value.result, SecantRoot) and not e.value.result.converged
    u = secant_root(H, 0.3 + 0.1j, tol=1e-13, max_iterations=2, allow_unconverged=True)
    assert not u.converged and u.reasons and "NOT converged" in u.explain()
    with pytest.raises(OptionError):
        secant_root(H, 0.3 + 0.1j, tol=1e-12, allow_unconverged="yes")


def test_secant_root_tuple_unpacking_is_deprecated_not_broken():
    H = _mathieu(-0.4, 0.5)
    r = secant_root(H, 0.3 + 0.1j, tol=1e-12, max_iterations=60)
    with pytest.warns(DeprecationWarning, match="r.mu and r.reasons"):
        mu, why = r
    assert mu == r.mu and why == []
    with pytest.warns(DeprecationWarning):
        assert r[0] == r.mu


def _ac(guess_amp):
    from conatus.expert import NewtonSolver
    m = cn.Model(cn.Box(x=cn.walls(0, 10)), name="AC", lam=1.0)
    m.scalar("u")
    m.equations("dx(dx(u)) + lam*u*(1 - u**2) = 0")
    m.at("walls", u=0)
    pb = m.compile(48)
    nw = NewtonSolver(pb.problem, tol=1e-11, params={"lam": 1.0})
    return nw.solve({pb.fields["u"]: lambda x: guess_amp * np.sin(np.pi * x / 10)})


def test_newton_flags_the_trivial_state():
    s = _ac(1e-6)
    assert s.converged and s.trivial
    assert "converged to the trivial state" in s.explain()
    big = _ac(1.0)
    assert big.converged and not big.trivial
    assert float(np.abs(big.grid("u")).max()) > 0.9
    assert "trivial-state check" in big.explain()


def test_travelling_wave_with_zero_speed_is_an_equilibrium():
    from conatus.expert import Equilibrium, NewtonKrylovHookstep, find_travelling_wave
    m = cn.Model(cn.Box(x=cn.periodic(22.0)), name="KS")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=0.01)
    sim.initial(u=lambda x: np.sin(2 * np.pi * x / 22.0))
    x0 = sim.solver.X.copy()
    tw = find_travelling_wave(sim, x0, 0.0, NewtonKrylovHookstep(tol=1e-10))
    assert isinstance(tw, Equilibrium) and tw.kind == "equilibrium"
    assert abs(tw.params["c"]) < 1e-8
    assert "this is an equilibrium" in tw.explain()
