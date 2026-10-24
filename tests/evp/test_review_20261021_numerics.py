"""External review of 0.1.1, findings F7, F2, F8.

Oracles: SciPy's eigenvalues of a nonsingular pencil (F7), the closed-form eigenvalues -1 and
1e-14 r of a diagonal problem (F2), NumPy's eigenvalues of the Jacobian of the Bogdanov-Takens
normal form at each point of its Hopf curve (F8).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigvals

import conatus as cn
from conatus.errors import ParameterError
from conatus.expert import solve_dense


@pytest.mark.parametrize("big", [1e10, 1e12, 1e13, 1e14])
def test_f7_nonsingular_mass_keeps_every_finite_eigenvalue(big):
    """F7: L = -diag(1, big), M = I: both eigenvalues finite (M nonsingular); 1e13 was dropped."""
    L, M = -np.diag([1.0, big]), np.eye(2)
    exact = np.sort(eigvals(-L, M).real)
    r = solve_dense(L, M)
    assert r.n_infinite == 0 and r.values.size == 2, (r.values, r.log)
    assert np.allclose(np.sort(r.values.real), exact, rtol=1e-12, atol=0)
    assert "nonsingular" in r.log[-1]


def test_f7_eigenvalue_beyond_one_over_eps_is_removed_and_flagged():
    """Audit of 0.1.2, F03: L = -diag(1, 1e16), M = I is, row by row, the pencil
    L = -I, M = diag(1, 1e-16) (second equation divided by 1e16), whose mass row is below n eps of
    its L row. The classification is the same for both (invariant to equation scaling): the
    eigenvalue 1e16 (scipy.linalg.eigvals) is removed as numerically infinite and reported in
    ``infinite_ambiguous`` with its modulus, and named in the log; it is not dropped silently."""
    exact = np.sort(eigvals(np.diag([1.0, 1e16]), np.eye(2)).real)
    for L, M in ((-np.diag([1.0, 1e16]), np.eye(2)), (-np.eye(2), np.diag([1.0, 1e-16]))):
        r = solve_dense(L, M)
        assert r.values.size == 1 and r.n_infinite == 1, (r.values, r.log)
        assert abs(r.values[0] - exact[0]) < 1e-12
        assert r.infinite_ambiguous.size == 1
        assert r.infinite_ambiguous[0] == pytest.approx(exact[1], rel=1e-10)
        assert "ambiguous" in r.log[-1] and "1.000e+16" in r.log[-1], r.log[-1]


def test_f7_singular_mass_still_removes_its_infinite_eigenvalue():
    """M with one zero row (a boundary row): one infinite eigenvalue, the finite one kept."""
    L = np.array([[-2.0, 0.0], [1.0, 1.0]])
    M = np.array([[1.0, 0.0], [0.0, 0.0]])
    # (L + s M) x = 0: det = (s - 2) * 1 -> one finite eigenvalue s = 2
    r = solve_dense(L, M)
    assert r.n_infinite == 1 and r.values.size == 1
    assert abs(r.values[0] - 2.0) < 1e-12
    assert "rank 1 of 2" in r.log[-1]
    assert r.infinite_ambiguous.size == 0          # an exact zero row: proven infinite


def test_f7_mixed_singular_and_large_finite_is_named_ambiguous():
    """M singular (one zero row) and a finite eigenvalue 1e14 relative: the rule cannot tell it
    from a Jordan block at infinity, and the log names it with its modulus."""
    L = -np.diag([1.0, 1e14, 1.0])
    M = np.diag([1.0, 1.0, 0.0])
    r = solve_dense(L, M)
    assert r.values.size + r.n_infinite == 3
    if r.values.size == 1:
        assert "ambiguous" in r.log[-1] and "1.000e+14" in r.log[-1], r.log[-1]
    else:
        assert np.allclose(np.sort(r.values.real), [1.0, 1e14], rtol=1e-10, atol=0)


def test_f7_noise_row_of_m_does_not_turn_an_infinite_eigenvalue_finite():
    """Release 0.1.2: the exact pencil L = -diag(1, 1e13, 1), M = diag(1, 1, 0) has the finite
    eigenvalues 1 and 1e13 (analytic: (s - 1)(s - 1e13) = 0) and one infinite eigenvalue (the zero
    row of M). Rounding noise of 1e-17 ||M|| in that row must not make M look nonsingular: the
    rank of M was taken after dividing each row by its own largest entry, the noise row became
    of order one, and the infinite eigenvalue came back as a finite value near 1e17. The genuine
    eigenvalue 1e13 stays finite."""
    L = -np.diag([1.0, 1e13, 1.0])
    M = np.diag([1.0, 1.0, 0.0])
    M[2] = 1e-17 * np.array([1.0, -2.0, 1.0])
    r = solve_dense(L, M)
    assert r.n_infinite == 1 and r.values.size == 2, (r.values, r.log)
    assert np.allclose(np.sort(r.values.real), [1.0, 1e13], rtol=1e-10, atol=0), r.values
    assert "rank 2 of 3" in r.log[-1] and "ambiguous" in r.log[-1], r.log[-1]


def _f2_model():
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), r=1.0)
    m.scalar("u")
    m.scalar("v")
    m.eigenvalue("s", growth="real")
    m.equations("s*u + u = 0", "s*v - 1e-14*r*v = 0")
    return m


@pytest.mark.parametrize("r", [1.0, 1e10, 1e14])
def test_f2_small_parameter_term_is_kept(r):
    """F2: eigenvalues -1 and 1e-14 r (closed form); the r term was deleted as 'identically zero'."""
    P = _f2_model().parametric(8, over=("r",))
    assert P.n_dropped == 0 and P.n_terms == 2
    got = np.sort(P.solve(r=r).values.real)
    exact = np.sort([-1.0, 1e-14 * r])
    assert np.allclose(got, exact, rtol=1e-10, atol=1e-30), (got, exact)
    assert "identically zero" not in P.explain()


def test_f2_entries_at_rounding_level_refuse_where_they_would_matter():
    """An entry 1 + 1e-15 r: the r part is below ENTRY_DROP_TOL of the entry and discarded; at
    r = 1e14 it would be 0.1 of the pencil, and the evaluation is refused, not answered."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), r=1.0)
    m.scalar("u")
    m.eigenvalue("s", growth="real")
    m.equations("s*u + (1 + 1e-15*r)*u = 0")
    P = m.parametric(8, over=("r",))
    assert abs(P.solve(r=2.0).values[0].real + 1) < 1e-12        # -1 - 2e-15: rounding
    with pytest.raises(ParameterError, match="discarded"):
        P.solve(r=1e14)


def test_f8_hopf_curve_stops_at_the_bogdanov_takens_point():
    """F8: x' = y, y' = b1 + b2 x + x^2 - x y. The Hopf curve is b1 = 0, b2 < 0; at each point
    NumPy's eigenvalues of the Jacobian [[0, 1], [b2 + 2x - y, -x]] are +-i sqrt(-b2)."""
    from conatus.expert import (ContinuationSettings, continue_codim2, continue_equilibria,
                                ir_system, ode_problem)
    bt = ir_system(ode_problem(["x", "y"], {"b1": -0.5, "b2": -1.0},
                               lambda X, P: [X["y"], P["b1"] + P["b2"] * X["x"] + X["x"] ** 2
                                             - X["x"] * X["y"]]), ["b1", "b2"])
    cfg = ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.3, adapt=1.3, fast_iterations=3,
                               tolerance=1e-11, max_newton=12, max_points=40, dp_fd=None)
    eq = continue_equilibria(bt, np.array([(1 - np.sqrt(3)) / 2, 0.0]), "b1", cfg,
                             marginal_tol=1e-9, bounds=(-2, 2))
    h = eq.events_of("hopf")[0]
    curve = continue_codim2(bt, h.point.z[:-1], "hopf", ("b1", "b2"), cfg,
                            par={"b1": h.point.z[-1]}, bounds={"b2": (-1.5, 1.5)})
    b2 = curve.column("b2")
    assert b2.max() < 0 and "Bogdanov" in curve.stop_reason, (b2, curve.stop_reason)
    for p in curve.points:
        x, y = p.z[0], p.z[1]
        lam = np.linalg.eigvals([[0.0, 1.0], [p.z[-1] + 2 * x - y, -x]])
        assert np.abs(lam.real).max() < 1e-9 and np.abs(lam.imag).min() > 0, (p.z, lam)
    (ev,) = curve.events_of("bt")
    assert np.abs(ev.point.z[-2:]).max() < 1e-10
