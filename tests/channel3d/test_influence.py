"""Pressure treatment (T4): Kleiser–Schumann influence matrix vs the plan's generalized
tau, on a manufactured Stokes-resolvent mode. Both converge spectrally to the exact solution;
the influence solver leaves an O(tau) interior divergence (no tau correction), reported."""
import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

from stokes_mode import cheb_coeffs, exact
from conatus.timestep.influence import KleiserSchumannStokes

KX, KZ, CC, RE = 1.0, 2.0, 50.0, 100.0


def _errors(N):
    (u, v, w), p, r = exact(KX, KZ, CC, RE)
    rc = np.array([cheb_coeffs(f, N) for f in r])
    ks = KleiserSchumannStokes(N, RE, CC)
    U, P = ks.solve(KX, KZ, rc)
    yy = np.linspace(-1, 1, 201)
    err_u = max(np.max(np.abs(C.chebval(yy, U[i]) - f(yy))) for i, f in enumerate((u, v, w)))
    err_p = np.max(np.abs(C.chebval(yy, P) - p(yy)))
    div = ks.divergence(KX, KZ, U)
    wall = np.abs(np.array([C.chebval(t, U[i]) for i in range(3) for t in (-1, 1)])).max()
    return err_u, err_p, np.max(np.abs(div)), wall


def test_influence_matrix_solution_converges_spectrally():
    e = {N: _errors(N) for N in (16, 24, 32)}
    assert e[32][0] < 1e-12 and e[32][1] < 1e-11          # velocity and pressure vs exact
    assert e[16][0] > 10 * e[24][0] > 10 * e[32][0] or e[24][0] < 1e-12
    assert e[32][3] < 1e-13                                 # no-slip to roundoff


def test_influence_matrix_divergence_is_small_but_reported():
    """No tau correction: the interior divergence is the tau residual, small at convergence."""
    d16, d32 = _errors(16)[2], _errors(32)[2]
    assert d32 < 1e-10 and d32 < d16


def test_mean_mode_refused():
    ks = KleiserSchumannStokes(16, RE, CC)
    with pytest.raises(ValueError, match="mode \\(0, 0\\)"):
        ks.solve(0.0, 0.0, np.zeros((3, 16), complex))
    assert "no tau-error correction" in ks.explain()


def test_tau_and_influence_agree_with_the_exact_mode():
    """Same manufactured mode through the plan's generalized tau (LBVP): both methods reach the
    exact solution; at N = 16 tau is slightly more accurate (measured 6.8e-11 vs 5.1e-10)."""
    from conatus.reference import transforms as T
    from stokes_mode import tau_resolvent
    (u, v, w), p, _ = exact(KX, KZ, CC, RE)
    errs = {}
    for N in (16, 24):
        plan, _, sol = tau_resolvent(N, KX, KZ, CC, RE)
        g = [T.grid(b, dealias=False) for b in plan.domain.bases]
        X, Y, Z = np.meshgrid(*g, indexing="ij")
        e = np.exp(1j * (KX * X + KZ * Z))
        U = sol.grid("u")
        errs[N] = max(np.abs(U[i] - f(Y) * e).max() for i, f in enumerate((u, v, w)))
        assert sol.max_residual < 1e-13
    assert errs[24] < 1e-13
    assert errs[16] < _errors(16)[0]            # tau at least as accurate as the influence solve
