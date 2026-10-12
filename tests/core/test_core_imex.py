"""IMEX in C++: observed temporal order on a forced heat equation (Chebyshev tau), P1 factor reuse,
bitwise determinism."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.discretize.ultraspherical import conversion_chain, differentiation
from conatus.ir import CartesianCoordinates, Chebyshev
from conatus.reference import transforms as REF
from conatus.runtime import ImexIntegrator

N = 24
_cs = CartesianCoordinates("x")
_b = Chebyshev(_cs["x"], N, (-1, 1), dealias=1)
_xg = REF.grid(_b)
S = conversion_chain(N, 0, 2)                  # T -> C^(2)
D2 = differentiation(N, 0, 2)


def heat_system():
    """u_t - u_xx + lift taus = F(u, t), u(+-1) = 0. State = [u_T (N), tau1, tau2]."""
    n = N + 2
    M = sp.lil_matrix((n, n))
    L = sp.lil_matrix((n, n))
    M[:N, :N] = S
    L[:N, :N] = -D2
    L[N - 1, N] = 1.0                          # lift(tau1, -1): last C^(2) mode
    L[N - 2, N + 1] = 1.0                      # lift(tau2, -2)
    L[N, :N] = (-1.0) ** np.arange(N)          # u(-1)
    L[N + 1, :N] = 1.0                         # u(+1)
    return M.tocsr(), L.tocsr(), [N, N + 1]


# time factor of the exact solution u = sin(pi x) phi(t), and phi'
PHI = {"cos": (np.cos, lambda t: -np.sin(t)),
       # flat start: phi^(k)(0) = 0 for k = 1..4, so the lower-order startup steps of SBDF3/4
       # (Dedalus convention: SBDF1, SBDF2, ... for the first steps) commit an O(dt^5) error and
       # the global error shows the scheme's own order (phi^(5) = 120 keeps it non-zero)
       "flat": (lambda t: 1.0 + t ** 5, lambda t: 5.0 * t ** 4)}


def exact(t, phi="cos"):
    return np.sin(np.pi * _xg) * PHI[phi][0](t)


def make_F(phi="cos"):
    """Explicit part: -u + g, g manufactured so that u = sin(pi x) phi(t) is exact."""
    f, df = PHI[phi]

    def F(x, t):
        u = REF.backward(x[:N], _b)
        g = np.sin(np.pi * _xg) * (df(t) + (np.pi ** 2 + 1) * f(t))
        out = np.zeros(N + 2)
        out[:N] = S @ REF.forward(-u + g, _b)
        return out
    return F


F = make_F("cos")


def error(scheme, nsteps, T=0.5, phi="cos"):
    M, L, border = heat_system()
    integ = ImexIntegrator(M, L, border, scheme, T / nsteps, F=make_F(phi))
    x0 = np.zeros(N + 2)
    x0[:N] = REF.forward(exact(0.0, phi), _b)
    x, t = integ.run(x0, 0.0, nsteps)
    assert abs(t - T) < 1e-12
    return np.max(abs(REF.backward(x[:N], _b) - exact(T, phi))), integ


@pytest.mark.parametrize("scheme,order,phi", [
    ("SBDF1", 1, "cos"), ("SBDF2", 2, "cos"), ("SBDF3", 3, "flat"), ("SBDF4", 4, "flat"),
    ("RK111", 1, "cos"), ("RK222", 2, "cos"), ("RK443", 3, "cos")])
def test_observed_order(scheme, order, phi):
    """Each scheme at its own order (coverage hole 6: SBDF3/4 used to be accepted at 2).

    SBDF3/4 run on the flat-start solution (see PHI). Observed log2(e40 / e80) on SBDF1 0.992, SBDF2 2.005, SBDF3 2.943, SBDF4 3.962, RK111 0.992, RK222 1.977, RK443 2.920;
    every deficit shrinks when dt is halved (SBDF3 2.972, SBDF4 3.981, RK443 2.959 at 80/160):
    pre-asymptotic, not an order loss. Two-sided band 0.1: an order-2 SBDF3/4 (lower-order
    start dominating) or an accidental superconvergence both fail."""
    e1, _ = error(scheme, 40, phi=phi)
    e2, _ = error(scheme, 80, phi=phi)
    p = np.log2(e1 / e2)
    assert abs(p - order) < 0.1, (scheme, e1, e2, p)


@pytest.mark.parametrize("scheme", ["SBDF3", "SBDF4"])
def test_lower_order_start_caps_sbdf34_at_two_on_a_generic_solution(scheme):
    """Why the flat start above: with u = sin(pi x) cos t the SBDF1 start step (O(dt^2) local
    error) dominates and SBDF3/4 look second order (observed 2.4 / 2.2 at 40/80, then 0.8 / 2.1:
    no clean rate). This is the startup, documented here, not the scheme (tests/ivp/test_orders.py
    reaches 3 / 4 with the RK443 startup)."""
    e1, _ = error(scheme, 40)
    e2, _ = error(scheme, 80)
    assert np.log2(e1 / e2) < 2.6


def test_rk443_single_factorization_and_determinism():
    e, integ = error("RK443", 20)
    assert integ.n_factorizations == 1         # all H_ii = 1/2 (P1)
    e_again, _ = error("RK443", 20)
    assert e == e_again                        # bitwise reproducible
    _, sb = error("SBDF4", 20)
    assert sb.n_factorizations == 4            # startup orders 1..4
    assert "RK443" in integ.explain()


def test_unknown_scheme_refused():
    M, L, border = heat_system()
    with pytest.raises(ValueError):
        ImexIntegrator(M, L, border, "RK999", 0.1)
