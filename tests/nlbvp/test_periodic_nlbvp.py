"""Periodic NLBVP (all-Fourier domain): Newton with a dense coupled Jacobian and a declared phase
condition (gallery gap: the banded NewtonSolver refuses an x-dependent iterate,
"non-constant coefficient 'U_u' varies along a Fourier direction").

Oracle: Swift–Hohenberg rolls r u - (1 + d_xx)^2 u - u^3 = 0 at r = 0.1; the odd-subspace
Chebyshev solve of examples/patterns/swift_hohenberg.py gives A = 0.365243301464940 (N = 24,
0.365243301464942 at N = 32): an independent discretization of the same periodic solution.
"""
import numpy as np
import pytest

from conatus.discretize.compiler import DiscretizationError
from conatus.ir import (NLBVP, CartesianCoordinates, ComplexFourier, Domain, Field, Parameter,
                        ProblemError, diff, integ)
from conatus.reference import transforms as T
from conatus.solvers import NewtonSolver, PeriodicNewtonSolver, PhaseCondition

A_ODD = 0.365243301464940        # roll_solver(24, 0.1) of the gallery example (odd subspace)


def sh_rolls(N, basis=None, r=0.1, dealias=2):
    """Swift–Hohenberg steady rolls on the 2 pi-periodic line."""
    from conatus.ir import RealFourier
    cs = CartesianCoordinates("x")
    x = cs["x"]
    B = (basis or RealFourier)(x, N, (0, 2 * np.pi), dealias=dealias)
    u, rr = Field("u", Domain(cs, [B])), Parameter("r", r)

    def opl(f):
        return f + diff(diff(f, x), x)
    pb = NLBVP([u], name="Swift-Hohenberg rolls (periodic)")
    pb.add_equation((rr * u - opl(opl(u)) - u * u * u, 0))
    return pb, u, x


def _solver(pb, phase, **kw):
    return PeriodicNewtonSolver(pb, tolerance=1e-13, damping=1.0, max_iterations=20, phase=phase, **kw)


def test_banded_newton_refuses_the_periodic_rolls():
    """The gap itself: the per-wavenumber Newton step cannot hold an x-dependent Jacobian."""
    pb, u, x = sh_rolls(16)
    s = NewtonSolver(pb, tolerance=1e-12, damping=1.0, max_iterations=5)
    with pytest.raises(DiscretizationError, match="varies along a Fourier direction"):
        s.solve({u: lambda X: 0.3 * np.cos(X)})


@pytest.mark.parametrize("N", [12, 16, 24])
def test_rolls_match_odd_subspace_and_converge_in_N(N):
    pb, u, x = sh_rolls(N)
    s = _solver(pb, PhaseCondition(u, x, kind="sin1"))
    sol = s.solve({u: lambda X: np.sqrt(0.4 / 3) * np.cos(X)})
    assert sol.converged and sol.iterations[-1].rel_residual < 1e-13
    assert abs(sol.harmonic("u", x, 1) - A_ODD) < 1e-12, sol.harmonic("u", x, 1)
    assert abs(sol.multipliers["mu_x"]) < 1e-14                       # steady: no drift
    c = sol.coeffs["u"]
    assert np.abs(c[3::2]).max() < 1e-14                               # pure cosines (phase)
    assert np.abs(c[4::4]).max() < 1e-14                               # even harmonics absent
    assert len(sol.iterations) <= 5                                    # quadratic convergence
    assert "dense coupled matrix" in s.explain() and "phase condition along x" in s.explain()


def test_complex_fourier_with_reference_phase():
    """Same rolls on ComplexFourier, phase fixed by orthogonality to d_x of the initial guess;
    the guess is shifted by 0.3, so the solution is the roll shifted to where <u, d_x u0> = 0."""
    pb, u, x = sh_rolls(16, basis=ComplexFourier)
    s = _solver(pb, PhaseCondition(u, x, kind="reference"))
    sol = s.solve({u: lambda X: np.sqrt(0.4 / 3) * np.cos(X - 0.3)})
    assert sol.converged and abs(sol.harmonic("u", x, 1) - A_ODD) < 1e-12
    assert abs(sol.multipliers["mu_x"]) < 1e-14
    assert np.abs(sol.grid("u").imag).max() < 1e-13                   # real solution


def test_no_phase_condition_is_refused_with_the_fix():
    pb, u, x = sh_rolls(16)
    with pytest.raises(ProblemError, match="PhaseCondition"):
        _solver(pb, None)


def test_x_dependent_forcing_needs_no_phase_and_scalar_unknown():
    """u'' - u - u^3 + A = f(x) with f from u* = 0.5 cos x + 0.2 sin 2x and <u> = 0 fixing A = 0.3:
    no translation invariance (f depends on x), a scalar unknown A, exact recovery."""
    from conatus.ir import RealFourier
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 16, (0, 2 * np.pi), dealias=2)])
    u, f, A = Field("u", D), Field("f", D), Field("A", D.without(x))
    pb = NLBVP([u, A])
    pb.add_equation((diff(diff(u, x), x) - u - u * u * u + A, f))
    pb.add_equation((integ(u), 0))

    def us(X):
        return 0.5 * np.cos(X) + 0.2 * np.sin(2 * X)

    def fs(X):
        return -0.5 * np.cos(X) - 0.8 * np.sin(2 * X) - us(X) - us(X) ** 3 + 0.3
    s = PeriodicNewtonSolver(pb, tolerance=1e-13, damping=1.0, max_iterations=20, ncc={f: fs})
    assert "translation-invariant directions: none" in s.explain()
    sol = s.solve({u: lambda X: 0 * X})
    X = T.grid(D.bases[0], dealias=False)
    assert np.abs(sol.grid("u") - us(X)).max() < 1e-13
    assert abs(sol.value("A") - 0.3) < 1e-13 and sol.multipliers == {}


def test_travelling_state_is_refused_as_steady():
    """With an advection term the rolls drift: u_t = F(u) - c u_x has steady solutions only in
    the co-moving frame; declared multiplier catches it (allow_drift=True accepts it)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    from conatus.ir import RealFourier
    D = Domain(cs, [RealFourier(x, 16, (0, 2 * np.pi), dealias=2)])
    u = Field("u", D)

    def opl(v):
        return v + diff(diff(v, x), x)
    # dispersive term d_xxx u breaks x -> -x: the rolls travel (phase speed from the dispersion)
    pb = NLBVP([u])
    pb.add_equation((0.1 * u - opl(opl(u)) - u * u * u + 0.05 * diff(diff(diff(u, x), x), x), 0))
    with pytest.raises(ProblemError, match="travelling"):
        _solver(pb, PhaseCondition(u, x)).solve({u: lambda X: 0.36 * np.cos(X)})
    sol = _solver(pb, PhaseCondition(u, x), allow_drift=True).solve({u: lambda X: 0.36 * np.cos(X)})
    # single-harmonic estimate: 0.05 d_xxx cos x = 0.05 sin x balanced by mu d_x cos x -> mu = 0.05;
    # the third harmonic shifts it by O(A^2 * 1e-4) (observed 0.0500009742)
    assert sol.converged and 1e-8 < abs(sol.multipliers["mu_x"] - 0.05) < 1e-5
