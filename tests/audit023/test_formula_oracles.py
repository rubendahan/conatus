"""Formulas that surviving mutants showed no test could tell apart (verification map, mutation
sample). Each oracle is independent of the code: a closed form, quadrature, or a numerical solve."""
import math
import warnings

import numpy as np
import pytest
from numpy.polynomial import chebyshev as Ch
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import eval_hermite, gammaln

import conatus as cn
from conatus.bifurcation import ir_system, normal_form, ode_problem
from conatus.discretize.families import HermiteFamily
from conatus.ecs.newton import arnoldi_gmres
from conatus.errors import DefaultChoiceWarning
from conatus.ir import CartesianCoordinates, Hermite
from conatus.solvers.evp import matrix_derivative
from conatus.verify.uncertainty import uncertainty


@pytest.mark.parametrize("c, k", [(4.0, 2.0), (0.25, -1.0), (3.0, -1.0)])
def test_pitchfork_amplitude_law_away_from_unit_a3(c, k):
    """x' = mu x + k x y - c x^3, y' = -y + x^2: on the slaved branch y = x^2 the steady amplitude
    is x^2 = mu / (c - k) at every mu, and a numerical root finder confirms it."""
    pb = ode_problem(["x", "y"], {"mu": 0.0},
                     lambda X, P: [P["mu"] * X["x"] + k * X["x"] * X["y"]
                                   - c * X["x"] * X["x"] * X["x"],
                                   -1.0 * X["y"] + X["x"] * X["x"]], "pitchfork")
    nf = normal_form(ir_system(pb, ["mu"]), np.zeros(2), kind="zero", parameter="mu",
                     symmetry=np.diag([-1.0, 1.0]))
    assert abs(abs(nf.a3) - 1) > 0.2                       # the mutant a3 * ... needs |a3| != 1
    dp = 0.05
    a = nf.amplitude(dp)
    x, y = fsolve(lambda v: [dp * v[0] + k * v[0] * v[1] - c * v[0] ** 3, -v[1] + v[0] ** 2],
                  [math.sqrt(dp / (c - k)), dp / (c - k)], xtol=1e-13)
    assert abs((a * abs(nf.q[0])) - abs(x)) < 1e-10
    assert math.isnan(nf.amplitude(-dp))


@pytest.mark.parametrize("w, a, b", [(1.0, -2.0, 3.0), (2.5, -0.5, -1.0)])
def test_hopf_period_and_radius_with_shear(w, a, b):
    """z' = (mu + i w) z + (a + i b) |z|^2 z in Cartesian form: on the cycle |z|^2 = -mu/a and the
    angular speed is w + b |z|^2, so the period is exact."""
    def rhs(X, P):
        r2 = X["x"] * X["x"] + X["y"] * X["y"]
        return [P["mu"] * X["x"] - w * X["y"] + a * r2 * X["x"] - b * r2 * X["y"],
                w * X["x"] + P["mu"] * X["y"] + a * r2 * X["y"] + b * r2 * X["x"]]
    nf = normal_form(ir_system(ode_problem(["x", "y"], {"mu": 0.0}, rhs, "hopf"), ["mu"]),
                     np.zeros(2), kind="hopf", parameter="mu")
    dp = 0.2
    r2 = -dp / a
    assert nf.period(dp) == pytest.approx(2 * np.pi / (w + b * r2), rel=1e-12)
    orbit = nf.orbit(dp, np.linspace(0, 1, 17))
    assert np.max(np.hypot(orbit[:, 0], orbit[:, 1])) == pytest.approx(math.sqrt(r2), rel=1e-10)


def test_h2_residual_with_a_quadratic_term_in_rotated_coordinates():
    """The reported residual of the second-order solve is the residual of its own equation: with
    a2 != 0 and a kernel away from the axes it has to reach round-off."""
    Rt = np.array([[0.6, -0.8], [0.8, 0.6]])

    def rhs(X, P):
        u = Rt[0, 0] * X["x"] + Rt[1, 0] * X["y"]         # kernel coordinate
        v = Rt[0, 1] * X["x"] + Rt[1, 1] * X["y"]
        du = P["mu"] * u + 1.7 * u * u
        dv = -2.0 * v + u * u
        return [Rt[0, 0] * du + Rt[0, 1] * dv, Rt[1, 0] * du + Rt[1, 1] * dv]
    nf = normal_form(ir_system(ode_problem(["x", "y"], {"mu": 0.0}, rhs, "rot"), ["mu"]),
                     np.zeros(2), kind="zero", parameter="mu")
    assert abs(nf.a2) > 0.5
    assert nf.checks["residual h2"] < 1e-12


def test_hermite_integration_row_against_quadrature():
    """Integral over the line of the orthonormal Hermite functions psi_n, by adaptive quadrature;
    every odd mode integrates to zero. A scale multiplies all of them by one factor."""
    x_ = CartesianCoordinates("x")["x"]
    row = HermiteFamily(Hermite(x_, 16, scale=1.0, dealias=1.0)).integration_row(0)
    for n in range(16):
        log_norm = 0.5 * (n * math.log(2) + gammaln(n + 1) + 0.5 * math.log(math.pi))
        val, _ = quad(lambda t, n=n, c=log_norm: eval_hermite(n, t) * np.exp(-t * t / 2 - c),
                      -30, 30, limit=400, epsabs=1e-11, epsrel=1e-11)
        assert row[n] == pytest.approx(val, abs=1e-8)
    other = HermiteFamily(Hermite(x_, 16, scale=0.5, dealias=1.0)).integration_row(0)
    ratio = other[::2] / row[::2]
    assert np.ptp(ratio) < 1e-12


def test_matrix_derivative_when_M_depends_on_the_parameter():
    """L = p^2 A, M = p^3 B: the derivatives are 2 p A and 3 p^2 B by hand."""
    A, B = np.array([[1.0, 2.0], [0.0, 3.0]]), np.array([[0.5, 0.0], [4.0, -1.0]])
    dL, dM = matrix_derivative(lambda p: (p ** 2 * A, p ** 3 * B), 1.7)
    assert np.allclose(dL, 2 * 1.7 * A, atol=1e-8)
    assert np.allclose(dM, 3 * 1.7 ** 2 * B, atol=1e-8)


@pytest.mark.parametrize("n", [17, 18])
def test_eigenvector_tail_against_exact_chebyshev_coefficients(n):
    """u'' = s u, u(+-1) = 0, mode sin(n pi (z + 1) / 2): the energy fraction in the last 10 % of
    the Chebyshev coefficients, from the exact function by interpolation far beyond N. Odd and even
    n: each has a parity, so half of the coefficients vanish and a window off by one shows on one."""
    N = 48
    m = cn.Model(cn.Box(z=cn.walls(-1, 1)), name="tail")
    m.scalar("u")
    m.eigenvalue("s", growth="real")
    m.equations("dz(dz(u)) = s*u")
    m.at("walls", u=0)
    r = m.eigenvalues(N)
    i = int(np.argmin(np.abs(r.values + (n * np.pi / 2) ** 2)))
    assert abs(r.values[i] + (n * np.pi / 2) ** 2) < 1e-6
    c = Ch.chebinterpolate(lambda z: np.sin(n * np.pi * (z + 1) / 2), 256)[:N]
    k = max(2, math.ceil(0.1 * N))
    exact = np.sum(c[-k:] ** 2) / np.sum(c ** 2)
    assert 1e-30 < exact < 1e-3
    # the discrete mode's tail follows the exact one within a factor well below the decay per index
    assert r.tails[i] == pytest.approx(exact, rel=2.0)


def test_uncertainty_spatial_study_uses_the_finest_dt_only():
    """A separable quantity: the spatial estimate must not depend on the temporal term at all, and
    equals the one of the same quantity with the temporal term removed."""
    full = lambda N, dt: 1.0 + 3.0 / N ** 2 + 0.7 * dt       # noqa: E731
    spatial = lambda N, dt: 1.0 + 3.0 / N ** 2               # noqa: E731
    a = uncertainty(full, N=[16, 32, 64], dt=[0.04, 0.02, 0.01])
    b = uncertainty(spatial, N=[16, 32, 64], dt=[0.04, 0.02, 0.01], scheme_order=1)
    assert a.spatial.estimate == pytest.approx(b.spatial.estimate, rel=1e-9)


def test_gmres_reported_residual_is_the_true_one():
    """The relative residual that GMRES returns against |b - A x| / |b| computed with NumPy."""
    rng = np.random.default_rng(3)
    A = np.eye(60) + 0.4 * rng.standard_normal((60, 60)) / math.sqrt(60)
    b = rng.standard_normal(60)
    V, H, beta, y, relres = arnoldi_gmres(lambda v: A @ v, b, max_iterations=8, rtol=1e-30)
    x = V[:, :-1] @ y
    assert relres == pytest.approx(np.linalg.norm(b - A @ x) / np.linalg.norm(b), rel=1e-8)
    assert relres < 1.0


def test_given_field_named_t_is_stated_as_a_constant():
    """A given field named t in an equation is a constant, not the time: said to the user."""
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="t_given")
    m.given(t=2.0)
    m.scalar("u")
    m.equations("dt(u) - dz(dz(u)) = t")
    m.at("walls", u=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.compile(16)
    assert any(issubclass(x.category, DefaultChoiceWarning) and "the given field t" in str(x.message)
               for x in w)
