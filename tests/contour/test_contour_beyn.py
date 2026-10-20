"""Beyn's contour-integral method, linear and nonlinear, certified (conatus.contour.beyn)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg as sla

import conatus as cn
from conatus.contour import beyn, count
from conatus.errors import ConvergenceError, OptionError


def _match(found, exact, tol):
    """Every exact value has one found value within tol, and conversely."""
    found, exact = np.sort_complex(np.asarray(found)), np.sort_complex(np.asarray(exact))
    assert found.size == exact.size, (found, exact)
    for z in exact:
        assert np.min(np.abs(found - z)) < tol, (z, found)


def test_linear_pencil_known_eigenvalues():
    rng = np.random.default_rng(11)
    lam = np.array([0.3 + 0.2j, -0.4 + 0.1j, 0.1 - 0.35j, 1.5 + 1.5j, -2.0, 0.05 + 0.05j])
    n = 30
    lam_all = np.r_[lam, 3 + rng.standard_normal(n - lam.size)]
    S = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = S @ np.diag(-lam_all) @ np.linalg.inv(S)          # (A + s I) x = 0 at s = lam
    b = cn.box(re=(-0.5, 0.5), im=(-0.5, 0.5))
    r = beyn((A, np.eye(n)), b)
    _match(r.values, lam[b.contains(lam)], 1e-11)
    assert r.count == 4 and (r.multiplicity == 1).all() and (r.residuals < 1e-13).all()
    for i in range(r.values.size):               # the vectors are eigenvectors
        x = r.vectors[:, i]
        assert np.linalg.norm(A @ x + r.values[i] * x) < 1e-11
    text = r.explain()
    assert "residual" in text and "count Z = 4" in text and "Hankel" in text
    assert "argument principle" in r.count_method


def _diag_problem():
    """T(z) = Q diag(e^z - 2, z^2 + 1, cosh z) Q^-1 and T'(z): known roots."""
    rng = np.random.default_rng(5)
    Q = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    Qi = np.linalg.inv(Q)

    def T(z):
        return Q @ np.diag([np.exp(z) - 2, z * z + 1, np.cosh(z)]) @ Qi

    def dT(z):
        return Q @ np.diag([np.exp(z), 2 * z, np.sinh(z)]) @ Qi
    return T, dT


def _exact_diag_roots(b):
    ks = range(-3, 4)
    roots = [math.log(2) + 2j * math.pi * k for k in ks] + [1j, -1j] \
        + [1j * (math.pi / 2 + math.pi * k) for k in ks]
    roots = np.array(roots)
    return roots[b.contains(roots)]


@pytest.mark.parametrize("with_derivative", [True, False])
def test_nonlinear_known_roots(with_derivative):
    T, dT = _diag_problem()
    b = cn.box(re=(-1.0, 1.0), im=(-2.0, 2.0))
    exact = _exact_diag_roots(b)
    assert exact.size == 5
    r = beyn(T, b, poles=0, derivative=dT if with_derivative else None)
    _match(r.values, exact, 1e-10)
    assert (r.residuals <= 1e-10).all() and r.count == 5
    # a box straddling more roots, cosh and exp branches together
    b2 = cn.box(re=(0.2, 0.9), im=(-7.0, 7.1))
    r2 = beyn(T, b2, poles=0, derivative=dT if with_derivative else None)
    _match(r2.values, _exact_diag_roots(b2), 1e-10)


def test_scalar_transcendental_needs_moments():
    # n = 1: one probe column, the eigenvalues come from the block Hankel moments
    b = cn.box(re=(-4.0, 7.0), im=(-1.0, 1.0))
    r = beyn(np.sin, b, poles=0, derivative=np.cos)
    _match(r.values, [-math.pi, 0.0, math.pi, 2 * math.pi], 1e-12)
    assert r.columns == 1 and r.moments >= 4


def test_quadratic_eigenproblem_against_linearisation():
    rng = np.random.default_rng(2)
    n = 8
    K, C, Mq = (rng.standard_normal((n, n)) for _ in range(3))
    Z, I = np.zeros((n, n)), np.eye(n)
    w = sla.eigvals(np.block([[Z, I], [-K, -C]]), np.block([[I, Z], [Z, Mq]]))
    b = cn.box(re=(-1.1, 0.9), im=(-1.2, 1.3))
    exact = w[b.contains(w)]
    assert exact.size >= 3
    r = beyn(lambda z: K + z * C + z * z * Mq, b, poles=0, derivative=lambda z: C + 2 * z * Mq)
    _match(r.values, exact, 1e-9)


def test_hill_mathieu_pair_with_control():
    from conatus.expert import HillBlocks, exponents_dense
    a, q = 1.5, 0.5
    L0 = np.array([[0, -1], [a, 0]], complex)
    Cq = np.array([[0, 0], [-q, 0]], complex)
    H = HillBlocks(diag=lambda mu, n: L0 + (mu + 2j * n) * np.eye(2), couplings={1: Cq, -1: Cq},
                   Nh=24, size=2, label="Mathieu", omega=2.0, mass=np.eye(2))
    mus = exponents_dense(H, finite_cap=1e3)
    pair = cn.box(re=(-0.5, 0.5), im=(0.25, 1.75))           # the neutral pair i nu, i (2 - nu)
    assert np.count_nonzero(pair.contains(mus)) == 2
    c = count(H, pair)
    assert c.count == 2 and c.poles == 0
    r = beyn(H, pair, control=(cn.box(re=(-0.5, 0.5), im=(-0.9, 0.9)), 0))
    _match(r.values, mus[pair.contains(mus)], 1e-12)
    assert np.abs(r.values.real).max() < 1e-12                 # neutral: on the imaginary axis
    # the pair on an edge of the box: refused, not counted
    with pytest.raises(ConvergenceError):
        count(H, cn.box(re=(0.0, 0.5), im=(0.25, 1.75)))


def test_zero_count_and_refusals():
    b = cn.box(re=(5, 6), im=(5, 6))
    r = beyn((np.diag([1.0, 2.0]), np.eye(2)), b)
    assert r.count == 0 and len(r) == 0 and "no eigenvalue" in r.explain()
    with pytest.raises(OptionError, match="poles"):
        beyn(lambda z: z - 0.1, cn.box(re=(-1, 1), im=(-1, 1)))
    with pytest.raises(OptionError):
        beyn((np.eye(2), np.eye(2)), b, residual_tol=0.0)
    with pytest.raises(OptionError):
        beyn((np.eye(2), np.eye(2)), b, quadrature=0)
    other = count((np.eye(2), np.eye(2)), cn.box(re=(0, 1), im=(0, 1)))
    with pytest.raises(OptionError):
        beyn((np.eye(2), np.eye(2)), b, counted=other)


def test_wrong_pole_declaration_is_caught():
    a, bz, c = 0.2 + 0.1j, -0.3 - 0.2j, 0.05 + 0.3j

    def f(z):
        return (z - a) * (z - bz) / (z - c)
    b = cn.box(re=(-1, 1), im=(-1, 1))
    r = beyn(f, b, poles=1)
    _match(r.values, [a, bz], 1e-12)
    with pytest.raises(ConvergenceError, match="do not account"):
        beyn(f, b, poles=0)                       # count 1, two certified roots: refused


def test_reproducible_with_seed():
    T, dT = _diag_problem()
    b = cn.box(re=(-1.0, 1.0), im=(-2.0, 2.0))
    r1, r2 = beyn(T, b, poles=0, derivative=dT, seed=4), beyn(T, b, poles=0, derivative=dT, seed=4)
    assert np.array_equal(r1.values, r2.values)


def test_callable_box_is_never_cut():
    # one node per edge cannot resolve 20 roots; a declared P = 0 never allows a cut:
    # refused, where a pencil would be cut (tests/contour/test_contour_region_evp.py)
    lam = np.exp(2j * np.pi * np.arange(20) / 20) * 0.6 + 0.05
    b = cn.box(re=(-0.9, 0.9), im=(-0.85, 0.95))
    with pytest.raises(ConvergenceError, match="refused"):
        beyn(lambda z: np.prod(z - lam), b, poles=0, quadrature=1, max_rounds=1)


@pytest.mark.parametrize("m", [3, 6, 12])
def test_defective_eigenvalue_is_one_root_with_its_multiplicity(m):
    """A Jordan block of order m: rounding splits its eigenvalue into m values on a circle of
    radius ~ eps^(1/m), each with a backward residual of rounding size. Every one used to be
    returned as a "certified" simple root (order 12: twelve roots 0.04 away from the eigenvalue),
    because only groups closer than 1e-6 rho were counted locally (critic review of L10). Each
    returned root now carries its own argument-principle count: one root, multiplicity m, at the
    eigenvalue to rounding."""
    lam0 = 0.5 + 0.5j
    L = lam0 * np.eye(m) + np.diag(np.ones(m - 1), 1)
    box = cn.box(re=(0, 1), im=(0, 1))
    b = beyn((L, -np.eye(m)), box)
    assert b.count == m
    assert b.values.size == 1 and int(b.multiplicity[0]) == m, (b.values, b.multiplicity)
    assert abs(b.values[0] - lam0) < 1e-8
    assert b.residuals[0] <= b.residual_tol


def test_defective_pair_next_to_a_simple_eigenvalue():
    L = np.diag([0.5 + 0.5j, 0.5 + 0.5j, 0.2 + 0.8j, 0.7 + 0.3j]).astype(complex)
    L[0, 1] = 1.0
    b = beyn((L, -np.eye(4)), cn.box(re=(0, 1), im=(0, 1)))
    got = sorted(zip(np.round(b.values, 8), b.multiplicity), key=lambda t: (t[0].real, t[0].imag))
    assert [int(k) for _, k in got] == [1, 2, 1]
    _match(b.values, [0.2 + 0.8j, 0.5 + 0.5j, 0.7 + 0.3j], 1e-8)
