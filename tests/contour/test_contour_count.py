"""Eigenvalue counting by the argument principle (conatus.contour.count).

Counts are checked against exact roots and dense spectra; every refusal (an eigenvalue on the
boundary, a non-integer winding, a failed control box) is checked to raise rather than report.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

import conatus as cn
from conatus.contour import count
from conatus.contour.operators import permutation_sign
from conatus.errors import ConvergenceError, InputTypeError, OptionError


def _inside(box, values) -> int:
    return int(np.count_nonzero(box.contains(np.asarray(values))))


def test_box_geometry_and_refusals():
    b = cn.box(re=(-1.0, 2.0), im=(0.0, 1.0))
    assert repr(b) == "cn.box(re=(-1.0, 2.0), im=(0.0, 1.0))"
    assert b.center == complex(0.5, 0.5) and b.perimeter == 8.0
    assert np.allclose(b.point([0, 1, 2, 3, 4]), [-1, 2, 2 + 1j, -1 + 1j, -1], rtol=0, atol=1e-15)
    assert b.contains(0.5 + 0.5j) and not b.contains(3.0)
    assert "counter-clockwise" in b.explain()
    for bad in [dict(re=(1.0, 0.0), im=(0.0, 1.0)), dict(re=(0.0, np.inf), im=(0.0, 1.0)),
                dict(re=0.5, im=(0.0, 1.0)), dict(re=(0.0, 1.0), im=(2.0, 2.0))]:
        with pytest.raises(OptionError):
            cn.box(**bad)
    with pytest.raises(TypeError):
        cn.box((0, 1), (0, 1))                    # keyword-only: no silent re/im swap


def test_polynomial_counts_with_multiplicity():
    roots = np.array([0.1 + 0.2j, -0.3 + 0.1j, 0.25 - 0.4j, 0.7 + 0.7j, 0.1 + 0.2j])  # one double

    def f(z):
        return np.prod(z - roots)
    for b in [cn.box(re=(-1, 1), im=(-1, 1)), cn.box(re=(0.0, 0.5), im=(0.0, 0.5)),
              cn.box(re=(-0.5, 0.5), im=(-0.5, 0.5)), cn.box(re=(2, 3), im=(2, 3))]:
        r = count(f, b, poles=0)
        assert r.count == r.winding == _inside(b, roots)
        assert r.winding_error < 1e-10
    r = count(f, cn.box(re=(0.0, 0.5), im=(0.0, 0.5)), poles=0)
    assert r.count == 2                           # the double root counts twice
    text = r.explain()
    assert "Z - P" in text and "Z = 2" in text and "P = 0" in text


def test_pencil_dense_and_sparse_against_eigenvalues():
    rng = np.random.default_rng(3)
    n = 40
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    M = np.eye(n)
    lam = np.linalg.eigvals(-A)                   # (A + s I) x = 0  <=>  s eigenvalue of -A
    for b in [cn.box(re=(-3, 3), im=(-3, 3)), cn.box(re=(0, 4), im=(-2, 5)),
              cn.box(re=(-8, -1), im=(-1, 8))]:
        exact = _inside(b, lam)
        assert np.min(b.distance_to_boundary(lam[b.contains(lam)]), initial=1) > 1e-3
        assert count((A, M), b).count == exact
        assert count((sp.csr_matrix(A), sp.identity(n, format="csr")), b).count == exact


def test_singular_mass_pencil_counts_only_finite_eigenvalues():
    # a tau-like pencil: M singular on two rows (infinite eigenvalues are never counted)
    rng = np.random.default_rng(7)
    n = 12
    L = rng.standard_normal((n, n))
    M = np.diag(np.r_[np.ones(n - 2), 0.0, 0.0])
    import scipy.linalg as sla
    w = sla.eigvals(-L, M)
    w = w[np.isfinite(w)]
    b = cn.box(re=(-2.2, 2.2), im=(-2.1, 2.3))
    assert count((L, M), b).count == _inside(b, w)


def test_no_overflow_of_a_huge_determinant():
    # n = 4000, |det| ~ 10^(3 n): det itself is far beyond FP64, its phase and log modulus are not
    n = 4000
    s = np.linspace(-5, 5, n) + 1j * np.linspace(-1, 1, n) ** 3
    L = sp.diags(-s * 1e3)                       # eigenvalues of L + z (1e3 I) are s
    M = sp.identity(n) * 1e3
    h = 10 / (n - 1)                             # edges half-way between eigenvalues: 10 inside
    b = cn.box(re=(-5 + 2000.5 * h, -5 + 2010.5 * h), im=(-0.25, 0.25))
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.prod((L + 0.3 * M).diagonal()))
    r = count((L, M), b)
    assert r.count == _inside(b, s) == 10


def test_poles_are_negative_zeros():
    a, bz, c = 0.2 + 0.1j, -0.3 - 0.2j, 0.05 + 0.3j

    def f(z):
        return (z - a) * (z - bz) / (z - c)
    b = cn.box(re=(-1, 1), im=(-1, 1))
    r = count(f, b)                               # poles not declared: the winding only
    assert r.winding == 1 and r.count is None and r.poles is None
    assert "not declared" in r.explain() and "Z - P = 1" in r.explain()
    assert count(f, b, poles=1).count == 2
    # a box around the pole alone: winding -1
    assert count(f, cn.box(re=(0.0, 0.1), im=(0.25, 0.35))).winding == -1


def test_control_box():
    roots = np.array([0.5j, -0.5j, 2.0])

    def f(z):
        return np.prod(z - roots)
    pair = cn.box(re=(-0.2, 0.2), im=(-1.0, 1.0))
    r = count(f, cn.box(re=(-1, 3), im=(-1, 1)), poles=0, control=(pair, 2))
    assert r.count == 3 and "as expected" in r.control and "control" in r.explain()
    with pytest.raises(ConvergenceError, match="expected 1"):
        count(f, cn.box(re=(-1, 3), im=(-1, 1)), poles=0, control=(pair, 1))


def test_eigenvalue_on_the_boundary_is_refused():
    L, M = np.diag([-0.5, -2.0, 0.3]), np.eye(3)  # eigenvalues 0.5, 2, -0.3
    with pytest.raises(ConvergenceError, match="refused"):
        count((L, M), cn.box(re=(0.0, 1.0), im=(0.0, 1.0)))    # 0.5 on the bottom edge
    with pytest.raises(ConvergenceError, match="exactly zero"):
        count((L, M), cn.box(re=(0.5, 1.0), im=(0.0, 1.0)))    # 0.5 at a corner (a sample point)
    # just inside: counted
    assert count((L, M), cn.box(re=(0.0, 1.0), im=(-1e-6, 1.0))).count == 1


def test_non_integer_winding_is_refused():
    start = complex(-1, -1)
    seen = {"n": 0}

    def f(z):                                     # the start corner returns another phase the 2nd time
        v = z - 0.1
        if z == start:
            seen["n"] += 1
            if seen["n"] > 1:
                return v * np.exp(1j * math.pi / 5)
        return v
    with pytest.raises(ConvergenceError, match="not an integer"):
        count(f, cn.box(re=(-1, 1), im=(-1, 1)), poles=0)


def test_options_refused():
    b = cn.box(re=(0, 1), im=(0, 1))
    with pytest.raises(OptionError):
        count((np.eye(2), np.eye(2)), (0, 1))
    with pytest.raises(OptionError):
        count((np.eye(2), np.eye(2)), b, phase_step=4.0)
    with pytest.raises(OptionError):
        count((np.eye(2), np.eye(2)), b, initial_points=1)
    with pytest.raises(OptionError):
        count(lambda z: z, b, poles=-1)
    with pytest.raises(OptionError):
        count((np.eye(2), np.eye(2)), b, poles=2)
    with pytest.raises(InputTypeError):
        count("not an operator", b)
    with pytest.raises(InputTypeError):
        count(lambda z: np.ones((2, 3)), b, poles=0)
    with pytest.raises(OptionError):
        count((np.eye(2), np.eye(3)), b)


def test_budget_refusal_names_the_place():
    with pytest.raises(ConvergenceError, match="evaluations"):
        count(lambda z: np.prod(z - np.linspace(-0.9, 0.9, 40) - 0.95j), cn.box(re=(-1, 1), im=(-1, 1)),
              poles=0, max_evaluations=70)


def test_permutation_sign():
    assert permutation_sign(np.arange(5)) == 1
    assert permutation_sign(np.array([1, 0, 2])) == -1
    assert permutation_sign(np.array([1, 2, 0])) == 1
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.permutation(7)
        assert permutation_sign(p) == round(np.linalg.det(np.eye(7)[p]))


def test_hill_count_through_a_block_thomas_breakdown():
    """A regular Hill matrix whose unpivoted block elimination breaks down at mu = 1j (det = -2j,
    cond 5): the count used to be refused as "det T is exactly zero at z = 1j on the boundary (an
    eigenvalue on the contour)" (critic review of L10). The operator falls back to the sparse LU
    of the assembled matrix there; the count is the dense one."""
    from conatus.contour import beyn
    from conatus.floquet.hill_general import hill_from_blocks
    h = hill_from_blocks(np.eye(1), np.zeros((1, 1)), {1: np.ones((1, 1)), -1: np.ones((1, 1))},
                         Nh=1, omega=1.0, label="breakdown", note="")
    A0, A1 = h.dense(0.0), h.dense(1.0) - h.dense(0.0)
    exact = np.linalg.eigvals(-np.linalg.solve(A1, A0))           # M(mu) = A0 + mu A1
    box = cn.box(re=(-1.5, 1.5), im=(-0.5, 1.0))                     # top edge through mu = 1j
    r = count(h, box, initial_points=17)                             # a node exactly at 1j
    assert r.count == int(np.sum(box.contains(exact))) == 3
    b = beyn(h, box, counted=r)
    # complex values against the dense oracle, not only the real parts
    inside = np.sort_complex(exact[box.contains(exact)])
    assert b.values.size == inside.size == 3
    assert np.allclose(np.sort_complex(b.values), inside, rtol=0, atol=1e-10), (b.values, inside)
    assert np.allclose(np.sort(b.values.real), [-1.0, 0.0, 1.0], rtol=0, atol=1e-10)


@pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-9, 1e-12])
@pytest.mark.parametrize("x0", [0.53125, 1 / 3])
def test_zeros_hugging_an_edge_are_counted_or_refused(eps, x0):
    """Clusters of zeros at distance eps from the bottom edge, between the initial points of the
    walk (two inside, one inside and its mirror outside, five inside 1e-4 apart): the count is
    exact, or refused when the zeros are within rounding of the edge; never another integer
    (critic review of L10: the principal-increment bound is an acceptance test, not a proof)."""
    box = cn.box(re=(0, 1), im=(0, 1))
    cases = [([x0 + 1j * eps, x0 + 1e-3 + 1j * eps, 2 + 2j], 2),
             ([x0 + 1j * eps, x0 - 1j * eps, 2 + 2j], 1),
             ([x0 + k * 1e-4 + 1j * eps for k in range(5)] + [3j], 5)]
    for lams, expected in cases:
        L = np.diag(np.asarray(lams, complex))
        try:
            got = count((L, -np.eye(len(lams))), box).count
        except ConvergenceError:
            assert eps <= 1e-9, (lams, "refused although clear of the edge")
            continue
        assert got == expected, (lams, got)
