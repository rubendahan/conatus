"""External review of 0.1.1, contour findings F1, F3, F9, F18.

Each test states the correct mathematical answer from an independent oracle (the analytic roots
of a rational or polynomial function, NumPy's eigenvalues, a residual computed here) and checks
that the contour methods return it or say what they did not establish.
"""
from __future__ import annotations

import numpy as np
import pytest

import conatus as cn
from conatus.contour import beyn, count
from conatus.errors import ConvergenceError, OptionError

BOX = cn.box(re=(-1, 1), im=(-1, 1))


@pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-8])
def test_f1_dipole_across_the_edge_is_not_called_certified(eps):
    """F1: T(z) = (z - a) / (z - p), a inside and p outside, both eps from the top edge. The
    analytic count is 1. The sampled walk cannot see the pair (|T| = 1 on that edge); the count
    must be 1, or the result must say that it is a sampled test whose scope excludes this case."""
    a, p = 0.123 + (1 - eps) * 1j, 0.123 + (1 + eps) * 1j
    exact = 1                                               # one zero inside, the pole outside

    def T(z):
        return (z - a) / (z - p)

    c = count(T, BOX, poles=0)
    try:
        r = beyn(T, BOX, poles=0)
    except ConvergenceError:          # a root 1e-8 from the edge, next to a pole: refused
        r = None
    if c.count == exact:
        if r is not None:
            assert r.values.size == 1 and abs(r.values[0] - a) < 1e-9, r.values
        return
    assert r is not None
    assert not c.poles_by_construction
    for text in (c.count_method, c.explain(), r.count_method, r.explain(), repr(r)):
        assert "certif" not in text.lower(), text
    assert "not a proof" in c.count_method and "pole" in c.scope
    assert "not a proof" in r.count_method


def test_f1_pencil_count_states_its_polynomial_scope():
    c = count((-np.diag([0.2, 0.5]), np.eye(2)), BOX)
    assert c.count == 2 and c.poles_by_construction
    assert "polynomial" in c.scope and "certif" not in c.explain().lower()


@pytest.mark.parametrize("delta", [1e-3, 1e-5, 1e-7, 1e-9])
def test_f3_close_distinct_pencil_eigenvalues_are_two_roots(delta):
    """F3: diag pencil with eigenvalues 0.2 and 0.2 + delta (NumPy oracle): two roots, each of
    multiplicity 1, each at its eigenvalue; at 1e-7 Beyn returned one value of multiplicity 2.
    At 1e-9 the moments may not resolve the pair: one value is then allowed only as a cluster
    whose square holds both eigenvalues."""
    L = -np.diag([0.2, 0.2 + delta])
    exact = np.sort(np.linalg.eigvals(-L))
    r = beyn((L, np.eye(2)), BOX)
    if delta < 1e-8 and r.values.size == 1:
        assert int(r.multiplicity[0]) == 2
        assert (np.abs(exact - r.values[0]) <= r.enclosure[0]).all(), (r.values, r.enclosure)
        return
    assert r.values.size == 2 and (r.multiplicity == 1).all(), (r.values, r.multiplicity)
    got = np.sort_complex(r.values)
    assert np.abs(got - exact).max() <= 1e-3 * delta, (got, exact)
    assert r.enclosure.size == 2 and (r.enclosure < delta).all()


def test_f3_close_distinct_polynomial_roots_are_two_roots():
    """F3: (z - 0.2)(z - 0.2000001), the review's scalar case: both roots, not their mean."""
    a, c = 0.2, 0.2000001
    r = beyn(lambda z: (z - a) * (z - c), BOX, poles=0, derivative=lambda z: 2 * z - a - c)
    assert r.values.size == 2 and (r.multiplicity == 1).all(), (r.values, r.multiplicity)
    got = np.sort(r.values.real)
    assert abs(got[0] - a) < 1e-9 and abs(got[1] - c) < 1e-9, got


def test_f3_double_eigenvalue_stays_one_value_with_two_in_its_square():
    """A true double eigenvalue (semisimple) is not split: one value, 2 counted in its square."""
    r = beyn((-np.diag([0.2, 0.2]), np.eye(2)), BOX)
    assert r.values.size == 1 and int(r.multiplicity[0]) == 2
    assert abs(r.values[0] - 0.2) < 1e-12 and r.enclosure[0] > 0


def test_f9_count_of_another_pencil_is_refused():
    """F9: the count of L = [[2]] (root -2, outside: 0) passed for L = [[0]] (root 0 inside)."""
    other = count((np.array([[2.0]]), np.eye(1)), BOX)
    assert other.count == 0
    with pytest.raises(OptionError, match="another operator"):
        beyn((np.array([[0.0]]), np.eye(1)), BOX, counted=other)
    same = count((np.array([[0.0]]), np.eye(1)), BOX)
    r = beyn((np.array([[0.0]]), np.eye(1)), BOX, counted=same)
    assert r.count == 1 and abs(r.values[0]) < 1e-14          # root 0, as numpy says


def test_f9_count_of_another_parameter_value_is_refused():
    """A callable whose parameter changed after the count (root 0.3 moved to 2.3, outside)."""
    state = {"p": 0.3}

    def f(z):
        return z - state["p"]

    c = count(f, BOX, poles=0)
    assert c.count == 1
    state["p"] = 2.3
    with pytest.raises(OptionError, match="another operator"):
        beyn(f, BOX, poles=0, counted=c)
    with pytest.raises(OptionError, match="CountResult"):
        beyn(f, BOX, poles=0, counted=1)


def test_f9_count_without_poles_gets_the_undeclared_poles_message():
    """Integration review of 0.1.2: a count made without poles= (P unknown, count None) was refused
    as "another operator" when given to beyn(poles=0); the operator is the same, and the message
    must name the missing P."""
    def f(z):
        return z - 0.3                                      # root 0.3, inside

    c = count(f, BOX)
    assert c.count is None
    with pytest.raises(OptionError, match="was not declared"):
        beyn(f, BOX, poles=0, counted=c)


def test_f18_scalar_residual_is_the_documented_reference_scaled_one():
    """F18: for f(z) = z^2 - 2 the reported residual is |f(z)| / max(|f(z)|, s_ref), s_ref the
    median |f| at the corners and edge midpoints of the box, computed here independently."""
    b = cn.box(re=(1, 2), im=(-1, 1))
    r = beyn(lambda z: z * z - 2, b, poles=0)
    z = r.values[0]
    assert abs(z - np.sqrt(2)) < 1e-12
    pts = [complex(1, -1), complex(2, -1), complex(2, 1), complex(1, 1),
           complex(1.5, -1), complex(2, 0), complex(1.5, 1), complex(1, 0)]
    s_ref = float(np.median([abs(p * p - 2) for p in pts]))
    expected = abs(z * z - 2) / max(abs(z * z - 2), s_ref)
    assert r.residuals[0] == pytest.approx(expected, rel=1e-6, abs=1e-30)
    assert "median" in r.residual_definition and f"{s_ref:.3e}" in r.residual_definition
    assert "||T(z) v|| / (||T(z)|| ||v||)" not in r.explain()


def test_f18_pencil_residual_definition():
    L, M = -np.diag([0.2, 3.0]), np.eye(2)
    r = beyn((L, M), BOX)
    assert "||L||_1 + |z| ||M||_1" in r.residual_definition
    z, v = r.values[0], r.vectors[:, 0]
    scale = np.linalg.norm(L, 1) + abs(z) * np.linalg.norm(M, 1)
    expected = np.linalg.norm((L + z * M) @ v) / (scale * np.linalg.norm(v))
    assert abs(r.residuals[0] - expected) <= 1e-6 * expected + 1e-16, (r.residuals[0], expected)
