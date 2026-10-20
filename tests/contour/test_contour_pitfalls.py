"""Known pitfalls of contour counting, as generic tests.

1. A determinant assembled from labelled factors (pivots without their permutation sign, one
   eigenvalue picked by its order) is discontinuous where the labels swap: its phase jumps by
   +-pi there, and a walk that took principal increments anyway could close on a clean but wrong
   integer. The walk must refuse it; the determinant of the whole matrix (permutation signs
   included) counts right.
2. Poles of det T count as negative zeros (N = Z - P): covered in test_contour_count/beyn.
3. Coalescing eigenvalues and branch points inside the box: the count is unaffected (det T is
   analytic there), and Beyn's roots add up to it with multiplicities.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import conatus as cn
from conatus.contour import beyn, count
from conatus.errors import ConvergenceError


def _swap_matrix(z):
    """det = z - 1; partial pivoting swaps its rows where |z| crosses 1."""
    return np.array([[z, 1.0], [1.0, 1.0]], complex)


def test_pivot_order_changes_along_the_contour():
    b = cn.box(re=(0.5, 1.5), im=(-0.5, 0.5))         # crosses |z| = 1: the pivot row changes
    lu = [sla.lu_factor(_swap_matrix(z))[1][0] for z in b.point(np.linspace(0, 4, 41))]
    assert len(set(lu)) == 2                           # the pivot order does change on the walk
    assert count(_swap_matrix, b, poles=0).count == 1  # LU with permutation signs: right

    def pivot_product(z):                              # labelled: prod diag(U), no sign
        return np.prod(np.diag(sla.lu_factor(_swap_matrix(z))[0]))
    with pytest.raises(ConvergenceError, match="refused"):
        count(pivot_product, b, poles=0)
    assert count(lambda z: np.linalg.det(_swap_matrix(z)), b, poles=0).count == 1


def test_branch_labelled_eigenvalue_is_refused_and_the_product_counts():
    c0 = 0.2 + 0.1j                                    # branch point of mu = +-sqrt(z - c0)

    def B(z):
        return np.array([[0, 1], [z - c0, 0]], complex)

    def labelled(z):                                   # the eigenvalue of largest real part
        mu = np.linalg.eigvals(B(z))
        return mu[np.argmax(mu.real)]
    b = cn.box(re=(-1, 1), im=(-1, 1))
    with pytest.raises(ConvergenceError, match="refused"):
        count(labelled, b, poles=0)
    # the permutation-invariant product mu_1 mu_2 = det B = -(z - c0): one zero, counted
    assert count(lambda z: np.prod(np.linalg.eigvals(B(z))), b, poles=0).count == 1
    assert count(B, b, poles=0).count == 1


@pytest.mark.parametrize("eps", [1e-2, 1e-8, 1e-14, 0.0, -1e-8])
def test_coalescence_inside_the_box(eps):
    # roots +-sqrt(eps): distinct, nearly double (a cluster), defective double, imaginary pair
    def T(z):
        return np.array([[z, 1.0], [eps, z]], complex)

    def dT(z):
        return np.eye(2, dtype=complex)
    b = cn.box(re=(-0.45, 0.55), im=(-0.5, 0.52))
    assert count(T, b, poles=0).count == 2
    r = beyn(T, b, poles=0, derivative=dT)
    assert int(r.multiplicity.sum()) == 2 and (r.residuals <= 1e-10).all()
    s = np.sqrt(complex(eps))
    if abs(s) > 1e-5:
        assert r.values.size == 2
        for z in (s, -s):
            assert np.min(np.abs(r.values - z)) < 1e-10
    else:                                              # one root of multiplicity 2
        assert r.values.size == 1 and r.multiplicity[0] == 2
        assert abs(r.values[0]) < 1e-6


def test_semisimple_double_eigenvalue():
    L, M = np.diag([-0.1, -0.1, 0.2, -0.9]), np.eye(4)  # 0.1 twice (two eigenvectors), -0.2, 0.9
    b = cn.box(re=(-0.3, 0.3), im=(-0.3, 0.3))
    r = beyn((L, M), b)
    assert r.count == 3
    i = int(np.argmin(np.abs(r.values - 0.1)))
    assert abs(r.values[i] - 0.1) < 1e-12 and r.multiplicity[i] == 2
    assert "multiplicity 2" in r.explain()


def test_parameter_sweep_through_a_branch_point():
    # the eigenvalues of [[z, 1], [p, z]] as functions of p have a branch point at p = 0:
    # ordering them by a label is not continuous in p, the count of the box never changes
    b = cn.box(re=(-0.45, 0.55), im=(-0.5, 0.52))
    counts = [count(lambda z, p=p: np.array([[z, 1.0], [p, z]], complex), b, poles=0).count
              for p in np.linspace(-0.04, 0.04, 9)]
    assert counts == [2] * 9
