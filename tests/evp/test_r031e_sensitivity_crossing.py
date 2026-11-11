"""Eigenvalue sensitivity at a multiple eigenvalue (hunt 5D).

Oracle (Rellich): L(p) = -(I + p B), M = I, B = [[1, 2], [2, -1]] symmetric. The branches through
the double eigenvalue 1 at p = 0 are 1 + p eig(B), slopes +-sqrt(5). The first-order formula on one
left/right pair returned the diagonal entries 1 and -1, with no warning.
"""
import numpy as np
import pytest

from conatus.errors import SpectralError
from conatus.sensitivity import eigenvalue_sensitivity
from conatus.solvers.evp import cluster_sensitivity, multiple_eigenvalue, sensitivity, solve_dense

B = np.array([[1.0, 2.0], [2.0, -1.0]])


def test_double_eigenvalue_is_refused_with_the_rellich_slopes():
    res = solve_dense(-np.eye(2), np.eye(2), left=True)
    for i in range(2):
        with pytest.raises(SpectralError, match="multiple"):
            sensitivity(res, i, np.eye(2), -B)
        with pytest.raises(SpectralError):
            eigenvalue_sensitivity(res, i, np.eye(2), -B)
    slopes = np.sort(cluster_sensitivity(res, multiple_eigenvalue(res, 0), np.eye(2), -B).real)
    assert np.allclose(slopes, [-np.sqrt(5), np.sqrt(5)], rtol=1e-12)


def test_double_eigenvalue_inside_a_rotated_4x4():
    rng = np.random.default_rng(5)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    D = np.diag([1.0, 1.0, 3.0, -2.0])
    Bf = np.zeros((4, 4))
    Bf[:2, :2] = B
    Bf[2:, 2:] = rng.standard_normal((2, 2))
    A, dA = Q @ D @ Q.T, Q @ Bf @ Q.T
    res = solve_dense(-A, np.eye(4), left=True)
    idx = multiple_eigenvalue(res, int(np.argmin(np.abs(res.values - 1.0))))
    assert idx.size == 2
    slopes = np.sort(cluster_sensitivity(res, idx, np.eye(4), -dA).real)
    assert np.allclose(slopes, [-np.sqrt(5), np.sqrt(5)], rtol=1e-8)
    j = int(np.argmin(np.abs(res.values - 3.0)))         # a simple eigenvalue: slope B_f[2, 2]
    assert np.isclose(sensitivity(res, j, np.eye(4), -dA), Bf[2, 2], rtol=1e-10)


def test_defective_eigenvalue_is_refused():
    J = np.array([[1.0, 1.0], [0.0, 1.0]])
    res = solve_dense(-J, np.eye(2), left=True)
    idx = multiple_eigenvalue(res, 0)
    assert idx.size == 2
    with pytest.raises(SpectralError, match="defective"):
        cluster_sensitivity(res, idx, np.eye(2), -B)
