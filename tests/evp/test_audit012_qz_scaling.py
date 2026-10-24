"""Audit of 0.1.2, F03: the finite/infinite split of dense QZ does not depend on the scale of
each equation.

Oracle: scipy.linalg.eigvals of the nonsingular pencil, and the closed form
det(L + s M) = (s - 1)(s - 1/eps) of L = -I, M = diag(1, eps).
"""
import numpy as np
import pytest
from scipy.linalg import eigvals

from conatus.expert import solve_dense


@pytest.mark.parametrize("eps", [1e-12, 1e-13, 1e-14])
@pytest.mark.parametrize("row", [1.0, 1e-8, 1e13])
def test_small_physical_mass_keeps_its_finite_eigenvalue_in_any_equation_units(eps, row):
    """L = -I, M = diag(1, eps) (a small physical mass, M nonsingular), second equation multiplied
    by ``row``: both eigenvalues 1 and 1/eps are returned, for every row scale (a genuine 1e13 is
    finite)."""
    D = np.diag([1.0, row])
    L, M = D @ -np.eye(2), D @ np.diag([1.0, eps])
    exact = np.sort(eigvals(-L, M).real)
    assert np.allclose(exact, [1.0, 1 / eps], rtol=1e-12)
    r = solve_dense(L, M)
    assert r.n_infinite == 0 and r.values.size == 2, (r.values, r.log)
    assert np.allclose(np.sort(r.values.real), exact, rtol=1e-10, atol=0)
    assert r.infinite_ambiguous.size == 0


@pytest.mark.parametrize("eps", [1e-16, 1e-17])
def test_audit_pencil_and_its_rescaled_form_agree(eps):
    """The audit's pair: L = -I, M = diag(1, eps) and D L, D M with D = diag(1, 1/eps). Before the
    fix the first lost 1/eps as infinite and the second kept it. Both now give the same answer:
    the root 1 (exact) kept, 1/eps reported in infinite_ambiguous with its modulus."""
    L, M = -np.eye(2), np.diag([1.0, eps])
    D = np.diag([1.0, 1 / eps])
    out = []
    for a, b in ((L, M), (D @ L, D @ M)):
        r = solve_dense(a, b)
        out.append((r.values.size, r.n_infinite))
        assert abs(np.sort(r.values.real)[0] - 1.0) < 1e-12
        assert r.infinite_ambiguous.size == 1
        assert r.infinite_ambiguous[0] == pytest.approx(1 / eps, rel=1e-8)
    assert out[0] == out[1] == (1, 1)
