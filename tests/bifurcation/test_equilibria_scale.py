"""Finding 13: the finite / infinite classification of generalized
eigenvalues depended on the scale of the equations."""
import numpy as np
import pytest

from conatus.bifurcation.equilibria import generalized_eigenvalues


@pytest.mark.parametrize("cJ, cM", [(1.0, 1e-12), (1e12, 1.0), (1e-8, 1e-8), (1e8, 1.0)])
def test_scale_invariance(cJ, cM):
    rng = np.random.default_rng(1)
    J = rng.standard_normal((4, 4))
    M = np.diag([1.0, 2.0, 0.0, 0.0])              # two infinite eigenvalues (DAE rows)
    ref = generalized_eigenvalues(J, M)
    got = generalized_eigenvalues(cJ * J, cM * M)
    assert got.size == ref.size == 2
    np.testing.assert_allclose(got, ref * cJ / cM, rtol=1e-9)


def test_reviewer_case():
    I = np.eye(2)
    np.testing.assert_allclose(generalized_eigenvalues(I, 1e-12 * I), [1e12, 1e12])
    np.testing.assert_allclose(generalized_eigenvalues(1e12 * I, I), [1e12, 1e12])
    np.testing.assert_allclose(generalized_eigenvalues(0 * I, I), [0, 0])
