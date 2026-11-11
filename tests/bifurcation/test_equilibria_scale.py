"""The finite / infinite classification of generalized
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


@pytest.mark.parametrize("c", [1e-6, 1.0, 1e6])
def test_pencil_spectrum_ambiguous_moduli_in_the_caller_unit(c):
    """J = c diag(-1, -1), M = diag(1, 1e-14): eigenvalues -c and -c 1e14. The second is finite
    but within the infinite threshold's ambiguity band; its modulus is reported in the caller's
    unit, c 1e14 (0.3.1 review: it was 1e14 whatever c, the unit of the normalised pencil)."""
    from conatus.bifurcation.equilibria import pencil_spectrum
    ps = pencil_spectrum(c * np.diag([-1.0, -1.0]), np.diag([1.0, 1e-14]))
    big = np.concatenate([ps.ambiguous_kept, ps.ambiguous_removed,
                          np.abs(ps.values[np.abs(ps.values) > 10 * c])])
    assert big.size >= 1
    assert np.allclose(big, c * 1e14, rtol=1e-6)
