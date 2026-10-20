"""Finding 11 of the external review: the Hill solver failed (NaN) on a regular, well-conditioned
matrix because block-Thomas does not pivot across super-blocks."""
import numpy as np
import pytest

from conatus.errors import NumericalError
from conatus.floquet.hill_general import hill_from_blocks


def _reviewer_hill():
    return hill_from_blocks(np.eye(1), np.zeros((1, 1)), {1: np.ones((1, 1)), -1: np.ones((1, 1))},
                            Nh=1, omega=1., label="audit", note="")


def test_singular_super_block_regular_matrix():
    h = _reviewer_hill()
    A = h.dense(1j)
    sign, logabs = np.linalg.slogdet(A)
    phase, la = h.slogdet(1j)
    assert np.isfinite(la) and abs(la - logabs) < 1e-13 and abs(phase - sign) < 1e-13
    assert h.factor(1j).method.startswith("banded")
    b = np.ones((3, 1))
    x = h.solve(1j, b)
    np.testing.assert_allclose(x.ravel(), np.linalg.solve(A, b.ravel()), atol=1e-14)


@pytest.mark.parametrize("Mh", [1, 2])
def test_banded_and_thomas_agree_with_dense(Mh):
    rng = np.random.default_rng(3)
    blocks = {m: rng.standard_normal((3, 3)) for m in range(-Mh, Mh + 1) if m}
    h = hill_from_blocks(np.eye(3), rng.standard_normal((3, 3)), blocks, Nh=4, omega=1.3,
                         label="t", note="")
    mu = 0.3 + 0.2j
    A = h.dense(mu)
    ref = np.linalg.slogdet(A)
    for F in (h.factor(mu), h._banded(mu)):
        ph, la = F.slogdet()
        assert abs(la - ref[1]) < 1e-11 and abs(ph - ref[0]) < 1e-11
        b = rng.standard_normal((9, 3)) + 0j
        np.testing.assert_allclose(F.solve(b).ravel(), np.linalg.solve(A, b.ravel()), atol=1e-11)


def test_exactly_singular_gives_minus_inf_not_nan():
    h = hill_from_blocks(np.eye(1), np.zeros((1, 1)), {}, Nh=0, omega=1., label="z", note="")
    ph, la = h.slogdet(0.0)
    assert la == -np.inf and ph == 0
    with pytest.raises(NumericalError, match="backward error"):
        h.solve(0.0, np.ones((1, 1)))


def test_winding_is_certified_not_sampled():
    """HillBlocks.winding sampled the phase at n_points fixed points and unwrapped it: a closed
    sampling is an integer by construction, so a coarse one gave a clean wrong count (critic-science: three roots inside, 4 samples -> -1) that certify(winding=...) accepted
    as "a positive integer". It now walks the circle adaptively (conatus.contour)."""
    from conatus.floquet.hill_general import hill_from_blocks
    h = hill_from_blocks(np.eye(1), np.zeros((1, 1)), {1: np.ones((1, 1)), -1: np.ones((1, 1))},
                         Nh=1, omega=1.0, label="winding", note="")
    A0, A1 = h.dense(0.0), h.dense(1.0) - h.dense(0.0)
    roots = np.linalg.eigvals(-np.linalg.solve(A1, A0))          # -1, 0, 1
    assert np.allclose(np.sort(roots.real), [-1, 0, 1], atol=1e-12)
    assert h.winding(0.0, 1.5, 4) == 3.0                          # 4 initial points only
    assert h.winding(0.0, 0.5, 4) == 1.0
    assert h.winding(5.0, 0.5, 8) == 0.0
