"""Bordered banded solves (C++) vs dense oracle, real and complex, with transpose and batching."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.runtime import BorderedBanded, BorderedBandedBatch


def almost_banded(rng, n, kl, ku, border, dtype):
    """Random well-conditioned banded matrix whose rows `border` are dense (BC-like)."""
    A = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        for j in range(max(0, i - kl), min(n, i + ku + 1)):
            A[i, j] = rng.standard_normal() + (1j * rng.standard_normal() if dtype == complex else 0)
        A[i, i] += 4 * (kl + ku + 1)
    for i in border:
        A[i] = rng.standard_normal(n) + (1j * rng.standard_normal(n) if dtype == complex else 0)
    return A


@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("r", [1, 2, 3, 4])
def test_bordered_solve_matches_dense(dtype, r, rng):
    n, kl, ku = int(rng.integers(20, 80)), int(rng.integers(0, 4)), int(rng.integers(0, 4))
    border = sorted(rng.choice(n, size=r, replace=False).tolist())
    # A permuted so that the border rows are the ones removed from the band: build banded on the
    # permuted layout, then scatter back.
    Ap = almost_banded(rng, n, kl, ku, range(r), dtype)
    order = border + [i for i in range(n) if i not in border]
    A = np.empty_like(Ap)
    A[order] = Ap
    s = BorderedBanded(sp.csr_matrix(A), border)
    assert s.rank == r and s.kl <= kl and s.ku <= ku
    b = rng.standard_normal((n, 3)) + (1j * rng.standard_normal((n, 3)) if dtype == complex else 0)
    x = s.solve(b)
    assert np.max(abs(A @ x - b)) < 1e-12 * np.max(abs(b)) * n
    xt = s.solve(b, transpose=True)
    assert np.max(abs(A.T @ xt - b)) < 1e-12 * np.max(abs(b)) * n
    assert "border rank" in s.explain()


def test_tau_poisson_structure_from_plan(rng):
    from conatus.discretize import discretize
    from conatus.ir import LBVP, CartesianCoordinates, Chebyshev, Domain, Field, interp, lap, lift
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, 64, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, f = Field("u", D), Field("f", D)
    t1, t2 = Field("t1", D.without(x)), Field("t2", D.without(x))
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    gm = discretize(pb).matrices()
    s = BorderedBanded(gm.L, gm.bc_rows)
    assert s.kl + s.ku < 10                      # banded after moving BC rows up, not dense
    rhs = rng.standard_normal(gm.shape[0])
    x_ = s.solve(rhs)
    assert np.max(abs(gm.L @ x_ - rhs)) < 1e-12


def test_singular_band_is_refused():
    A = np.eye(6)
    A[3, 3] = 0.0                                # empty non-border row
    with pytest.raises(RuntimeError, match="singular"):
        BorderedBanded(sp.csr_matrix(A), [0])


def test_batch_matches_individual(rng):
    n, mats, bs = 40, [], []
    for _ in range(6):
        A = almost_banded(rng, n, 2, 2, [0, 1], complex)
        mats.append(sp.csr_matrix(A))
        bs.append(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    batch = BorderedBandedBatch(mats, [0, 1])
    X = batch.solve(np.array(bs))
    for A, b_, x_ in zip(mats, bs, X):
        assert np.max(abs(A @ x_ - b_)) < 1e-11
