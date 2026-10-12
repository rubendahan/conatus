"""Inner products (§30): exact Gram matrices, adjoint = W^-1 A^H W, refusal on tau columns."""
import numpy as np
import pytest

from cases import orr_sommerfeld
from conatus.discretize import discretize
from conatus.ir import (EVP, CartesianCoordinates, ComplexFourier, Domain, Field, Parameter,
                        diff)
from conatus.linearize import InnerProduct, chebyshev_gram


def test_chebyshev_gram_is_exact():
    n = 12
    G = chebyshev_gram(n, 0.0, 2.0)
    x, w = np.polynomial.legendre.leggauss(40)
    T = np.polynomial.chebyshev.chebvander(x, n - 1)
    Gq = (T.T * w) @ T            # on (-1, 1)
    assert np.max(abs(G - Gq)) < 1e-13   # (b - a)/2 = 1


def test_adjoint_for_weighted_product_on_fourier():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = ComplexFourier(x, 16, (0, 3.0), dealias=1)
    D = Domain(cs, [b])
    u, v = Field("u", D), Field("v", D)
    s = Parameter("s", dtype=complex)
    pb = EVP([u, v], eigenvalue=s)
    pb.add_equation((diff(u, x) + 2 * v + s * u, 0))
    pb.add_equation((diff(diff(v, x), x) - u + s * v, 0))
    plan = discretize(pb)
    g = next(g for g in plan.groups if g[0].k > 0)
    A = plan.matrices(g).L
    ip = InnerProduct(plan, {"u": 1.0, "v": 3.0})
    Ad = ip.adjoint(A, g)
    rng = np.random.default_rng(0)
    xv = rng.standard_normal(A.shape[1]) + 1j * rng.standard_normal(A.shape[1])
    yv = rng.standard_normal(A.shape[1]) + 1j * rng.standard_normal(A.shape[1])
    lhs = ip.inner(A @ xv, yv, g)
    rhs = ip.inner(xv, Ad @ yv, g)
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)
    assert not np.allclose(Ad, A.toarray().conj().T)   # weighted: adjoint != conjugate transpose
    assert "weights" in ip.explain()


def test_adjoint_refused_with_tau_columns():
    plan = orr_sommerfeld(16)
    ip = InnerProduct(plan)
    with pytest.raises(ValueError, match="singular"):
        ip.adjoint(plan.matrices().L)
    assert ip.norm(np.ones(plan.matrices().shape[1])) > 0


def test_chebyshev_gram_raises_no_spurious_warning_and_docs_match_behaviour():
    """Docs-site UX review: chebyshev_gram evaluated both np.where branches (1/(1 - k²) at k = 1,
    divide-by-zero RuntimeWarning on a correct result), and the module docstring advertised a
    ``fields_only=True`` option that does not exist (a singular W is refused)."""
    import warnings

    import conatus.linearize.inner_product as ipm
    from conatus.linearize.inner_product import chebyshev_gram
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        G = chebyshev_gram(6)
    assert np.isclose(G[0, 0], 2.0) and np.isclose(G[1, 1], 2.0 / 3.0) and G[0, 1] == 0.0
    assert "fields_only" not in ipm.__doc__
