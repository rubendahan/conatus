"""Regression tests for the (docs/AUDIT_J0.md): silent substitutions that must be refused."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.ir import (EVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, ProblemError,
                        diff, interp, lift)
from conatus.solvers import solve_dense, solve_sparse


def _string(with_eigenvalue: bool):
    """-v'' = sigma v on (-1, 1), Dirichlet; optionally forget the eigenvalue term."""
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, 16, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    v = Field("v", D)
    t1, t2 = Field("t1", D.without(y)), Field("t2", D.without(y))
    s = Parameter("s", dtype=complex)
    lhs = diff(diff(v, y), y) + lift(t1, b, -1) + lift(t2, b, -2)
    if with_eigenvalue:
        lhs = lhs + s * v
    pb = EVP([v, t1, t2], eigenvalue=s)
    pb.add_equation((lhs, 0))
    pb.add_equation((interp(v, y, -1), 0))
    pb.add_equation((interp(v, y, 1), 0))
    return pb


def test_evp_without_eigenvalue_is_refused():
    """A1: the eigenvalue appearing in no equation gave M = 0 and an empty spectrum, no error."""
    pb = _string(with_eigenvalue=False)
    with pytest.raises(ProblemError, match="eigenvalue"):
        pb.check()


def test_evp_with_eigenvalue_passes_check():
    _string(with_eigenvalue=True).check()


@pytest.mark.parametrize("solver", ["dense", "sparse"])
def test_eigen_solvers_refuse_zero_mass_matrix(solver):
    """A1 (solver side): M identically zero means no eigenvalue at all; refuse instead of []."""
    L = sp.identity(8, format="csr") * 2.0
    M = sp.csr_matrix((8, 8))
    with pytest.raises(ValueError, match="M is identically zero"):
        if solver == "dense":
            solve_dense(L, M)
        else:
            solve_sparse(L, M, target=0.5, n_modes=2)


def test_sparse_solver_counts_dropped_modes():
    """A3: shift-invert modes with nu = 0 (infinite sigma) are counted in the log, not dropped silently."""
    n = 10
    L = sp.diags(np.arange(1.0, n + 1)).tocsr()
    M = sp.diags([1.0] * (n - 2) + [0.0, 0.0]).tocsr()     # two infinite eigenvalues
    r = solve_sparse(L, M, target=-1.1, n_modes=4)
    assert any("infinite" in s for s in r.log)
    assert r.values.size + r.n_infinite == min(4, n - 2)


def test_parameter_complex_value_needs_complex_dtype():
    """A2: a complex value in a real-declared Parameter hid its type in the hash and explain()."""
    with pytest.raises(TypeError, match="complex"):
        Parameter("a", 2j)
    p = Parameter("a", 2j, dtype=complex)
    assert np.dtype(p.dtype).kind == "c"
    Parameter("b", 2.0)        # real value, real dtype: fine


def test_linearize_refuses_equation_that_vanishes():
    """A4: an equation whose linearization is identically zero was dropped without a word."""
    from conatus.ir import IVP, dt
    from conatus.linearize import linearize
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, 8, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, w = Field("u", D), Field("w", D)
    s = Parameter("s", dtype=complex)
    ok = IVP([u, w])
    ok.add_equation((dt(u) - diff(diff(u, y), y), 0))
    ok.add_equation((w, 1))                        # affine: linearizes to dw = 0 and is kept
    assert len(linearize(ok, {}, s).equations) == 2
    bad = IVP([u, w])
    bad.add_equation((dt(u) - diff(diff(u, y), y), 0))
    bad.add_equation((w, w))                       # LHS - RHS = 0: linearization vanishes
    with pytest.raises(ProblemError, match="linearizes to 0"):
        linearize(bad, {}, s)


def test_mms_refuses_missing_physical_unknown():
    """A6: manufacture() set every missing unknown to 0, not only the taus."""
    from conatus.ir import LBVP
    from conatus.verify.mms import manufacture
    import sympy as sy
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, 8, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, w = Field("u", D), Field("w", D)
    t1, t2 = Field("t1", D.without(y)), Field("t2", D.without(y))
    pb = LBVP([u, w, t1, t2])
    pb.add_equation((diff(diff(u, y), y) + w + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((w - u, 0))
    pb.add_equation((interp(u, y, -1), 0))
    pb.add_equation((interp(u, y, 1), 0))
    Y = sy.Symbol("y")
    with pytest.raises(ValueError, match="missing"):
        manufacture(pb, {u: 1 - Y ** 2})
    manufacture(pb, {u: 1 - Y ** 2, w: 1 - Y ** 2})        # taus alone may default to 0
