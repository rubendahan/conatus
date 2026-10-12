"""Plan: public matrices, rank report refusing ill-posed systems (§14), gauges, symmetry subspace."""
import numpy as np
import pytest
import scipy.sparse as sp

from conatus.discretize import RankError, discretize
from conatus.discretize.compiler import DiscretizationError
from conatus.ir import (EVEN, LBVP, ODD, CartesianCoordinates, Chebyshev, Domain, Field,
                        RealFourier, Reflection, SymmetryError, SymmetrySubspace, diff, integ,
                        interp, lap, lift)
from conatus.reference import FieldData
from conatus.reference import transforms as T


def _setup(N=32):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    return cs, x, b, D, Field("u", D), Field("f", D), Field("tau1", D.without(x)), \
        Field("tau2", D.without(x))


def test_public_matrices_are_sparse_and_banded():
    cs, x, b, D, u, f, t1, t2 = _setup()
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    plan = discretize(pb)
    gm = plan.matrices()
    assert sp.issparse(gm.L) and gm.shape == (34, 34)
    assert list(gm.bc_rows) == [32, 33] and list(gm.tau_cols) == [32, 33]
    bulk = gm.L[:32, :32]
    assert bulk.nnz == 30                     # D^2 in C^(2): one superdiagonal
    assert gm.lam[(0, ())] == 2
    assert "PASS" in plan.explain()


def test_missing_bc_refused():
    cs, x, b, D, u, f, t1, t2 = _setup()
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    with pytest.raises(RankError, match="FAIL"):
        discretize(pb).check()


def test_extra_bc_refused():
    cs, x, b, D, u, f, t1, t2 = _setup()
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    for p in (-1, 1, 0):
        pb.add_equation((interp(u, x, p), 0))
    with pytest.raises(RankError, match="not square|FAIL"):
        discretize(pb).solve({f: lambda X: 0 * X})


def test_pure_neumann_without_gauge_refused_with_gauge_solved():
    cs, x, b, D, u, f, t1, t2 = _setup()
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(diff(u, x), x, -1), 0))
    pb.add_equation((interp(diff(u, x), x, 1), 0))
    with pytest.raises(RankError) as err:
        discretize(pb).check()
    assert "Structural rank" in str(err.value)
    c = Field("c", Domain(cs, []))
    pb = LBVP([u, t1, t2, c])
    pb.add_equation((lap(u) + c + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(diff(u, x), x, -1), 0))
    pb.add_equation((interp(diff(u, x), x, 1), 0))
    pb.add_equation((integ(u), 0))
    plan = discretize(pb)
    rep = plan.rank_report()
    assert rep.ok and rep.equations["constraint"] == 1 and rep.unknowns["constant"] == 1
    sol = plan.solve({f: lambda X: -np.pi ** 2 * np.cos(np.pi * X)})
    X = T.grid(b, dealias=False)
    assert np.max(abs(sol.grid("u") - np.cos(np.pi * X))) < 1e-13
    assert abs(sol.coeffs["c"][0]) < 1e-12


def test_structural_and_numerical_singularity():
    # pure Neumann with a pinned constant: the T_0 column of u is empty -> structural FAIL
    cs, x, b, D, u, f, t1, t2 = _setup(16)
    c = Field("c", Domain(cs, []))
    pb = LBVP([u, t1, t2, c])
    pb.add_equation((lap(u) + c + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(diff(u, x), x, -1), 0))
    pb.add_equation((interp(diff(u, x), x, 1), 0))
    pb.add_equation((c, 0))
    rep = discretize(pb).rank_report()
    assert rep.structural_rank == rep.n_unknowns - 1 and not rep.ok
    # Helmholtz exactly at the first Dirichlet eigenvalue: structurally full, numerically singular
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + (np.pi / 2) ** 2 * u + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, -1), 0))
    pb.add_equation((interp(u, x, 1), 0))
    rep = discretize(pb).rank_report()
    assert rep.structural_rank == rep.n_unknowns
    assert rep.numerical.startswith("deficient") and "eigenvalue" in rep.null_hint
    assert not rep.ok and "FAIL" in rep.explain()


def test_bare_arrays_refused_fielddata_accepted():
    cs, x, b, D, u, f, t1, t2 = _setup(16)
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    plan = discretize(pb)
    X = T.grid(b, dealias=False)
    with pytest.raises(TypeError, match="explicit layout"):
        plan.solve({f: np.ones(16)})
    s1 = plan.solve({f: FieldData(f, np.exp(X), "grid")})
    s2 = plan.solve({f: FieldData(f, T.forward(np.exp(X), b, dealias=False), "coeff")})
    assert np.max(abs(s1.coeffs["u"] - s2.coeffs["u"])) < 1e-14


def test_ncc_reported_in_plan():
    cs, x, b, D, u, f, t1, t2 = _setup(24)
    a = Field("a", D)
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + a * diff(u, x) + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    plan = discretize(pb, ncc={a: lambda X: 1 + X ** 2})
    X = T.grid(b, dealias=False)
    ue = np.sin(np.pi * X)
    sol = plan.solve({f: lambda X: -np.pi ** 2 * np.sin(np.pi * X)
                      + (1 + X ** 2) * np.pi * np.cos(np.pi * X)})
    assert np.max(abs(sol.grid("u") - ue)) < 1e-12
    text = plan.explain()
    assert "NCC a: 3 of 24" in text or "NCC a:" in text
    rep = next(iter(plan.ncc_reports.values()))
    assert rep.cutoff is None


def test_ncc_along_fourier_refused():
    cs = CartesianCoordinates("x", "z")
    x, z = cs["x"], cs["z"]
    bx, bz = RealFourier(x, 8, (0, 1), dealias=1), Chebyshev(z, 8, (-1, 1), dealias=1)
    D = Domain(cs, [bx, bz])
    u, f, a = Field("u", D), Field("f", D), Field("a", D)
    t1, t2 = Field("t1", D.without(z)), Field("t2", D.without(z))
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + a * u + lift(t1, bz, -1) + lift(t2, bz, -2), f))
    pb.add_equation((interp(u, z, -1), 0))
    pb.add_equation((interp(u, z, 1), 0))
    with pytest.raises(DiscretizationError, match="Fourier"):
        discretize(pb, ncc={a: lambda X, Z: np.cos(2 * np.pi * X) + 0 * Z}).matrices()


def test_odd_poisson_in_symmetry_subspace():
    cs, x, b, D, u, f, t1, t2 = _setup(32)
    Rx = Reflection(x, about=0)
    fo = Field("f", D, symmetry={Rx: ODD})
    G = SymmetrySubspace(Rx, u=ODD, tau1=EVEN, tau2=ODD)
    pb = LBVP([u, t1, t2], symmetry=G)
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), fo))
    pb.add_equation((interp(u, x, -1), 0))
    pb.add_equation((interp(u, x, 1), 0))
    plan = discretize(pb)
    red, full = plan.matrices(reduced=True), plan.matrices(reduced=False)
    assert red.shape == (17, 17) and full.shape == (34, 34)       # size halved
    assert red.L.nnz * 2 <= full.L.nnz + 2
    fe = lambda X: np.sin(3 * X) * np.exp(-X ** 2)
    sr, sf = plan.solve({fo: fe}), plan.solve({fo: fe}, reduced=False)
    assert np.max(abs(sr.coeffs["u"] - sf.coeffs["u"])) < 1e-14
    assert np.max(abs(sr.coeffs["u"][::2])) == 0                 # E injects odd modes only
    with pytest.raises(SymmetryError, match="outside the symmetry subspace"):
        plan.solve({fo: lambda X: np.cos(X)})
    assert "symmetry-breaking modes are not represented" in plan.explain()


def test_fourier_reflection_subspace_2d():
    cs = CartesianCoordinates("x", "z")
    x, z = cs["x"], cs["z"]
    bx, bz = RealFourier(x, 16, (0, 2 * np.pi), dealias=1), Chebyshev(z, 16, (-1, 1), dealias=1)
    D = Domain(cs, [bx, bz])
    Dx = D.without(z)
    Rx = Reflection(x, about=0)
    u, f = Field("u", D), Field("f", D, symmetry={Rx: ODD})
    t1, t2 = Field("t1", Dx), Field("t2", Dx)
    G = SymmetrySubspace(Rx, u=ODD, t1=ODD, t2=ODD)
    pb = LBVP([u, t1, t2], symmetry=G)
    pb.add_equation((lap(u) + lift(t1, bz, -1) + lift(t2, bz, -2), f))
    pb.add_equation((interp(u, z, -1), 0))
    pb.add_equation((interp(u, z, 1), 0))
    plan = discretize(pb)
    fe = lambda X, Z: np.sin(X) * np.cos(Z) + np.sin(3 * X) * Z ** 2
    sr, sf = plan.solve({f: fe}), plan.solve({f: fe}, reduced=False)
    assert np.max(abs(sr.coeffs["u"] - sf.coeffs["u"])) < 1e-14
    g = plan.groups[1]
    assert plan.matrices(g, True).shape[0] * 2 == plan.matrices(g, False).shape[0]
    assert plan.matrices(plan.groups[0], True).shape == (0, 0)    # k=0 has no sine


def test_embed_project_and_ivp_mass_matrix():
    cs, x, b, D, u, f, t1, t2 = _setup(16)
    Rx = Reflection(x, about=0)
    G = SymmetrySubspace(Rx, u=ODD, tau1=EVEN, tau2=ODD)
    pb = LBVP([u, t1, t2], symmetry=G)
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), Field("f", D, symmetry={Rx: ODD})))
    pb.add_equation((interp(u, x, -1), 0))
    pb.add_equation((interp(u, x, 1), 0))
    plan = discretize(pb)
    y = np.random.default_rng(3).standard_normal(9)
    assert np.array_equal(plan.project(plan.embed(y)), y)
    xf = plan.embed(y)
    assert np.array_equal(plan.embed(plan.project(xf)), xf)

    from conatus.ir import IVP, dt
    from conatus.discretize import ultraspherical as us
    pb = IVP([u, t1, t2])
    pb.add_equation((dt(u) - 0.1 * lap(u) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(u, x, -1), 0))
    pb.add_equation((interp(u, x, 1), 0))
    gm = discretize(pb).matrices()
    S = us.conversion_chain(16, 0, 2).toarray()
    assert np.max(abs(gm.M[:16, :16].toarray() - S)) < 1e-15        # M = S_1 S_0 on the bulk
    assert gm.M[16:].nnz == 0 and gm.M[:, 16:].nnz == 0             # no dt on BCs / taus
    D2 = us.differentiation(16, 0, 2).toarray()                    # interval (-1, 1): d/dx = d/dt
    assert np.max(abs(gm.L[:16, :16].toarray() + 0.1 * D2)) < 1e-12


def test_ncc_key_not_in_problem_refused():
    """A coefficient supplied for a field the problem never uses is refused (typo guard)."""
    cs, x, b, D, u, f, t1, t2 = _setup(24)
    a, stray = Field("a", D), Field("stray", D)
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + a * u + lift(t1, b, -1) + lift(t2, b, -2), f))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    with pytest.raises(DiscretizationError, match="stray"):
        discretize(pb, ncc={a: lambda X: 1 + X, stray: lambda X: X})
