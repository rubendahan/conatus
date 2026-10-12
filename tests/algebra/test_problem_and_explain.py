"""Level A (§69, §14, §87): LHS/RHS checks, counting report, explain() on every node type."""
import numpy as np
import pytest

from conatus import ir
from conatus.ir import (EVP, IVP, LBVP, CartesianCoordinates, Chebyshev, Domain, Equation, Field,
                        NonlinearLHSError, Parameter, ProblemError, RealFourier, VectorField,
                        ComplexFourier, Legendre, Ultraspherical, Jacobi, PolarCoordinates,
                        conj, comp, curl, diff, div, dot, dt, grad, integ, interp, lap, lift,
                        outer, power)


def _poisson_1d(with_bcs=2):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, 32, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u, f = Field("u", D), Field("f", D)
    t1, t2 = Field("tau1", D.without(x)), Field("tau2", D.without(x))
    pb = LBVP([u, t1, t2])
    pb.add_equation((lap(u) + lift(t1, b, -1) + lift(t2, b, -2), f))
    if with_bcs >= 1:
        pb.add_equation((interp(u, x, "left"), 0))
    if with_bcs >= 2:
        pb.add_equation((interp(u, x, "right"), 0))
    return pb


def test_poisson_counting_consistent():
    rep = _poisson_1d().rank_report()
    assert rep.square and rep.status == "CONSISTENT"
    d = rep.directions[0]
    assert (d.order_sum, d.lifted_tau_components, d.boundary_components,
            d.tau_variable_components) == (2, 2, 2, 2)


def test_missing_bc_fails():
    rep = _poisson_1d(with_bcs=1).rank_report()
    assert rep.status == "FAIL"
    assert "not square" in rep.explain()


def test_nonlinear_lhs_refused():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [RealFourier(x, 32, (0, 2 * np.pi), dealias=3 / 2)])
    u = Field("u", D)
    pb = IVP([u])
    with pytest.raises(NonlinearLHSError, match="RHS"):
        pb.add_equation((dt(u) + u * diff(u, x), 0))
    pb.add_equation((dt(u) - 0.1 * lap(u), -u * diff(u, x)))  # Burgers, correct split
    assert "IVP" in pb.explain()


def test_forcing_on_lhs_refused_and_rhs_rules():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [Chebyshev(x, 16, (-1, 1), dealias=1)])
    u, f = Field("u", D), Field("f", D)
    with pytest.raises(ProblemError, match="does not depend"):
        LBVP([u]).add_equation((lap(u) + f, 0))
    with pytest.raises(ProblemError, match="depends on the variables"):
        LBVP([u]).add_equation((lap(u), u * u))
    with pytest.raises(ProblemError, match="dt"):
        LBVP([u]).add_equation((dt(u), 0))
    with pytest.raises(ProblemError, match="time derivative on the RHS"):
        IVP([u]).add_equation((u, dt(u)))


def test_evp_declaration():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, 16, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    u = Field("u", D)
    t1, t2 = Field("t1", D.without(x)), Field("t2", D.without(x))
    s = Parameter("s")
    pb = EVP([u, t1, t2], eigenvalue=s)
    pb.add_equation((s * u - lap(u) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(u, x, -1), 0))
    pb.add_equation((interp(u, x, 1), 0))
    assert pb.rank_report().status == "CONSISTENT"
    assert "Eigenvalue: s" in pb.explain()
    with pytest.raises(ProblemError):
        pb.add_equation((u, u))


def test_rayleigh_benard_2d_counting():
    """RB-like counting (Dedalus-style taus). Buoyancy omitted: needs unit vectors."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, 32, (0, 4), dealias=3 / 2)
    bz = Chebyshev(z, 16, (0, 1), dealias=3 / 2)
    D = Domain(cs, [bx, bz])
    Dx = D.without(z)
    p, b = Field("p", D), Field("b", D)
    u = VectorField("u", D)
    tp = Field("tau_p", Domain(cs, []))
    tb1, tb2 = Field("tau_b1", Dx), Field("tau_b2", Dx)
    tu1, tu2 = VectorField("tau_u1", Dx), VectorField("tau_u2", Dx)
    Ra, Pr = Parameter("Ra", 1e6), Parameter("Pr", 1.0)
    pb = IVP([p, b, u, tp, tb1, tb2, tu1, tu2])
    kappa, nu = (Ra * Pr) ** -0.5, (Ra / Pr) ** -0.5
    pb.add_equation((div(u) + tp, 0))
    pb.add_equation((dt(b) - kappa * lap(b) + lift(tb1, bz, -1) + lift(tb2, bz, -2),
                     -dot(u, grad(b))))
    pb.add_equation((dt(u) - nu * lap(u) + grad(p) + lift(tu1, bz, -1) + lift(tu2, bz, -2),
                     -dot(u, grad(u))))
    pb.add_equation((interp(b, z, "left"), 1))
    pb.add_equation((interp(u, z, "left"), 0))
    pb.add_equation((interp(b, z, "right"), 0))
    pb.add_equation((interp(u, z, "right"), 0))
    pb.add_equation((integ(p), 0))
    rep = pb.rank_report()
    assert rep.square
    assert rep.gauge_components == 1
    # continuity has order 1 in z but no z-tau (closed by the pressure gauge): honest CHECK
    dz = rep.directions[0]
    assert (dz.order_sum, dz.lifted_tau_components, dz.boundary_components,
            dz.tau_variable_components) == (7, 6, 6, 6)
    assert rep.status == "CHECK"
    assert "Direction z" in rep.explain()


def _all_nodes():
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = ComplexFourier(x, 8, (0, 2 * np.pi), dealias=1)
    bz = Chebyshev(z, 8, (-1, 1), dealias=1)
    D = Domain(cs, [bx, bz])
    u = Field("u", D, dtype=np.complex128)
    v = VectorField("v", D)
    tau = Field("tau", D.without(z))
    P = Parameter("P", 2.0)
    return [u, v, P, ir.Constant(3), u + 1, P * u, dot(v, v), outer(v, v), power(u, 2), conj(u),
            comp(v, 1), diff(u, x), grad(u), div(v), lap(u), curl(v), curl(u), dt(u), integ(u),
            interp(u, z, 0.5), lift(tau, bz, -1)]


@pytest.mark.parametrize("node", _all_nodes(), ids=lambda n: type(n).__name__)
def test_explain_never_crashes(node):
    text = node.explain()
    assert type(node).__name__ in text
    assert node.inspect()["node"] == type(node).__name__
    assert node.tree()


def test_explain_on_objects():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    for b in (RealFourier(x, 8, (0, 1), dealias=1), Legendre(x, 8, (-1, 1), dealias=1),
              Ultraspherical(x, 8, (-1, 1), lam=1.5, dealias=1),
              Jacobi(x, 8, (-1, 1), a=0, b=1, dealias=2)):
        assert "coefficients" in b.explain()
    assert "metric" in cs.explain()
    assert "separable" in Domain(cs, [RealFourier(x, 8, (0, 1), dealias=1)]).explain()
    assert "DiskBasis" in PolarCoordinates("phi", "r").explain()
    assert "System count" in _poisson_1d().explain()
    assert "Equation" in Equation(lap(Field("u", Domain(cs, [Chebyshev(x, 4, (-1, 1), dealias=1)]))), 0).explain()
