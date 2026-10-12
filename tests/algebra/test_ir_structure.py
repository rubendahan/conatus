"""Level A (§69): structure of the IR — hashing, immutability, canonical forms, typing."""
import pytest

from conatus.ir import (CartesianCoordinates, Chebyshev, ComplexFourier, Constant, Domain,
                        Field, Legendre, Parameter, PolarCoordinates, CylindricalCoordinates, RealFourier,
                        SphericalCoordinates, Ultraspherical, Jacobi, VectorField, add, curl,
                        diff, div, dot, grad, interp, lap, lift, mul)


@pytest.fixture
def box2d():
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, 16, (0, 6.283185307179586), dealias=3 / 2)
    bz = Chebyshev(z, 12, (-1, 1), dealias=3 / 2)
    return cs, x, z, bx, bz, Domain(cs, [bx, bz])


def test_structural_hash_stable_and_equality(box2d):
    cs, x, z, bx, bz, D = box2d
    u = Field("u", D)
    e1 = u * diff(u, x) + lap(u)
    # rebuilt from scratch, other construction order
    u2 = Field("u", Domain(cs, [bz, bx]))
    e2 = lap(u2) + diff(u2, x) * u2
    assert e1 == e2
    assert hash(e1) == hash(e2)
    assert e1.structural_hash == e2.structural_hash
    assert e1 != lap(u)


def test_structural_hash_is_process_independent(box2d):
    # sha256 of the structural key: fixed string, not Python's randomized hash()
    cs, x, z, bx, bz, D = box2d
    u = Field("u", D)
    h = (u * diff(u, x)).structural_hash
    assert isinstance(h, str) and len(h) == 16
    assert h == (diff(u, x) * u).structural_hash


def test_immutability(box2d):
    cs, x, z, bx, bz, D = box2d
    u = Field("u", D)
    e = diff(u, x)
    with pytest.raises(AttributeError):
        e.coord = z
    with pytest.raises(AttributeError):
        u.name = "v"
    with pytest.raises(AttributeError):
        del e.args


def test_canonical_add_mul(box2d):
    cs, x, z, bx, bz, D = box2d
    u = Field("u", D)
    assert (u + u) == 2 * u
    assert (u - u).is_zero()
    assert (2 * (3 * u)) == 6 * u
    assert (0 * u).is_zero()
    assert mul(1, u) == u
    assert add(u, 0) == u
    assert (u * diff(u, z)) == (diff(u, z) * u)


def test_tensor_typing(box2d):
    cs, x, z, bx, bz, D = box2d
    u = VectorField("u", D)
    p = Field("p", D)
    assert grad(p).rank == 1
    assert div(u).rank == 0
    assert grad(u).rank == 2
    assert dot(u, grad(u)).rank == 1
    assert curl(u).rank == 0 and curl(p).rank == 1
    with pytest.raises(TypeError):
        u + p
    with pytest.raises(TypeError):
        u * u  # ambiguous: dot or outer


def test_derivative_of_constant_is_zero(box2d):
    cs, x, z, bx, bz, D = box2d
    Re = Parameter("Re", 1000.0)
    assert diff(Re, x).is_zero()
    assert lap(Constant(3.0)).is_zero()
    tau = Field("tau", D.without(z))
    assert diff(tau, z).is_zero()  # constant along z


def test_domain_separability_and_union(box2d):
    cs, x, z, bx, bz, D = box2d
    assert D.separable_coords == (x,)
    assert D.nonperiodic_coords == (z,)
    assert Domain.union(D.without(z), D) == D
    other = Domain(cs, [Chebyshev(z, 20, (-1, 1), dealias=3 / 2)])
    with pytest.raises(ValueError, match="incompatible bases"):
        Domain.union(D, other)


def test_lift_and_interp_domains(box2d):
    cs, x, z, bx, bz, D = box2d
    u = Field("u", D)
    tau = Field("tau", D.without(z))
    assert interp(u, z, "right").domain == D.without(z)
    assert lift(tau, bz, -1).domain == D
    with pytest.raises(ValueError):
        lift(u, bz, -1)  # u depends on z
    with pytest.raises(ValueError):
        lift(tau, bx, -1)  # periodic basis


def test_dealias_must_be_explicit():
    cs = CartesianCoordinates("x")
    x = cs["x"]
    for ctor in (lambda **k: RealFourier(x, 8, (0, 1), **k),
                 lambda **k: ComplexFourier(x, 8, (0, 1), **k),
                 lambda **k: Chebyshev(x, 8, (-1, 1), **k),
                 lambda **k: Legendre(x, 8, (-1, 1), **k),
                 lambda **k: Jacobi(x, 8, (-1, 1), a=0.5, b=0.5, **k),
                 lambda **k: Ultraspherical(x, 8, (-1, 1), lam=1, **k)):
        with pytest.raises(ValueError, match="dealias must be given explicitly"):
            ctor()
        with pytest.raises(TypeError):
            ctor(dealias=True)
        assert ctor(dealias=1).dealias == 1.0
        assert "dealias" in ctor(dealias=3 / 2).explain()


def test_curvilinear_declared_not_silently_used():
    cc = CylindricalCoordinates("phi", "r", "z")          # implemented
    assert cc.scale_factors() == ("r", 1, 1) and "eps" in cc.regularity()
    pc = PolarCoordinates("phi", "r")                      # implemented (regular bases)
    assert "r^|m|" in pc.regularity() and pc.scale_factors() == ("r", 1)
    assert Domain(pc, []).dim == 0                         # polar domains wired
    with pytest.raises(ValueError, match="DiskBasis"):     # no plain Chebyshev in r (axis!)
        Domain(pc, [Chebyshev(pc["r"], 8, (0, 1), dealias=1)])
    sc = SphericalCoordinates("phi", "theta", "r")
    assert "poles" in sc.explain()
    assert Domain(sc, []).dim == 0                         # spherical domains wired
    with pytest.raises(ValueError, match="SphereBasis"):   # no Chebyshev in theta (poles!)
        Domain(sc, [Chebyshev(sc["theta"], 8, (0, 3), dealias=1)])
