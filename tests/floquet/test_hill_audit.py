"""AUDIT_J1 regressions for the general Hill builder (hill_general.hill_from_plan)."""
import pytest


def test_plan_hill_refuses_base_flow_in_mass():
    """AUDIT_J1 B1: a base flow inside the σ (mass) term was dropped silently (only M(U=0) kept)."""
    from conatus.discretize import discretize
    from conatus.floquet.hill_general import hill_from_plan
    from conatus.ir import EVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, diff, interp, lift
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, 12, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    th, U = Field("theta", D), Field("U", D)
    t1, t2 = Field("t1", D.without(y)), Field("t2", D.without(y))
    s = Parameter("s", dtype=complex)
    pb = EVP([th, t1, t2], eigenvalue=s, name="mass with base flow")
    pb.add_equation((s * th + s * U * th - diff(diff(th, y), y) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(th, y, -1), 0))
    pb.add_equation((interp(th, y, 1), 0))
    base = {U: {1: lambda Y: 0.5 + 0 * Y + 0j, -1: lambda Y: 0.5 + 0 * Y + 0j}}
    with pytest.raises(ValueError, match="mass"):
        hill_from_plan(lambda ncc: discretize(pb, ncc={U: ncc[U]}), base, Nh=4, omega=1.0, label="m")
