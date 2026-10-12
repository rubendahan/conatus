"""dt(expr) must compile M = matrix of expr, never an empty mass operator (no silent dt drop)."""
from __future__ import annotations

import numpy as np
import pytest

from conatus.discretize import discretize
from conatus.discretize.compiler import DiscretizationError
from conatus.ir import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, RealFourier, diff, dt,
                        interp, lap, lift)


def _z(N=16):
    cs = CartesianCoordinates("z")
    z = cs["z"]
    b = Chebyshev(z, N, (-1, 1), dealias=1.5)
    return cs, z, b, Domain(cs, [b])


def _clamped(pb, p, z):
    pb.add_equation((interp(p, z, "left"), 0))
    pb.add_equation((interp(p, z, "right"), 0))
    pb.add_equation((interp(diff(p, z), z, "left"), 0))
    pb.add_equation((interp(diff(p, z), z, "right"), 0))


def _pair(op, rest, variables, bcs, ncc=None):
    """M of 'dt(op) + rest' must equal L('op + rest') - L('rest')."""
    mats = []
    for lhs in (dt(op) + rest, op + rest, rest):
        pb = IVP(list(variables))
        pb.add_equation((lhs, 0))
        bcs(pb)
        used = {f: v for f, v in (ncc or {}).items() if f in lhs.free_fields()}
        g = discretize(pb, ncc=used).matrices()
        mats.append(g)
    return mats[0].M, (mats[1].L - mats[2].L)


def test_dt_of_second_derivative_no_silent_drop():
    cs, z, b, D = _z()
    p = Field("p", D)
    ts = [Field(f"t{i}", D.without(z)) for i in range(4)]
    rest = -diff(diff(diff(diff(p, z), z), z), z) + sum(lift(t, b, -i - 1) for i, t in enumerate(ts))
    M, ref = _pair(diff(diff(p, z), z), rest, [p, *ts], lambda pb: _clamped(pb, p, z))
    assert M.nnz > 0
    assert abs(M - ref).max() < 1e-12


def test_dt_of_laplacian_2d_no_silent_drop():
    cs = CartesianCoordinates("x", "z")
    x, z = cs["x"], cs["z"]
    bx = RealFourier(x, 8, (0, 2 * np.pi), dealias=1.5)
    bz = Chebyshev(z, 12, (-1, 1), dealias=1.5)
    D = Domain(cs, [bx, bz])
    psi = Field("psi", D)
    ts = [Field(f"t{i}", D.without(z)) for i in range(4)]
    rest = -lap(lap(psi)) + sum(lift(t, bz, -i - 1) for i, t in enumerate(ts))

    def bcs(pb):
        _clamped(pb, psi, z)
    Ms = []
    for lhs in (dt(lap(psi)) + rest, lap(psi) + rest, rest):
        pb = IVP([psi, *ts])
        pb.add_equation((lhs, 0))
        bcs(pb)
        plan = discretize(pb)
        Ms.append([plan.matrices(g) for g in plan.groups])
    for gA, gB, gC in zip(*Ms):
        assert gA.M.nnz > 0
        assert abs(gA.M - (gB.L - gC.L)).max() < 1e-12


def test_dt_of_ncc_product_no_silent_drop():
    cs, z, b, D = _z()
    u, a = Field("u", D), Field("a", D)
    t1, t2 = Field("t1", D.without(z)), Field("t2", D.without(z))
    rest = -lap(u) + lift(t1, b, -1) + lift(t2, b, -2)

    def bcs(pb):
        pb.add_equation((interp(u, z, "left"), 0))
        pb.add_equation((interp(u, z, "right"), 0))
    M, ref = _pair(a * u, rest, [u, t1, t2], bcs, ncc={a: lambda Z: 1 + Z ** 2})
    assert M.nnz > 0
    assert abs(M - ref).max() < 1e-12


def test_dt_of_sum_of_two_unknowns_no_silent_drop():
    cs, z, b, D = _z()
    u, v = Field("u", D), Field("v", D)
    ts = [Field(f"t{i}", D.without(z)) for i in range(4)]
    pb = IVP([u, v, *ts])
    pb.add_equation((dt(u) + dt(v) - lap(u) + lift(ts[0], b, -1) + lift(ts[1], b, -2), 0))
    pb.add_equation((dt(v) - lap(v) + lift(ts[2], b, -1) + lift(ts[3], b, -2), 0))
    for f in (u, v):
        pb.add_equation((interp(f, z, "left"), 0))
        pb.add_equation((interp(f, z, "right"), 0))
    g = discretize(pb).matrices()
    n = 16
    M = g.M.toarray()
    # first equation's rows couple to both u and v columns
    rows0 = M[:n]
    assert abs(rows0[:, :n]).max() > 0 and abs(rows0[:, n:2 * n]).max() > 0


def test_nested_dt_refused():
    cs, z, b, D = _z()
    u = Field("u", D)
    t1, t2 = Field("t1", D.without(z)), Field("t2", D.without(z))
    pb = IVP([u, t1, t2])
    pb.add_equation((dt(dt(u)) - lap(u) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(u, z, "left"), 0))
    pb.add_equation((interp(u, z, "right"), 0))
    with pytest.raises(DiscretizationError, match="nested time derivative"):
        discretize(pb).matrices()
