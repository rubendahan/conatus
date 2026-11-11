"""verify()'s "spectral tail" measures every piece of a CompoundChebyshev unknown (hunt 5D, Q4).

Oracle: s u = u'' - r u on (0, 1), u(0) = u(1) = 0, r = 1: s_n = -(n pi)^2 - r with eigenfunction
sin(n pi x). On pieces (0, 0.5) and (0.5, 1) with sizes (10, 48) the mode n = 9 is unresolved on
the first piece and its eigenvalue is 2 % off; the check took the last 4 coefficients of the whole
unknown (the top of the last piece only) and printed PASS. The mirror sizes (48, 10) gave FAIL for
the same eigenvalue. Control: sizes (48, 48) give the exact eigenvalue and PASS.
"""
import numpy as np
import pytest

from conatus.expert import Parameter, ParametricEVP, linearize
from conatus.ir import (IVP, CartesianCoordinates, CompoundChebyshev, Domain, Field, Zero, add, diff,
                        dt, interp, lift)
from conatus.verify import verify


def _pencil(sizes):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = CompoundChebyshev(x, sizes, (0, 0.5, 1), dealias=1)
    D = Domain(cs, [b])
    u, r = Field("u", D), Parameter("r", 1.0)
    P = len(b.sizes)
    taus = [Field(f"t{p}{j}", D.without(x)) for p in range(P) for j in (1, 2)]
    pb = IVP([u] + taus, name="heat on two pieces")
    pb.add_equation((dt(u) + r * u - diff(diff(u, x), x)
                     + add(*[lift(taus[2 * p + j - 1], b, -j, piece=p) for p in range(P) for j in (1, 2)]), 0))
    for c in b.interfaces:
        for q in (u, diff(u, x)):
            pb.add_equation((interp(q, x, c, side="above") - interp(q, x, c, side="below"), 0))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    evp = linearize(pb, around={v: Zero(()) for v in pb.variables}, eigenvalue=Parameter("s", dtype=complex))
    return ParametricEVP(evp, [r])


def _mode(sizes, n):
    P = _pencil(sizes)
    res = P.solve(r=1.0)
    exact = -(n * np.pi) ** 2 - 1.0
    i = int(np.argmin(abs(res.values - exact)))
    mode = P.track(res.values[i], res.vectors[:, i], r=1.0)
    tail = next(c for c in verify(P, mode).checks if c.name == "spectral tail")
    return abs(mode.sigma - exact) / abs(exact), tail


@pytest.mark.parametrize("sizes", [(10, 48), (48, 10)])
def test_unresolved_piece_fails_in_either_order(sizes):
    err, tail = _mode(sizes, 9)
    assert err > 1e-2                        # the oracle: this eigenvalue is wrong by 2 %
    assert tail.status == "FAIL" and tail.value > 0.1, tail.explain()


def test_resolved_pieces_pass():
    err, tail = _mode((48, 48), 9)
    assert err < 1e-10
    assert tail.status == "PASS", tail.explain()
    assert "piece" in tail.what and "not sufficient" in tail.what
