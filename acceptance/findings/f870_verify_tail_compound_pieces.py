"""F870 (hunt 5D on 0.3.0, Q4): verify()'s spectral tail measures every piece of a compound basis.

Oracle: s u = u'' - u on (0, 1), u(0) = u(1) = 0: s_9 = -(9 pi)^2 - 1, eigenfunction sin(9 pi x).
CompoundChebyshev pieces (0, 0.5), (0.5, 1) with sizes (10, 48): the first piece is unresolved and
the eigenvalue is 2 % off; 0.3.0 printed "[PASS] spectral tail 3e-29" (the last piece only), and
FAIL for the mirror sizes (48, 10). Expected: FAIL in both orders; PASS for sizes (48, 48).
"""
import numpy as np
from _acceptance import check, passed, start

start(870, "verify() spectral tail per piece of a compound basis")
from conatus.expert import Parameter, ParametricEVP, linearize  # noqa: E402
from conatus.ir import (IVP, CartesianCoordinates, CompoundChebyshev, Domain, Field, Zero, add,  # noqa: E402
                        diff, dt, interp, lift)
from conatus.verify import verify  # noqa: E402


def tail_and_error(sizes, n=9):
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = CompoundChebyshev(x, sizes, (0, 0.5, 1), dealias=1)
    D = Domain(cs, [b])
    u, r = Field("u", D), Parameter("r", 1.0)
    taus = [Field(f"t{p}{j}", D.without(x)) for p in range(2) for j in (1, 2)]
    pb = IVP([u] + taus, name="heat on two pieces")
    pb.add_equation((dt(u) + r * u - diff(diff(u, x), x)
                     + add(*[lift(taus[2 * p + j - 1], b, -j, piece=p) for p in range(2) for j in (1, 2)]), 0))
    for c in b.interfaces:
        for q in (u, diff(u, x)):
            pb.add_equation((interp(q, x, c, side="above") - interp(q, x, c, side="below"), 0))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    P = ParametricEVP(linearize(pb, around={v: Zero(()) for v in pb.variables},
                                eigenvalue=Parameter("s", dtype=complex)), [r])
    res = P.solve(r=1.0)
    exact = -(n * np.pi) ** 2 - 1.0
    i = int(np.argmin(abs(res.values - exact)))
    mode = P.track(res.values[i], res.vectors[:, i], r=1.0)
    tail = next(c for c in verify(P, mode).checks if c.name == "spectral tail")
    return tail, abs(mode.sigma - exact) / abs(exact)


for sizes in ((10, 48), (48, 10)):
    tail, err = tail_and_error(sizes)
    check(err > 1e-2 and tail.status == "FAIL", f"sizes {sizes}: eigenvalue error {err:.1e}, {tail.explain()}")
tail, err = tail_and_error((48, 48))
check(err < 1e-10 and tail.status == "PASS", f"control (48, 48): error {err:.1e}, {tail.explain()}")
passed("unresolved first or last piece FAIL; resolved control PASS")
