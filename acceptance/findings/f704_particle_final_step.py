"""F704 (audit of 0.2.1, N14): a final position outside a wall is refused.

Oracle: x' = t from .75 on [0, 1] with RK2 and dt = 1 ends at 1.25 exactly."""
from _acceptance import check, passed, start

start(704, "a final position outside a wall is refused")
import numpy as np  # noqa: E402

from conatus.diagnostics import AnalyticVelocity, Chebyshev, Particles  # noqa: E402
from conatus.errors import CoordinateError  # noqa: E402

p = Particles([[.75]], AnalyticVelocity(lambda x, t: np.full_like(x, t), 1),
              axes=[Chebyshev(0, 1, 8)], scheme="rk2", boundary="raise")
try:
    p.advance(1.)
    check(False, f"accepted a final position {p.x[0, 0]}")
except CoordinateError as err:
    check("left" in str(err), f"unexpected error {err}")
check(p.t == 0. and p.x[0, 0] == .75, "the refused step changed the particle")
passed("refused, state kept")
