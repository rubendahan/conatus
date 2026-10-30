"""F746 (audit of 0.2.1, N30): the cutoff applies to the crossing time.

Oracle: x(t) = t - origin - 0.5, crossing at origin + 0.5.
"""
from _acceptance import check, passed, start

start(746, "Poincare cutoff")
import numpy as np  # noqa: E402


from conatus.manifolds import PoincareSection, poincare_crossings  # noqa: E402
from conatus.manifolds.trajectory import Trajectory  # noqa: E402

for origin in (0., 10.):
    tr = Trajectory(np.array([origin, origin + 1]), np.array([[-.5], [.5]]), "exact affine path",
                    lambda i, t, origin=origin: np.array([t - origin - .5]))
    sec = PoincareSection.hyperplane([1.], [0.], direction=1)
    for cut, expected in ((None, 1), (origin + .25, 1), (origin + .75, 0), (origin + 1.1, 0)):
        cr = poincare_crossings(tr, sec, xtol=1e-12, t_min=cut)
        check(cr.t.size == expected, f"origin {origin}, cutoff {cut}: {cr.t}")
        if expected:
            check(abs(cr.t[0] - origin - .5) < 1e-11, f"{cr.t}")
passed("cutoff on the crossing time")
