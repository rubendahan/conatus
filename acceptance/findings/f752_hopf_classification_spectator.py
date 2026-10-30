"""F752 (audit of 0.2.1, N51): a spectator mode changes neither the event type nor its place.

Oracle: polar form r_t = p r - r^3, theta_t = 1 plus z_t = -c z: Hopf at p = 0, for every c.
"""
from _acceptance import check, passed, start

start(752, "Hopf crossing next to a stable real mode")
import numpy as np  # noqa: E402


from conatus.continuation import ContinuationSettings, PseudoArclength  # noqa: E402


def run(slow, direction):
    def G(x, p):
        r2 = x[0] ** 2 + x[1] ** 2
        out = [p * x[0] - x[1] - r2 * x[0], x[0] + p * x[1] - r2 * x[1]]
        return np.array(out + ([slow * x[2]] if slow is not None else []))

    def J(x, p):
        r2 = x[0] ** 2 + x[1] ** 2
        a = np.array([[p - r2 - 2 * x[0] ** 2, -1 - 2 * x[0] * x[1]],
                      [1 - 2 * x[0] * x[1], p - r2 - 2 * x[1] ** 2]])
        if slow is None:
            return a
        return np.block([[a, np.zeros((2, 1))], [np.zeros((1, 2)), np.array([[slow]])]])
    n = 2 if slow is None else 3
    cfg = ContinuationSettings(ds=.2, ds_min=1e-3, ds_max=.2, tolerance=1e-12, max_points=1)
    eig = lambda x, p: np.linalg.eigvals(J(x, p))  # noqa: E731
    c = PseudoArclength(G, J, cfg, dGdp=lambda x, p: np.array(list(x[:2]) + ([0.] if n == 3 else [])),
                        eigen=eig, marginal_tol=1e-10)
    return c.run(np.zeros(n), -.1 * direction, direction=direction)


for slow in (None, -1., -.01, -.001):
    for direction in (1, -1):
        br = run(slow, direction)
        check(len(br.events) == 1, f"{slow}, {direction}: {br.events}")
        e = br.events[0]
        check(e.kind == "hopf", f"{slow}, {direction}: {e.kind}")
        check(e.converged and abs(e.p) < 1e-9, f"{slow}, {direction}: p = {e.p}, {e.converged}")
passed("Hopf at 0 with every spectator")
