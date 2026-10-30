"""F754 (audit of 0.2.1, N54): stable dimensions change neither the event nor its place.

Oracle: Stuart-Landau pair plus z_j_t = -0.1 z_j: Hopf at mu = 0 only.
"""
from _acceptance import check, passed, start

start(754, "Hopf test under stable embedding")
import numpy as np  # noqa: E402


from conatus.bifurcation import continue_equilibria, ir_system, ode_problem  # noqa: E402
from conatus.continuation import ContinuationSettings  # noqa: E402


def run(n, direction=1, ds=.2):
    names = ["u", "v"] + [f"z{j}" for j in range(n - 2)]

    def rhs(X, P):
        u, v = X["u"], X["v"]
        return [P["mu"] * u - v - (u * u + v * v) * u, u + P["mu"] * v - (u * u + v * v) * v] + \
            [-.1 * X[k] for k in names[2:]]
    s = ir_system(ode_problem(names, {"mu": -.1 * direction}, rhs), ["mu"])
    cfg = ContinuationSettings(ds=ds, ds_min=1e-4, ds_max=ds, tolerance=1e-12, crossing_tol=1e-11,
                               max_points=int(round(.2 / ds)))
    return continue_equilibria(s, np.zeros(s.n), "mu", cfg, direction=direction, marginal_tol=1e-10,
                               tests=("hopf",))


for n in (2, 6, 8, 16, 24):
    for direction in (1, -1):
        cv = run(n, direction)
        check(len(cv.events) == 1, f"n = {n}, {direction}: {len(cv.events)} events")
        e = cv.events[0]
        check(e.kind == "hopf" and e.converged and abs(e.point.z[-1]) < 1e-9, f"n = {n}: {e.point.z[-1]}")
for ds in (.1, .05):
    cv = run(10, 1, ds)
    check(len(cv.events) == 1 and abs(cv.events[0].point.z[-1]) < 1e-9, f"ds = {ds}: {cv.events}")
passed("one Hopf at 0 up to dimension 24")
