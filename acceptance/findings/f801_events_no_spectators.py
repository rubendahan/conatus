"""F801 (external review of 0.2.3, finding 2): events do not move when decoupled stable variables
are added, at sizes up to about 24, and each event checks its own critical distance.

Analytic oracles. Neimark-Sacker: transverse linear part (mu, +-0.3 i), multipliers
exp((mu +- 0.3 i) 2 pi), mu = 0. Period doubling: a rotating projector P(t) gives the Floquet
solution exp(mu t)(cos t/2, sin t/2), multiplier -exp(2 pi mu), mu = 0. Branch point of cycles:
a' = mu a - a^3, multiplier exp(2 pi mu), mu = 0. Fold of cycles: R = x^2 + y^2,
0 = -p + R - R^2, p = 1/4. Equilibria: x' = r - x^2 (fold at 0), x' = r x - x^2 (branch point at
0), the linear part mu +- i (Hopf at 0). A pair with sum 0 off the axis is no Hopf point.
"""
import numpy as np
from _acceptance import check, passed, start

start(801, "located events do not depend on decoupled stable variables")
import conatus.expert as C  # noqa: E402
from conatus.bifurcation import CollocationMesh, PeriodicOrbitProblem, continue_orbits  # noqa: E402

COUNTS = (0, 4, 20)
TOL = 1e-8


def cfg(ds, n=3):
    return C.ContinuationSettings(ds=ds, ds_min=1e-6, ds_max=max(ds, 0.1), tolerance=1e-10, max_points=n)


def orbit(names4, par, f, a0, test, ds, z_extra=None, n=3):
    for count in COUNTS:
        names = list(names4) + [f"s{j}" for j in range(count)]

        def rhs(X, P, names=names):
            return f(*[X[k] for k in names4], P[par]) + [-0.01 * X[k] for k in names[len(names4):]]
        system = C.ir_system(C.ode_problem(names, {par: a0}, rhs), [par])
        pb = PeriodicOrbitProblem(system, par, CollocationMesh.uniform(20, 4))
        tau = pb.mesh.times()
        U = np.zeros((len(tau), len(names)))
        rad = z_extra if z_extra is not None else 1.0
        U[:, 0], U[:, 1] = rad * np.cos(2 * np.pi * tau), rad * np.sin(2 * np.pi * tau)
        z = np.r_[U.ravel(), 2 * np.pi, a0]
        yield count, continue_orbits(pb, z, cfg(ds, n), tests=(test,)).events_of(test)


def ns(x, y, a, b, mu):
    R = 1 - x * x - y * y
    T = mu - a * a - b * b
    return [-y + x * R, x + y * R, T * a - 0.3 * b, 0.3 * a + T * b]


def pd(x, y, a, b, mu):
    R = 1 - x * x - y * y
    p11, p12, p22 = (1 + x) / 2, y / 2, (1 - x) / 2
    # A = mu P - (I - P) + J / 2
    a11, a12, a21, a22 = (mu * p11 - (1 - p11), mu * p12 + p12 - 0.5,
                          mu * p12 + p12 + 0.5, mu * p22 - (1 - p22))
    return [-y + x * R, x + y * R, a11 * a + a12 * b, a21 * a + a22 * b]


def bpc(x, y, a, b, mu):
    R = 1 - x * x - y * y
    return [-y + x * R, x + y * R, mu * a - a * a * a, -b]


for label, f in (("ns", ns), ("pd", pd), ("bpc", bpc)):
    for count, events in orbit("xyab", "mu", f, -0.1, label, 0.06):
        check(len(events) == 1, f"{label}, {count} spectators: {len(events)} events")
        e = events[0]
        check(e.converged and abs(e.point.z[-1]) < TOL,
              f"{label}, {count} spectators: mu = {e.point.z[-1]:.3e}, converged {e.converged}")
        check(e.data["critical_distance"] < TOL,
              f"{label}, {count}: critical distance {e.data['critical_distance']:.3e}")


def lpc(x, y, p):
    R = x * x + y * y
    g = -p + R - R * R
    return [-y + x * g, x + y * g]


for count, events in orbit("xy", "p", lpc, 0.16, "lpc", 0.03, z_extra=np.sqrt(0.8), n=30):
    # the first event: past it the branch runs down to the collapsed orbit R = 0
    check(events and events[0].converged and abs(events[0].point.z[-1] - 0.25) < 1e-6,
          f"lpc, {count} spectators: {[e.point.z[-1] for e in events]}")


def equilibrium(f, n0, x0, r0, test, ds, direction=1.0):
    for count in (0, 4, 22):
        names = [f"x{j}" for j in range(n0)] + [f"s{j}" for j in range(count)]

        def rhs(X, P, names=names):
            r = P["r"]
            # the spectators depend on the parameter, so a determinant changes size along the curve
            return f([X[k] for k in names[:n0]], r) + [-(1 + 40 * (r - r0) ** 2) * X[k] for k in names[n0:]]
        system = C.ir_system(C.ode_problem(names, {"r": r0}, rhs), ["r"])
        s = C.ContinuationSettings(ds=ds, ds_min=1e-6, ds_max=abs(ds), tolerance=1e-10, max_points=5)
        c = C.continue_equilibria(system, np.r_[x0, np.zeros(count)], "r", s, marginal_tol=1e-10,
                                  direction=direction, tests=(test,))
        yield count, c.events_of(test)


for label, f, n0, x0, r0, d, sgn in (
        ("fold", lambda u, r: [r - u[0] ** 2], 1, [1.0], 1.0, 0.4, -1.0),
        ("branch point", lambda u, r: [r * u[0] - u[0] ** 2], 1, [0.0], -0.5, 0.3, 1.0),
        ("hopf", lambda u, r: [r * u[0] - u[1], u[0] + r * u[1]], 2, [0.0, 0.0], -0.5, 0.3, 1.0)):
    for count, events in equilibrium(f, n0, x0, r0, label, d, sgn):
        check(len(events) == 1 and events[0].converged and abs(events[0].point.z[-1]) < TOL,
              f"{label}, {count} spectators: {[(e.point.z[-1], e.converged) for e in events]}")

# eigenvalues (r + 1) +- i and (r - 1) +- i: two sums vanish at r = 0, off the imaginary axis
quad = lambda u, r: [(r + 1) * u[0] - u[1], u[0] + (r + 1) * u[1],  # noqa: E731
                     (r - 1) * u[2] - u[3], u[2] + (r - 1) * u[3]]
for count, events in equilibrium(quad, 4, np.zeros(4), -0.5, "hopf", 0.2):
    check(not events, f"neutral saddle-focus reported as Hopf with {count} spectators")
passed("NS, PD, BPC, LPC, fold, branch point, Hopf stay at their exact parameter up to size 24")
