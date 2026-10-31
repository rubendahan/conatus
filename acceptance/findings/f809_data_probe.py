"""F809 (external review of 0.2.3, finding 10): input data the grid cannot see is warned.

Oracle: numpy.polynomial.Chebyshev integrates T64 twice in closed form. At N = 32, T64 is -1 on
every Chebyshev-Gauss node, so the solve gives a parabola with error 0.5. At N = 96 the
solve is exact. A Fourier mode at a multiple of N is the periodic version. Smooth resolved data draws no warning.
"""
import warnings

import numpy as np
from numpy.polynomial import Chebyshev as Cheb
from _acceptance import check, passed, start

start(809, "a callable given is probed off the grid")
import conatus as cn  # noqa: E402
from conatus.errors import UnderResolvedWarning  # noqa: E402

rhs = Cheb.basis(64)
prim = rhs.integ(2)
exact = prim - Cheb([(prim(1) + prim(-1)) / 2, (prim(1) - prim(-1)) / 2])


def solve_t64(N):
    m = cn.Model(cn.Box(z=cn.walls(-1, 1)))
    m.scalar("u")
    m.given("f", lambda z: np.cos(64 * np.arccos(z)))
    m.equations("dz(dz(u))=f")
    m.at("walls", u=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = m.solve(N)
    return sol, [x for x in w if issubclass(x.category, UnderResolvedWarning)]


sol, w = solve_t64(32)
check(np.max(np.abs(sol.grid("u") - exact(sol.coordinate("z")))) > 0.4, "the alias is still order one")
check(any("not resolved by this grid" in str(x.message) for x in w), "N = 32: no warning")
check("INPUT DATA NOT RESOLVED" in sol.explain(), "explain() says nothing")
sol, w = solve_t64(96)
check(np.max(np.abs(sol.grid("u") - exact(sol.coordinate("z")))) < 1e-12, "N = 96 error")
check(not w, f"N = 96: false alarm {[str(x.message) for x in w]}")

m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(-1, 1)))
m.scalar("u")
m.given("f", lambda x, z: np.cos(16 * x) * (1 - z * z))
m.equations("lap(u)-u=f")
m.at("walls", u=0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    m.solve((16, 16))
check(any("not resolved by this grid" in str(x.message) for x in caught), "Fourier alias: no warning")

for f, N in ((lambda z: np.exp(np.sin(3 * z)), 64), (lambda z: 1 + 0 * z, 32), (lambda z: z ** 3, 16)):
    m = cn.Model(cn.Box(z=cn.walls(-1, 1)))
    m.scalar("u")
    m.given("f", f)
    m.equations("dz(dz(u))=f")
    m.at("walls", u=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.solve(N)
    check(not [x for x in caught if "not resolved by this grid" in str(x.message)], f"false alarm N={N}")

# an initial-value problem is warned too
m = cn.Model(cn.Box(z=cn.walls(-1, 1)))
m.scalar("u")
m.given("f", lambda z: np.cos(64 * np.arccos(z)))
m.equations("dt(u)-dz(dz(u))=f")
m.at("walls", u=0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    m.simulate(32, scheme="RK443", dt=0.01)
check(any("not resolved by this grid" in str(x.message) for x in caught), "IVP given: no warning")
# The probe leaves the user's random streams and non-repeatable callables alone.
def solved(f, N=32):
    m = cn.Model(cn.Box(z=cn.walls(-1, 1)))
    m.scalar("u")
    m.given("f", f)
    m.equations("dz(dz(u))=f")
    m.at("walls", u=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(N)


np.random.seed(0)
solved(lambda z: np.cos(z) + 1e-9 * np.random.rand(*np.shape(z)))
got = np.random.rand()
np.random.seed(0)
np.random.rand(32)     # the one evaluation on the grid
check(got == np.random.rand(), "the probe advanced the global NumPy stream")
rng = np.random.default_rng(0)
calls = []


def held(z):
    calls.append(1)
    return np.cos(z) + 1e-9 * rng.normal(size=np.shape(z))


sol = solved(held)
check(len(calls) == 1, f"a held generator was drawn from {len(calls)} times")
check("random generator" in sol.explain(), "the left-alone callable is not named")
state = {"n": 0}


def counting(z):
    state["n"] += 1
    return (1.0 + state["n"]) * np.ones_like(z)


sol = solved(counting, 16)
check("INPUT DATA NOT RESOLVED" not in sol.explain() and "different values" in sol.explain(),
      "a non-repeatable callable was judged")
passed("T64 at N = 32 and a Fourier alias warned; resolved data draws no warning")
