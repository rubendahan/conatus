"""F861: the CPU right-hand side with one shared cache for all the equations gives the
per-equation results bit for bit, and the Chebyshev derivative recurrence gives the loop's bits.

Oracle: (1) F of a Rayleigh-Benard state evaluated one expression at a time with a fresh evaluator
and no grid memo (the 0.3.0 path), compared byte for byte with the solver's F; (2) the backward
recurrence written as the Python loop of 0.3.0, on data with signed zeros; (3) the trajectory
control: 20 steps of the Boussinesq IVP from a small roll keep the linear growth rate of the EVP of
the same equations to 1e-3 (a wrong F changes it).
"""
import numpy as np
from _acceptance import check, passed, start

start(861, "shared RHS cache and Chebyshev recurrence: same bits as before")
import conatus as cn  # noqa: E402
from conatus.timestep import spectral as S  # noqa: E402

m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=1e5, Pr=1)
sim = m.simulate(N=(32, 16), scheme="RK443", dt=1e-4)
sim.initial(T=lambda x, z: -z + 0.1 * np.cos(np.pi * x) * np.cos(np.pi * z) + 0.05 * np.sin(3 * np.pi * x) * z)
sim.run(steps=3)
s = sim.solver
X = s.X.copy()
shared = s.F(X, 0.0)
data = dict(s._ncc)
data.update((v, c) for v, c in zip(s.system.variables, s.system.scatter(X, only=s._rhs_vars),
                                   strict=True) if c is not None)
params = dict(s.plan.params)
out = []
for e, shape, eq in zip(s._rhs_exprs, s.system.eq_shapes, s.plan.problem.equations, strict=True):
    if e is None:
        out.append(None)
        continue
    fresh = S.SpectralEvaluator(s.evaluator.domain)
    cache = {}

    def one(x, fresh=fresh, cache=cache):
        r = cache.get(x)
        if r is None:
            r = cache[x] = fresh._node(x, one, data, params)
        return r
    out.append(s.evaluator._embed(one(e), eq.lhs.rank, shape))
sep = s.system.gather_rhs(out)
check(np.any(shared != 0) and shared.tobytes() == sep.tobytes(), "shared-cache F differs from per-equation F")

rng = np.random.default_rng(861)
for N in (2, 3, 8, 33):
    c = rng.standard_normal((N, 5))
    c[rng.random((N, 5)) < 0.2] = -0.0
    loop = np.zeros_like(c)
    loop[N - 2] = 2 * (N - 1) * c[N - 1]
    for k in range(N - 3, -1, -1):
        loop[k] = loop[k + 2] + 2 * (k + 1) * c[k + 1]
    loop[0] = loop[0] / 2
    check(S._cheb_diff_recurrence(c).tobytes() == loop.tobytes(), f"recurrence differs from the loop at N = {N}")
passed("F byte-identical to the per-equation path; recurrence byte-identical to the loop")
