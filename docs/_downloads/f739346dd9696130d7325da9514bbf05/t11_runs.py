"""Tutorial 11 -- the run registry: record, query, compare, reproduce, and reuse (only when asked).

Heat equation u_t = kappa u_xx on [0, 2 pi), u(x, 0) = sin(3 x): u = exp(-9 kappa t) sin(3 x).
Expected values (checked at the end of the script):
  the measured decay rate of each recorded run equals 9 kappa (closed form) to < 1e-9 relative;
  find(kappa__gt=0.15, status="certified") returns exactly the two certified runs;
  reproduce() gives the recorded end state bit for bit; the cache is used only with cache=True,
  and then returns the recorded end state bit for bit without taking a step.
Run: python t11_runs.py   (a few seconds of CPU)
"""
# [imports]
import os
import tempfile
import warnings

import numpy as np

import conatus as cn

# this tutorial's own registry, in a new temporary directory: nothing of yours is read or deleted
# (your own work: point CN_RUNS at a scratch disk, or give root= to cn.runs.track)
os.environ["CN_RUNS"] = tempfile.mkdtemp(prefix="conatus_t11_registry_")
# [/imports]

# [problem]
T_END, DT = 0.2, 1e-3
NSTEPS = round(T_END / DT)


def simulation(kappa: float):
    """Heat equation with diffusivity kappa (a model parameter: it becomes a run parameter)."""
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat sin(3x)", kappa=kappa)
    m.scalar("u")
    m.equations("dt(u) = kappa*lap(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=DT)
    return sim.initial(u=lambda x: np.sin(3 * x))


def amplitude(sim) -> float:
    """Coefficient of sin(3x) in u (projection on the grid, exact for this resolution)."""
    g = sim.grid("u")
    X = np.linspace(0, 2 * np.pi, g.size, endpoint=False)
    return 2 * float(np.mean(g * np.sin(3 * X)))
# [/problem]

# [record]
runs = {}
for kappa in (0.1, 0.2, 0.4):
    sim = simulation(kappa)
    with cn.runs.track(f"heat_k{kappa}", tags=["tutorial"]) as run:
        sim.solver.run(NSTEPS, record=run)             # numerics, problem hash, start/end state
        rate = -np.log(amplitude(sim)) / sim.t         # measured decay rate
        run.result(rate=rate, rel_error=abs(rate - 9 * kappa) / (9 * kappa))
    runs[kappa] = run
    print(run.explain().splitlines()[0])
for kappa in (0.2, 0.4):                               # certify with the evidence, never bare
    runs[kappa].certify("decay rate = 9 kappa (closed form)",
                        evidence={"rel_error": runs[kappa].results["rel_error"]})
# [/record]

# [query]
found = cn.runs.find(kappa__gt=0.15, status="certified", tags="tutorial", order="created")
print(found.explain())
cmp = cn.runs.compare(runs[0.2], runs[0.4])
print(cmp.explain())
# [/query]

# [reproduce]
rep = runs[0.2].reproduce(simulation(0.2).solver)      # same problem and numerics, checked
print(rep.explain())
# [/reproduce]

# [cache]
plain = simulation(0.4).solver
again = plain.run(NSTEPS, record="heat_k0.4_again")     # identical call, cache off: computed
cached = simulation(0.4).solver
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    hit = cached.run(NSTEPS, record="heat_k0.4_cached", cache=True)
print(caught[0].message)
print({k: hit.cache_report[k] for k in ("hit", "reused_from", "computed")})
print(cn.runs.lineage(again).explain())               # the cached run is a descendant
# [/cache]

# [check]
for kappa, run in runs.items():
    assert run.results["rel_error"] < 1e-9, (kappa, run.results)      # exp(-9 kappa t), closed form
assert [r.id for r in found] == [runs[0.2].id, runs[0.4].id]
assert rep.identical and rep.max_abs_diff == 0.0
assert again.rec["reused_from"] is None and again.cache_report is None
assert hit.cache_report["hit"] and hit.cache_report["reused_from"] in (runs[0.4].id, again.id)
assert np.array_equal(cached.X, plain.X) and cached.iteration == NSTEPS
print("OK: decay rates exact, query, bit-for-bit reproduce, cache only on request")
# [/check]
