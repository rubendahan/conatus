"""F948 (review H15): small typed-error gaps.

* ``continue_equilibria(direction=<array>)`` raised a bare ValueError: now a typed refusal
  (control: direction=-1.0 runs and the first step lowers the parameter).
* ``floquet_multipliers`` at u = 0 divided by the zero norms of v(u) and du/dx (NaN overlaps,
  RuntimeWarning). Oracle: u_t = a u - u^3 + u_xx linearised at u = 0 has growth a - k^2, so the
  k = 1 multiplier over T = 1 is exp(a - 1), twice (cos and sin).
"""
import warnings

import numpy as np
from _acceptance import check, passed, refused, start

start(948, "typed errors: continuation direction, Floquet overlaps at u = 0")
import conatus as cn  # noqa: E402
import conatus.expert as E  # noqa: E402
from conatus.bifurcation.system import DynamicalSystem  # noqa: E402

ds = DynamicalSystem.from_callables(lambda x, p: p["a"] - x, 1, {"a": 1.0}, fd_x=1e-4, fd_p=1e-4)
s = E.ContinuationSettings(ds=0.1, ds_min=1e-4, ds_max=1.0, tolerance=1e-8, max_points=3)
ok, _ = refused(E.continue_equilibria, ds, np.ones(1), "a", s, direction=np.array([1.0, 0.0]),
                marginal_tol=1e-9)
check(ok, "direction=<array> accepted")
cv = E.continue_equilibria(ds, np.ones(1), "a", s, direction=-1.0, marginal_tol=1e-9)
pars = [float(p.z[-1]) for p in cv.points]
check(len(pars) >= 2 and pars[1] < pars[0], f"control: direction=-1.0 gave parameters {pars}")

from conatus.ecs import floquet_multipliers  # noqa: E402
from conatus.ecs.solutions import RelativePeriodicOrbit  # noqa: E402

a = .5
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), a=a)
m.scalar("u")
m.equations("dt(u)-a*u-dx(dx(u))=-u*u*u")
sim = m.simulate(16, scheme="RK443", dt=.01)
sim.initial(u=lambda x: 0 * x)
sol = RelativePeriodicOrbit(kind="relative_periodic_orbit", X=np.array(sim.solver.X),
                            params={"T": 1.0, "ell": 0.0, "nsteps": 100}, residual=0.,
                            newton=None, energy=0.)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    f = floquet_multipliers(sim, sol, 4, arnoldi_tol=1e-12)
bad = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
check(not bad, f"RuntimeWarning: {bad}")
check(len(f.get("marginal_absent", [])) == 2, f"absent directions not named: {f.get('marginal_absent')}")
mu = np.sort(np.abs(f["multipliers"]))
k1 = np.exp(a - 1.0)
check(np.sum(np.abs(mu - k1) < 1e-6 * k1) == 2, f"multipliers {mu}, oracle exp(a - 1) = {k1} twice")
passed("array direction refused; Floquet at u = 0 without NaN, k = 1 pair matches exp(a - 1)")
