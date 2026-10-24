"""F601 (audit of 0.1.2, F01): invariant-solution searches keep the seed's mean and refuse a
result whose full equations do not hold.

Oracles: 1 + cos x is an exact travelling wave of u_t + u_x = 0 (speed 1, mean 1); u_t = 1 has
no equilibrium (the mean row of its residual is 1)."""
from _acceptance import check, passed, refused, start

start(601, "find_travelling_wave keeps the mean; a non-solution is refused")
import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.ecs import NewtonKrylovHookstep, find_travelling_wave  # noqa: E402


def sim(eq, u0):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations(eq)
    s = m.simulate(N=16, scheme="RK443", dt=.01)
    s.initial(u=u0)
    return s


s = sim("dt(u)+dx(u)=0", lambda x: 1 + np.cos(x))
x = s.solver.X.copy()
sol = find_travelling_wave(s, x, 1.0, NewtonKrylovHookstep(tol=1e-10))
err = float(np.abs(sol.X - x).max())
check(err < 1e-12, f"returned wave differs from the exact wave 1 + cos x by {err:.2e}")
check(abs(sol.params["c"] - 1) < 1e-10, f"speed {sol.params['c']} instead of 1")
s = sim("dt(u)=1", lambda x: np.cos(x))
was_refused, r = refused(find_travelling_wave, s, s.solver.X.copy(), 0.0,
                         NewtonKrylovHookstep(tol=1e-10))
check(was_refused, f"u_t = 1 returned {getattr(r, 'kind', r)!r}, which is not a solution")
passed("wave 1 + cos x returned with its mean; u_t = 1 refused")
