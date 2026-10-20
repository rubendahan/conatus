"""F010 (important): an adaptive step whose error exceeds tol is rejected, not taken.

du/dt = -u^2, u(0) = 1, RK222 with dt = 0.5 and tol = 1e-8: the first step's embedded error is
0.32 (3e7 x tol) and 0.1.0 kept it (only the next dt shrank) while the docs say steps are
"rejected or taken". Expected: the step returned by EmbeddedError.step() has error <= tol (the
large step was rejected and retried from the restored state), and u(t) = 1/(1+t) to that order.
A typed refusal of adapt=True is accepted.
"""
from _acceptance import check, passed, refused, start

import conatus as cn

start(10, "adaptive controller rejects steps above tolerance")
from conatus.timestep import EmbeddedError  # noqa: E402

m = cn.Model(cn.Box(x=cn.walls(-1, 1)))
m.scalar("u")
m.equations("dt(u) = -u*u")
s = m.simulate(4, scheme="RK222", dt=0.5).initial(u=1)
was_refused, e = refused(EmbeddedError, s.solver, adapt=True, tol=1e-8, safety=0.9)
if was_refused:
    passed("adapt=True refused")
error = e.step()
t = float(s.t)
u = float(s.grid("u")[0])
exact = 1 / (1 + t)
check(error <= e.tol, f"accepted step with error {error:.3g} = {error / e.tol:.3g} x tol (t = {t})")
check(abs(u - exact) < 1e-6, f"u({t}) = {u}, exact {exact}")
passed(f"accepted error {error:.2g} <= tol, t = {t:.3g}, |u - exact| = {abs(u - exact):.2g}")
