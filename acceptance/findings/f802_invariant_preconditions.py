"""F802 (external review of 0.2.3, finding 3): finders refuse equations without the symmetry they assume.

Oracle: u_t + u = cos x + sin x, u(x, 0) = cos x has the exact solution
cos x + (1 - exp(-t)) sin x, which differs from cos(x - t) by 0.50 at t = 1, so the state with a
zero comoving residual is no wave. Control: the viscous Burgers-type model without forcing, on
which a translate of a state is checked exactly and the check does not refuse.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(802, "invariant-solution finders check their symmetry preconditions")
import conatus as cn  # noqa: E402
import conatus.expert as C  # noqa: E402
from conatus.errors import SymmetryError  # noqa: E402
from conatus.ecs.equivariance import equivariance_defect  # noqa: E402


def sim_of(eq, given=None):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    if given:
        m.given("f", given)
    m.equations(eq)
    s = m.simulate(16, scheme="RK443", dt=0.01)
    s.initial(u=lambda x: np.cos(x))
    return s


def no(call):
    hit, exc = refused(call)
    return hit and isinstance(exc, SymmetryError)


N = C.NewtonKrylovHookstep(tol=1e-12)
forced = sim_of("dt(u)+u=f", lambda x: np.cos(x) + np.sin(x))
check(no(lambda: C.find_travelling_wave(forced, forced, 1.0, N)),
      "forced wave not refused")
coeff = sim_of("dt(u)=explicit(-f*u)", lambda x: 2 + np.cos(x))
check(no(lambda: C.find_travelling_wave(coeff, coeff, 1.0, N)),
      "x-dependent coefficient not refused")
check(no(lambda: C.find_relative_periodic_orbit(forced, forced, 1.0, 0.1, N, steps=20)),
      "forced RPO not refused")
timed = sim_of("dt(u)+u=cos(t)")
check(refused(lambda: C.find_travelling_wave(timed, timed, 1.0, N))[0],
      "time forcing not refused by the wave finder")
odd = sim_of("dt(u)+u=cos(t)")
odd.initial(u=lambda x: np.sin(x))
check(refused(lambda: C.find_equilibrium(odd, odd, N))[0],
      "time forcing not refused by the equilibrium finder")
# the exact solution of the forced problem agrees with the integrator, not with the wave
forced.run(steps=100)
x = forced.coordinate("x")
exact = np.cos(x) + (1 - np.exp(-1)) * np.sin(x)
check(np.max(np.abs(forced.grid("u") - exact)) < 1e-6, "trajectory control")
check(np.max(np.abs(np.cos(x - 1) - exact)) > 0.5, "claimed wave control")
# control: an equivariant nonlinear problem is not refused, its defect is at rounding level
free = sim_of("dt(u)-dx(dx(u))=-u*dx(u)")
from conatus.ecs.fourier_ops import FourierOps  # noqa: E402
ops = FourierOps(free.solver)
d = equivariance_defect(free.solver, lambda X: ops.translate(X, 2 * np.pi / 16), np.array(free.solver.X))
check(d < 1e-12, f"control defect {d:.1e}")
# A rough seed of an equivariant problem is not refused. On a 3/2 grid the aliases of a cubic or
# quintic product commute only with a shift of that grid, whose cell is 2 pi / 48, not 2 pi / 32.
rough = lambda x: np.cos(x) + 0.3 * sum(np.cos(k * x + k) for k in range(2, 16))  # noqa: E731
for eq in ("dt(u)-dx(dx(u))-u=-u*u*u", "dt(u)-dx(dx(u))=-u*u*u*u*u"):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
    m.scalar("u")
    m.equations(eq)
    s = m.simulate(32, scheme="RK443", dt=0.01, dealias=1.5)
    s.initial(u=rough)
    from conatus.ecs.solutions import _require_relative  # noqa: E402
    seed = np.array(s.solver.X)
    hit, exc = refused(lambda s=s, seed=seed: _require_relative(s.solver, FourierOps(s.solver), seed, "f802"))
    check(not hit, f"{eq}, rough seed on a 3/2 grid: false refusal {exc}")
passed("forcing, coefficient and time dependence refused; equivariant control passes")
