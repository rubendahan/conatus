"""Tutorial 16 -- Lagrangian tracers and a numerical dye: exact velocity at the particles, the
order of the time integration, and a dye whose mass is measured.

1. Decaying Taylor--Green vortices, u = e^{-2 nu t} (sin x cos y, -cos x sin y), an exact solution
   of the Navier--Stokes equations: a tracer follows the streamline psi = sin x sin y at the time
   s(t) = (1 - e^{-2 nu t}) / (2 nu); the reference is that path integrated to 1e-13 (DOP853).
2. Plane Couette flow (walls moving at +-1, u = y): tracers move at their height's speed, tracers
   on a wall move with it; a dye blob is sheared and diffuses, its mass is conserved.
Expected values (checked at the end of the script): the tracer error falls as dt^3 in the decaying
flow (cubic interpolation in time of the flow states; dt^4 in a steady flow); Couette tracers
exact to rounding; dye mass conserved to the level reported (1e-8 bound with walls).
Run: python t16_particles_dye.py   (under a minute of CPU)
"""
# [imports]
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import conatus as cn
# [/imports]

# [taylor_green]
nu, T = 0.5, 2.0


def taylor_green(dt):
    box = cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi))
    m = cn.Model(box, name="decaying Taylor-Green", nu=nu)
    m.velocity("u"), m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
    sim = m.simulate(N=(16, 16), scheme="RK443", dt=dt)
    sim.initial(u=lambda x, y: (np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))
    return sim


X0 = np.random.default_rng(0).uniform(0, 2 * np.pi, (32, 2))
s = (1 - np.exp(-2 * nu * T)) / (2 * nu)


P = len(X0)


def path(_t, Z):
    x, y = Z[:P], Z[P:]
    return np.concatenate([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)])


ref = solve_ivp(path, (0, s), np.concatenate([X0[:, 0], X0[:, 1]]), method="DOP853",
                rtol=1e-13, atol=1e-13).y[:, -1]
ref = np.stack([ref[:len(X0)], ref[len(X0):]], axis=1)
errors = []
for dt in (0.1, 0.05, 0.025):
    sim = taylor_green(dt)
    tracers = sim.particles(X0)
    sim.run(until=T)
    errors.append(np.abs(tracers.positions(unwrapped=True) - ref).max())
    print(f"dt = {dt:<6}: largest tracer error {errors[-1]:.2e}")
orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
print("observed orders:", np.round(orders, 2))
print(tracers.explain())
# [/taylor_green]

# [couette]
box = cn.Box(x=cn.periodic(4.0), y=cn.walls(-1, 1))
m = cn.Model(box, name="plane Couette with dye", nu=0.1)
m.velocity("u"), m.pressure("p")
m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)", "div(u) = 0")
m.at("bottom", u=(-1, 0))
m.at("top", u=(1, 0))
m.dye("c", velocity="u", kappa=0.01)      # dt(c) - 0.01*lap(c) = -div(c*u), dy(c) = 0 at walls
print(m.explain())
sim = m.simulate(N=(32, 32), scheme="RK443", dt=0.01)
sim.initial(u=lambda x, y: (y, 0 * x), c=lambda x, y: np.exp(-((x - 2) ** 2 + y ** 2) / 0.3))
out = Path(tempfile.mkdtemp())
starts = np.array([[0.0, -1.0], [1.0, -0.5], [2.0, 0.0], [3.0, 0.5], [0.5, 1.0]])
tracers = sim.particles(starts, path=out / "tracers.h5", every=50)
dye = sim.dye("c", out / "dye.h5", every=50)
sim.run(until=3.0)
print(dye.explain())
data = cn.io.read(out / "tracers.h5").to_xarray()
print(data["position_unwrapped"].isel(t=-1).values)
mass = cn.io.read(out / "dye.h5").tasks["mass"]
drift = np.abs(mass - mass[0]).max() / mass[0]
print(f"dye mass {mass[0]:.12f}, largest relative drift {drift:.1e}")
# [/couette]

# [checks]
assert np.all(orders > 2.6), orders                   # third order in an unsteady flow
exact = starts + np.column_stack([3.0 * starts[:, 1], 0 * starts[:, 1]])
assert np.abs(tracers.positions(unwrapped=True) - exact).max() < 1e-12
assert drift < 1e-8
print("checks passed: order 3 in time, Couette tracers exact, dye mass conserved")
# [/checks]
