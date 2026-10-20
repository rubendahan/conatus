"""F354: sim.cfl reads the velocity, not its first grid value (audit inputs, 2026-10-20).

On a 1D problem (Burgers) the scalar u was indexed as a vector: u[0], the first grid value, was
taken as the velocity of the whole domain (u = sin(x): u[0] = 0, an infinite step). On a 2D
problem a scalar's first grid rows were read as components; an unknown name was a bare
ValueError; safety="0.5" was parsed.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(354, "CFL velocity read on the whole grid; a 2D scalar refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = -u*dx(u)")
s = m.simulate(16, scheme="RK222", dt=0.01).initial(u=lambda x: np.sin(x))
step = s.cfl("u", safety=0.5)
want = 0.5 * (2 * np.pi / 16) / float(np.abs(s.grid("u")).max())
check(abs(step - want) < 1e-12, f"1D Burgers CFL {step} (from the whole grid: {want})")
r, _ = refused(s.cfl, "w", safety=0.5)
check(r, "cfl('w') accepted")
m3 = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi)), nu=0.1)
m3.scalar("T")
m3.equations("dt(T) - nu*lap(T) = 0")
s3 = m3.simulate((8, 8), scheme="RK222", dt=0.01).initial(T=lambda x, y: np.sin(x) + 0 * y)
r, _ = refused(s3.cfl, "T", safety=0.5)
check(r, "cfl of a 2D scalar accepted")
m2 = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m2.velocity("v")
m2.equations("dt(v) = 0")
s2 = m2.simulate(16, scheme="RK222", dt=0.01).initial(v=lambda x: (2.0 + 0 * x,))
r, _ = refused(s2.cfl, "v", safety="0.5")
check(r, "safety='0.5' accepted")
dt = s2.cfl("v", safety=0.5)
check(abs(dt - 0.5 * (2 * np.pi / 16) / 2.0) < 1e-12, f"cfl of u = 2: {dt}")
passed("1D step from the whole grid; 2D scalar, unknown name and text safety refused")
