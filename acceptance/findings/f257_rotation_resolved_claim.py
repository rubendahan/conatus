"""F257 (audit "assurances", 2026-10-20): an implicit rotation at |W| dt <= 1 is not "resolved".

Rotating periodic box, implicit Coriolis, SBDF1 at |W| dt = 1: the inertial wave keeps 0.82 of
its amplitude per step (15 % after 10 steps, exact 100 %) while explain() said "its waves are
resolved in time" (a scheme-independent threshold, not a measured accuracy).
"""
import numpy as np
from _acceptance import check, passed, start

start(257, "implicit rotation: no accuracy claim from |W| dt <= 1")
import conatus as cn  # noqa: E402

L = 2 * np.pi
m = cn.Model(cn.Box(x=cn.periodic(L), y=cn.periodic(L), z=cn.periodic(L)), Om=0.5)
m.vector("u")
m.scalar("p")
m.equations("dt(u) + 2*Om*cross(ez, u) + grad(p) = 0", "div(u) = 0")
sim = m.simulate(N=(8, 8, 8), scheme="SBDF1", dt=1.0)
sim.initial(u=lambda x, y, z: np.array([1.0, 1.0, -1.0])[:, None, None, None] * np.sin(x + z))
sim.run(steps=10)
amp = float(np.abs(sim.grid("u")).max()) / 1.0
line = next(ln for ln in sim.explain().splitlines() if "implicit rotation" in ln)
check("resolved in time" not in line, f"amplitude {amp:.2f} of 1 after 10 steps, explain: {line.strip()}")
passed(f"amplitude {amp:.2f} after 10 steps; explain: {line.strip()[:90]}")
