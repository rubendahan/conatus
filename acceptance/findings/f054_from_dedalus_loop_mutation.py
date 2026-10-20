"""F054: a state mutation inside the Dedalus time loop is not silently dropped.

The loop body does u['g'] += 1 before each step: 0.1.0 translated the loop to solver.run(n),
ignored the mutation and still reported complete=True. Expected: complete is False (the mutation
is reported unsupported), or the translated run applies it.
"""
from _acceptance import check, passed, refused, start

import conatus as cn

start(54, "from_dedalus reports physical mutations in the loop")
SOURCE = """
import numpy as np
import dedalus.public as d3
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
basis = d3.RealFourier(coords['x'], 8, bounds=(0, 2*np.pi), dealias=3/2)
u = dist.Field(name='u', bases=basis)
u['g'] = 1
problem = d3.IVP([u], namespace=locals())
problem.add_equation('dt(u) = 0')
solver = problem.build_solver(d3.RK443)
solver.stop_iteration = 10
while solver.proceed:
    u['g'] += 1
    solver.step(0.01)
"""
was_refused, r = refused(cn.from_dedalus, SOURCE)
if was_refused:
    passed("translation refused")
if not r.complete:
    passed(f"incomplete: counts {r.counts()}")
s = r.ivp_solver()
s.run(r.steps())
u = float(s.grid("u").mean())
check(abs(u - 11.0) < 1e-12, f"complete=True but the final u = {u} (Dedalus: 11)")
passed("mutation applied")
