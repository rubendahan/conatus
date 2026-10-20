"""F053: NumPy keywords in translated initial data are kept (or the line is unsupported).

u['g'] = np.linspace(0, 1, num=8): 0.1.0 dropped num=8 (50 points) and still counted the line as
translated; building the solver then crashed with a broadcast error. Expected: the initial state
is linspace(0, 1, 8) as the basis holds it (Dedalus drops the Nyquist component of grid data on a
RealFourier basis of even size, and so does Conatus), or the translation is not complete and the
solver build is refused.
"""
import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(53, "from_dedalus keeps NumPy keyword arguments")
SOURCE = """
import numpy as np
import dedalus.public as d3
coords = d3.CartesianCoordinates('x')
dist = d3.Distributor(coords, dtype=np.float64)
basis = d3.RealFourier(coords['x'], 8, bounds=(0, 2*np.pi), dealias=3/2)
u = dist.Field(name='u', bases=basis)
u['g'] = np.linspace(0, 1, num=8)
problem = d3.IVP([u], namespace=locals())
problem.add_equation('dt(u) = 0')
solver = problem.build_solver(d3.RK443)
solver.stop_iteration = 10
while solver.proceed:
    solver.step(0.01)
"""
r = cn.from_dedalus(SOURCE)
was_refused, s = refused(r.ivp_solver)
if was_refused:
    check(not r.complete, "solver build refused while the translation says complete")
    passed(f"incomplete translation (counts {r.counts()}), solver refused")
g = np.asarray(s.grid("u")).ravel()
# what Dedalus itself holds after u['g'] = linspace(0, 1, 8): the real Fourier basis of size 8 has
# no Nyquist mode, so the grid data lose their (-1)^j component (checked against Dedalus 3:
# [0.0714, 0.0714, 0.357, 0.357, ...]); the bug was 50 points, not this projection
data = np.linspace(0, 1, 8)
alt = (-1.0) ** np.arange(8)
expected = data - np.mean(data * alt) * alt
check(g.shape == (8,) and np.allclose(g, expected, rtol=0, atol=1e-12),
      f"initial u = {g}, expected linspace(0, 1, 8) without its Nyquist part {expected}")
passed("initial data = linspace(0, 1, 8) projected on the basis, as Dedalus holds it")
