"""F301 (new, audit "state" 2026-10-20): conatus.diagnostics.Grid.weights() hands out the grid's
own cached array.

w = grid.weights(1); w *= 2 (rescaling "my" weights) doubled every later grid.integrate(),
grid.mean() and grid.cell_volumes() of that grid; on a 1-D grid cell_volumes() was the cached
array itself. Expected: the integrals are unchanged by any edit of a returned array.
"""
import numpy as np
from _acceptance import check, passed, start

from conatus.diagnostics import Chebyshev, Grid, Periodic

start(301, "editing the returned quadrature weights never changes the grid's integrals")
g = Grid([Periodic(2 * np.pi, 16), Chebyshev(-1, 1, 12)])
f = np.ones(g.shape)
before = float(g.integrate(f))
w = g.weights(1)
w *= 2.0
after = float(g.integrate(f))
check(after == before, f"integral of 1 went from {before} to {after} after editing weights()")
g1 = Grid([Chebyshev(0, 1, 8)])
g1.cell_volumes()[:] = 0.0
one = float(g1.integrate(np.ones(8)))
check(abs(one - 1.0) < 1e-12, f"1-D integral of 1 is {one} after editing cell_volumes()")
passed("weights() and cell_volumes() are the caller's own arrays")
