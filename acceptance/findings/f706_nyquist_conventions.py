"""F706 (audit of 0.2.1, N23): curl at mixed Nyquist modes, dye under axis permutation.

Oracle: the grid curl of grid.d returns u; a swap of the axes swaps the solution."""
from _acceptance import check, passed, start

start(706, "curl at mixed Nyquist modes, dye under axis permutation")
import numpy as np  # noqa: E402

from conatus.diagnostics import Dye, Grid, Periodic  # noqa: E402
from conatus.diagnostics.euler import vector_potential  # noqa: E402

n = 8
g = Grid([Periodic(2 * np.pi, n) for _ in range(3)])
X, Y, _ = np.meshgrid(*[2 * np.pi * np.arange(n) / n] * 3, indexing="ij")
u = [np.cos(4 * X) * np.sin(Y), 0 * X, np.cos(4 * X) * np.cos(Y)]
A, _ = vector_potential(g, u)
curl = [g.d(A[2], 1) - g.d(A[1], 2), g.d(A[0], 2) - g.d(A[2], 0), g.d(A[1], 0) - g.d(A[0], 1)]
err = max(np.max(np.abs(c - w)) for c, w in zip(curl, u, strict=True))
check(err < 1e-12, f"curl A misses u by {err:.2e}")
g2 = Grid([Periodic(2 * np.pi, n), Periodic(2 * np.pi, n)])
X, Y = np.meshgrid(*[2 * np.pi * np.arange(n) / n] * 2, indexing="ij")
c0 = np.cos(4 * X) + np.cos(3 * Y) + .3 * np.cos(4 * X) * np.cos(4 * Y)
ux, uy = .7 + .2 * np.sin(Y), .4 + .2 * np.cos(X)
a, b = Dye(g2, c0, kappa=0., dealias=None), Dye(g2, c0.T, kappa=0., dealias=None)
a.step(.01, [ux, uy], nsteps=5)
b.step(.01, [uy.T, ux.T], nsteps=5)
gap = np.max(np.abs(a.c - b.c.T))
check(gap < 1e-12, f"axis permutation changes the dye by {gap:.2e}")
passed(f"curl {err:.1e}, permutation {gap:.1e}")
