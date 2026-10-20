"""F309 (new, review of the audit "state" 2026-10-20, F4): PipeTravellingWave.phi hands out its
cached arrays.

phi(x) returned the arrays it caches for the last x, and the travelling-wave result keeps the
same full state: an in-place edit of the returned (or stored) state changed what a later phi of
the same x returned. Expected: phi returns the caller's own copies.
"""
import numpy as np
from _acceptance import check, passed, start

from conatus.ecs.pipe import PipeTravellingWave

start(309, "PipeTravellingWave.phi returns the caller's own arrays")
p = PipeTravellingWave.__new__(PipeTravellingWave)      # the cache logic alone (no pipe solve)
p._cache, p.nsteps = {}, 1
p.advance = lambda X, n: 2.0 * X
p.to_X = lambda x: x.copy()
p.to_x = lambda X: X[:2].copy()
x = np.array([1.0, 2.0, 3.0])
y, Y = p.phi(x)
y *= 0.0
Y *= 0.0
y2, Y2 = p.phi(x)
check(np.array_equal(y2, [2.0, 4.0]) and np.array_equal(Y2, [2.0, 4.0, 6.0]),
      f"a later phi of the same x returned the edited arrays: {y2}, {Y2}")
passed("phi hands out copies")
