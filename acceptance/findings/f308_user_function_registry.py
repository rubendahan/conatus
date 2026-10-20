"""F308 (new, review of the audit "state" 2026-10-20, F5): the user-function registry keeps
every declaration for the life of the process.

conatus.ir.pointwise (and m.function) put each declaration in a process-wide registry that
was never emptied: a notebook cell run again with a new lambda added an entry and kept the
closure, and every array it captured, alive. Expected: a declaration nothing uses any more is
released; one still used (a model, a node) keeps working.
"""
import gc
import weakref

import numpy as np
from _acceptance import check, passed, start

import conatus as cn
from conatus.ir import pointwise

start(308, "user-function declarations nothing uses are released")
refs = []
for _ in range(20):                                   # a cell run 20 times
    big = np.zeros(1_000_000)
    pointwise("cell_g", lambda u, big=big: u + big[0])
    refs.append(weakref.ref(big))
    del big
gc.collect()
alive = sum(r() is not None for r in refs)
check(alive == 0, f"{alive} of 20 captured 8 MB arrays still alive after the cell's builders were dropped")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)))
m.scalar("u")
m.function("decay", lambda u: -u)
m.equations("dt(u) = decay(u)")
gc.collect()
s = m.simulate(8, scheme="RK443", dt=1e-3).initial(u=1)
s.run(steps=10)
err = abs(float(s.grid("u")[0]) - np.exp(-s.t))
check(err < 1e-10, f"a model's function no longer works after a collection (error {err:.1e})")
passed("dropped declarations released, used ones kept")
