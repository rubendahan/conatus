"""F366: sim.save("state.hdf") must not write a NumPy file named state.hdf.npz (review of the
inputs audit).

Any suffix but .h5/.hdf5 was written as NumPy with .npz appended: "state.hdf" (which reads as
HDF5) became state.hdf.npz. A suffix naming another format is refused with the two formats; a
dotted stem ("state_t1.5") keeps the historical rule; .h5 and .npz are written as asked.
"""
import importlib.util
import os

import numpy as np
from _acceptance import check, passed, refused, start

import conatus as cn

start(366, "checkpoint suffix of another format refused")
m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), nu=0.1)
m.scalar("u")
m.equations("dt(u) - nu*dx(dx(u)) = 0")
s = m.simulate(16, scheme="RK222", dt=0.01).initial(u=lambda x: np.sin(x))
for bad in ("state.hdf", "state.nc", "state.HD5"):
    r, err = refused(s.save, bad)
    check(r, f"save({bad!r}) accepted")
    check(".h5" in str(err), f"save({bad!r}): the message does not name .h5")
check(not os.listdir("."), "a refused save wrote a file")
check(str(s.save("state_t1.5")).endswith("state_t1.5.npz"), "dotted stem not kept")
r, _ = refused(s.save, "grids.h5", grids="no")
check(r, "save(grids='no') accepted")
check(str(s.save("state.npz")).endswith("state.npz"), ".npz not written as asked")
if importlib.util.find_spec("h5py") is not None:      # HDF5 is an optional dependency
    check(str(s.save("state.h5")).endswith("state.h5"), ".h5 not written as asked")
passed("refused with the two formats; .h5, .npz and dotted stems as before")
