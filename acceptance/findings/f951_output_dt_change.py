"""F951 (hunt 5C, M3): an output file records a dt change, and the chain resumes.

Oracle: the step sizes the script sets (0.01 then 0.005). The rows record each one; the file's dt
is the last one. A restart from the checkpoint (dt 0.005) appends to the same file, and the
continued rows equal those of an uninterrupted run bit for bit.
"""
import warnings

import numpy as np
from _acceptance import check, passed, start

start(951, "a dt change is recorded in the output and the chain resumes")
import conatus as cn  # noqa: E402

warnings.simplefilter("ignore")


def mk(dt):
    m = cn.Model(cn.Box(z=cn.walls(0, 1)), name="b")
    m.scalar("u")
    m.equations("dt(u) - dz(dz(u)) = 0")
    m.at("walls", u=0)
    return m.simulate(N=(16,), scheme="RK443", dt=dt)


def first_job(path, ck):
    s = mk(0.01)
    s.initial(u=lambda z: np.sin(np.pi * z))
    s.output(path, ["u"], every=2, mode="w")
    s.run(steps=4)
    s.solver.set_dt(0.005)
    s.run(steps=4)
    if ck:
        s.save(ck)
    return s


s = first_job("a.h5", "ck.npz")
d = cn.io.read("a.h5")
want = np.where(np.asarray(d.iteration) <= 4, 0.01, 0.005)   # set_dt after iteration 4
check(d.dt is not None and np.array_equal(d.dt, want), f"row dt {d.dt}, iterations {d.iteration}")
check(float(d.attrs["dt"]) == 0.005, f"file dt {d.attrs['dt']}")
r = mk(0.005)
r.restart("ck.npz")
r.output("a.h5", ["u"], every=2, mode="a")
r.run(steps=4)
ref = first_job("b.h5", None)
ref.run(steps=4)
a, b = cn.io.read("a.h5"), cn.io.read("b.h5")
check(np.array_equal(a.tasks["u"], b.tasks["u"]) and np.array_equal(a.sim_time, b.sim_time),
      "the resumed rows differ from the uninterrupted run")
passed("rows record dt; the chain resumes into the same file, bit for bit")
