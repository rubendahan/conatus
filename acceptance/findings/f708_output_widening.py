"""F708 (audit of 0.2.1, N33): an integer first value does not truncate later floats.

Oracle: the task returns the time, 0 first, so the rows are 0, .1, .2, .3."""
from _acceptance import check, passed, start

start(708, "an integer first value does not truncate later floats")
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402

m = cn.Model(cn.Box(x=cn.periodic(1.)))
m.scalar("u")
m.equations("dt(u)=0")
s = m.simulate(N=4, scheme="RK222", dt=.1)
s.initial(u=1.)
p = Path(tempfile.mkdtemp()) / "c.h5"
s.output(p, tasks={"clock": lambda sim: float(sim.t) if sim.iteration else 0}, every=1, mode="w")
s.run(steps=3)
got = cn.io.read(p).tasks["clock"]
check(np.allclose(got, [0, .1, .2, .3], atol=1e-14, rtol=0), f"rows {got}")
passed("rows kept")
