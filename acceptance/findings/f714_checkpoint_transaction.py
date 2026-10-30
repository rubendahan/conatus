"""F714 (audit of 0.2.1, N43): an incomplete checkpoint is skipped and changes nothing.

Oracle: the state before the failed load, and the complete older file."""
from _acceptance import check, passed, start

start(714, "an incomplete checkpoint is skipped and changes nothing")
import tempfile  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import conatus as cn  # noqa: E402
from conatus.errors import OptionError  # noqa: E402


def build():
    m = cn.Model(cn.Box(x=cn.periodic(1.)), name="f714")
    m.scalar("u")
    m.equations("dt(u) = -u")
    s = m.simulate(4, scheme="SBDF2", dt=.1)
    s.initial(u=2.)
    return s


d = Path(tempfile.mkdtemp())
a = build()
a.run(steps=2)
good = a.save(d / "checkpoint_002.npz")
a.run(steps=1)
bad = a.save(d / "checkpoint_003.npz")
with np.load(bad) as f:
    payload = {k: f[k] for k in f.files if k != "F0"}
np.savez(bad, **payload)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    check(cn.io.latest(d) == good, "latest chose the incomplete file")
s = build()
s.run(steps=1)
X = s.solver.X.copy()
try:
    s.restart(bad)
    check(False, "an incomplete checkpoint was loaded")
except OptionError:
    pass
check(np.array_equal(s.solver.X, X) and s.solver.iteration == 1, "the failed load changed the solver")
passed("skipped and refused")
