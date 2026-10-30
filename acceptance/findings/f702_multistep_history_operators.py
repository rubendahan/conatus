"""F702 (audit of 0.2.1, N20): a restart with another mass does not reuse its history.

Oracle: c u_t = 0 keeps u = 1 for every c."""
from _acceptance import check, passed, start

start(702, "a restart with another mass does not reuse its history")
import tempfile  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from conatus.errors import OptionError  # noqa: E402
from conatus.expert import IVPSolver, discretize  # noqa: E402
from conatus.ir import CartesianCoordinates, Domain, Field, IVP, Parameter, dt  # noqa: E402
from conatus.ir.bases import RealFourier  # noqa: E402


def build(c, order=(0, 1)):
    cs = CartesianCoordinates("x")
    dom = Domain(cs, [RealFourier(cs["x"], 4, (0., 1.), dealias=1)])
    u, v = Field("u", dom), Field("v", dom)
    C = Parameter("c", c)
    pb = IVP([u, v], name="f702")
    eqs = [(C * dt(u), 0), (dt(v) + v, 0)]
    for k in order:
        pb.add_equation(eqs[k])
    s = IVPSolver(discretize(pb), "SBDF2", .01, startup="rk443")
    s.set_state({u: lambda x: 1. + 0 * x, v: lambda x: 0 * x})
    return s


d = Path(tempfile.mkdtemp())
a = build(1.)
a.run(4)
p = a.save_checkpoint(d / "c.npz")
b = build(2.)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    b.load_checkpoint(p, allow=("c",))
b.run(4)
check(np.max(np.abs(b.X - a.X)) < 1e-12, "the constant state moved after the mass changed")
r = build(1., (1, 0))
try:
    r.load_checkpoint(p)
    check(False, "a reordered system was accepted with the history of the old rows")
except OptionError:
    pass
passed("history discarded with the mass, reordered rows refused")
