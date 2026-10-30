"""F711 (audit of 0.2.1, N38): a restart refuses another role of the time parameter.

Oracle: the same equation is u_t = t or u_t = 2 by that option; they differ by .34 at t = .4."""
from _acceptance import check, passed, start

start(711, "a restart refuses another role of the time parameter")
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from conatus.errors import OptionError  # noqa: E402
from conatus.expert import IVPSolver, discretize  # noqa: E402
from conatus.ir import CartesianCoordinates, Domain, Field, IVP, Parameter, dt  # noqa: E402
from conatus.ir.bases import RealFourier  # noqa: E402


def clock(tp):
    cs = CartesianCoordinates("x")
    u = Field("u", Domain(cs, [RealFourier(cs["x"], 4, (0., 1.), dealias=1)]))
    pb = IVP([u], name="clock")
    pb.add_equation((dt(u), Parameter("q", 2.)))
    s = IVPSolver(discretize(pb), "RK222", .1, time_parameter=tp)
    s.set_state({u: lambda x: 0 * x}, t=0.)
    return s


a = clock("q")
a.run(2)
p = a.save_checkpoint(Path(tempfile.mkdtemp()) / "c.npz")
try:
    clock(None).load_checkpoint(p)
    check(False, "accepted a checkpoint of another time role")
except OptionError:
    pass
passed("refused")
