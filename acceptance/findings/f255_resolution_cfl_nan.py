"""F255 (audit "assurances", 2026-10-20): the resolution report's CFL of a NaN velocity is not 0.

resolution_report turned every non-finite CFL step into "CFL 0" (a fluid at rest), so a NaN
velocity printed the best possible CFL number.
"""
import numpy as np
from _acceptance import check, passed, start

start(255, "resolution_report CFL of a non-finite velocity")
from conatus.discretize import discretize  # noqa: E402
from conatus.ir import (IVP, CartesianCoordinates, Chebyshev, Domain, Field, dt, interp,  # noqa: E402
                        lap, lift)
from conatus.reference.evaluate import FieldData  # noqa: E402

cs = CartesianCoordinates("x")
x = cs["x"]
b = Chebyshev(x, 16, (-1, 1), dealias=1.5)
D = Domain(cs, [b])
u, t1, t2 = Field("u", D), Field("t1", D.without(x)), Field("t2", D.without(x))
pb = IVP([u, t1, t2], name="heat")
pb.add_equation((dt(u) - lap(u) + lift(t1, b, -1) + lift(t2, b, -2), 0))
pb.add_equation((interp(u, x, "left"), 0))
pb.add_equation((interp(u, x, "right"), 0))
plan = discretize(pb)
rep = plan.resolution_report({u: FieldData(u, np.full(16, np.nan), layout="coeff")},
                             velocity="u", dt=0.1)
check(not rep.cfl["cfl"] == 0.0 and not rep.ok, f"CFL {rep.cfl['cfl']}, ok={rep.ok}")
passed(f"CFL {rep.cfl['cfl']}, report ok={rep.ok}")
