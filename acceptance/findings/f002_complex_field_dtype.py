"""F002 (blocking): an explicitly complex field must keep its imaginary part.

du/dt = 1j from u(0) = 1j over one step of 0.1 gives u = 1.1j everywhere. In 0.1.0 the system
was float64 and the result zeros (a NumPy ComplexWarning only). A typed refusal is accepted.
"""
import numpy as np
from _acceptance import check, passed, refused, start

start(2, "a complex field must not be cast to real")
from conatus.discretize import discretize  # noqa: E402
from conatus.ir import IVP, CartesianCoordinates, Chebyshev, Domain, Field, dt  # noqa: E402
from conatus.timestep import IVPSolver  # noqa: E402


def build_and_run():
    cs = CartesianCoordinates("x")
    b = Chebyshev(cs["x"], 8, (-1, 1), dealias=1)
    u = Field("u", Domain(cs, [b]), dtype=complex)
    pb = IVP([u])
    pb.add_equation((dt(u), 1j))
    s = IVPSolver(discretize(pb), "RK222", 0.1)
    s.set_state({u: lambda x: 1j + 0 * x})
    s.run(1)
    return s.grid("u")


was_refused, g = refused(build_and_run)
if was_refused:
    passed("complex field refused")
g = np.asarray(g)
err = float(np.max(np.abs(g - 1.1j)))
check(err < 1e-12, f"u = {g.ravel()[:3]}..., expected 1.1j (max error {err:.3g})")
passed(f"u = 1.1j (max error {err:.2g})")
