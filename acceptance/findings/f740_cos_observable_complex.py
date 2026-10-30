"""F740 (audit of 0.2.1, N21): the cosine observable is complex-linear.

Oracle: l(a + i b) = l(a) + i l(b), with l(a), l(b) on real vectors; u_t = mu u - v, v_t = u + mu v whose Hopf vector has its v part in quadrature with its u part.
"""
from _acceptance import check, passed, start

start(740, "cosine observable keeps the imaginary quadrature")
import numpy as np  # noqa: E402


from conatus.bifurcation import ir_system  # noqa: E402
from conatus.bifurcation.amplitude import cos_mode_observable  # noqa: E402
from conatus.ir import CartesianCoordinates, Domain, Field, IVP, Parameter, dt  # noqa: E402
from conatus.ir.bases import RealFourier  # noqa: E402

cs = CartesianCoordinates("x")
D = Domain(cs, [RealFourier(cs["x"], 8, (0, 2 * np.pi), dealias=1)])
u, v, mu = Field("u", D), Field("v", D), Parameter("mu", 0.)
pb = IVP([u, v])
pb.add_equation((dt(u), mu * u - v))
pb.add_equation((dt(v), u + mu * v))
s = ir_system(pb, ["mu"])
ell = cos_mode_observable(s, "v")
rng = np.random.default_rng(0)
a, b = rng.standard_normal(s.n), rng.standard_normal(s.n)
check(abs(ell(a + 1j * b) - complex(ell(a), ell(b))) < 1e-13, "not complex-linear")
check(abs(ell(1j * b) - 1j * ell(b)) < 1e-13 and ell(b) != 0, "the quadrature is dropped")
passed("l(a + i b) = l(a) + i l(b)")
