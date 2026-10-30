"""F751 (audit of 0.2.1, N42): the residual of a scalar neutral mode is defined.

Oracle: s = p - c, neutral at p = c, where L = 0 and s = 0.
"""
from _acceptance import check, passed, start

start(751, "residual of a zero operator")
import numpy as np  # noqa: E402


import conatus.expert as X  # noqa: E402

for c in (0., 1.):
    cs = X.CartesianCoordinates("x")
    D = X.Domain(cs, [])
    u, p, s = X.Field("u", D), X.Parameter("p", 1.), X.Parameter("s", dtype=complex)
    pb = X.EVP([u], eigenvalue=s, name="neutral")
    pb.growth = "real"
    pb.add_equation((s * u - (p - c) * u, 0))
    r = X.critical(X.ParametricEVP(pb, [p]), "p", bracket=(-2., 2.), xtol=1e-12)
    check(abs(r.value - c) < 1e-11 and r.verified, f"{r.value}")
    check(np.isfinite(r.mode.residual) and r.mode.residual <= 1e-12, f"residual {r.mode.residual}")
passed("finite residual")
