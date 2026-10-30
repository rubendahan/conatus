"""F710 (audit of 0.2.1, N37): a derivative through a discarded coefficient is refused.

Oracle: L(p) = -(1 + p) has dL/dp = -1 for every p."""
from _acceptance import check, passed, start

start(710, "a derivative through a discarded coefficient is refused")
from conatus import expert as X  # noqa: E402
from conatus.errors import ParameterError  # noqa: E402

cs = X.CartesianCoordinates("x")
u = X.Field("u", X.Domain(cs, []))
p = X.Parameter("p", 1e-14)
s = X.Parameter("s", dtype=complex)
pb = X.EVP([u], eigenvalue=s, name="f710")
pb.growth = "real"
pb.add_equation((s * u - (1 + p) * u, 0))
P = X.ParametricEVP(pb, [p])
try:
    got = P.derivative("p", p=1e-14)[0][0, 0]
    check(abs(got + 1) < 1e-8, f"dL/dp = {got}, exact -1")
except ParameterError:
    pass
passed("exact or refused")
