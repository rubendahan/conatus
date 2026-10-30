"""F753 (audit of 0.2.1, N52): products of different harmonics are not affine in the base flow.

Oracle: u_t = U V u with U, V at harmonics 1, 2: G(T) = exp(2 I(T)) with I from the integral of the product, refused or solved; U + V stays affine, gain from the integral.
"""
from _acceptance import check, passed, start

start(753, "mixed base-flow harmonics")
import numpy as np  # noqa: E402


from conatus.errors import EquationError  # noqa: E402
from conatus.growth import operator_gain, periodic_operator  # noqa: E402
import conatus.expert as X  # noqa: E402


def problem(form, N=8):
    cs = X.CartesianCoordinates("x")
    D = X.Domain(cs, [X.Chebyshev(cs["x"], N, (-1, 1), dealias=1)])
    u, U, V = [X.Field(n, D) for n in ("u", "U", "V")]
    pb = X.IVP([u], name="two harmonic coefficients")
    if form == "mass":
        pb.add_equation(((1 + .25 * U * V) * X.dt(u) + u, 0))
    else:
        pb.add_equation((X.dt(u) - (U * V if form == "product" else U + V) * u, 0))
    return pb


half = lambda x: 0 * x + .5  # noqa: E731
base = {"U": {1: half, -1: half}, "V": {2: half, -2: half}}
for form in ("product", "mass"):
    try:
        periodic_operator(lambda p, f=form: problem(f), base, 0., 0., fields=("u",), period=1.)
        check(False, f"{form}: accepted")
    except EquationError:
        pass
op = periodic_operator(lambda p: problem("sum"), base, 0., 0., fields=("u",), period=1.)
T = .125
ref = np.exp(2 * (np.sin(2 * np.pi * T) / (2 * np.pi) + np.sin(4 * np.pi * T) / (4 * np.pi)))
G = operator_gain(op, 0., T, 160, "RK443").G
check(abs(G - ref) < 1e-6 * ref, f"affine control {G} against {ref}")
passed("refused; affine control exact")
