"""F732 (audit of 0.2.1, N13): multiplying M and G by one constant leaves the equation unchanged.

Oracle: x_dot = (mu-1) x - x^3 at mu = 0 has eigenvalue -1 (no bifurcation); x_dot = mu + x^2 is a fold at 0; x_dot = (mu-1) x - x^3 at mu = 1 is a pitchfork.
"""
from _acceptance import check, passed, start

start(732, "normal form independent of the equation scale")
import numpy as np  # noqa: E402


from conatus.bifurcation import ir_system, normal_form  # noqa: E402
from conatus.errors import ConvergenceError  # noqa: E402
from conatus.ir import CartesianCoordinates, Domain, Field, IVP, Parameter, dt  # noqa: E402
from conatus.ir.bases import RealFourier  # noqa: E402


def build(c, rhs, mu0):
    cs = CartesianCoordinates("dummy")
    D = Domain(cs, [RealFourier(cs["dummy"], 2, (0, 1), dealias=1)])
    u, mu = Field("u", D), Parameter("mu", mu0)
    pb = IVP([u])
    pb.add_equation((c * dt(u), c * rhs(u, mu)))
    return ir_system(pb, ["mu"])


ref = None
for c in (1.0, 1e4, 1e8, 1e12):
    s = build(c, lambda u, mu: (mu - 1) * u - u ** 3, 0.)
    try:
        normal_form(s, np.zeros(s.n), kind="zero", parameter="mu")
        check(False, f"scale {c:g}: eigenvalue -1 admitted as critical")
    except ConvergenceError:
        pass
    s = build(c, lambda u, mu: mu + u ** 2, 0.)
    nf = normal_form(s, np.zeros(s.n), kind="zero", parameter="mu")
    check(nf.kind == "fold", f"scale {c:g}: fold called {nf.kind}")
    s = build(c, lambda u, mu: (mu - 1) * u - u ** 3, 1.)
    nf = normal_form(s, np.zeros(s.n), kind="zero", parameter="mu")
    check(nf.kind == "pitchfork" and nf.a3 < 0, f"scale {c:g}: critical point called {nf.kind}")
    ref = ref if c > 1 else nf.amplitude(.01)
    check(abs(nf.amplitude(.01) - ref) < 1e-9 * ref, f"scale {c:g}: amplitude {nf.amplitude(.01)} against {ref}")
passed("same decision at scales 1 .. 1e12")
