"""NLBVP test cases shared by tests/nlbvp and the Dedalus parity test."""
import numpy as np

from conatus.ir import CartesianCoordinates, Chebyshev, Domain, Field, NLBVP, diff, interp, lift, power
from conatus.solvers.nlbvp import NewtonSolver

XI1_N3 = 6.896848619376960     # first zero of the n=3 Lane-Emden function (Horedt 1986; 6.89684862 to 8 digits)


def lane_emden(n, N):
    """r f'' + 2 f' + R^2 r f^n = 0 on r in [0, 1] (the radial Laplacian multiplied by r).

    R = xi_1 is an unknown scalar; f(0) = 1, f(1) = 0 and the regularity condition f'(0) = 0
    close the system: no 1/r, no 1/(r + eps). Same formulation as Dedalus' Lane-Emden example,
    written in 1D Chebyshev.
    """
    cs = CartesianCoordinates("r")
    r = cs["r"]
    b = Chebyshev(r, N, (0, 1), dealias=1)
    D = Domain(cs, [b])
    D0 = D.without(r)
    f, R, t1, t2, rr = Field("f", D), Field("R", D0), Field("t1", D0), Field("t2", D0), Field("rr", D)
    pb = NLBVP([f, R, t1, t2], name=f"Lane-Emden n={n}")
    pb.add_equation((rr * diff(diff(f, r), r) + 2 * diff(f, r) + R * R * rr * power(f, n)
                     + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(f, r, 0), 1))
    pb.add_equation((interp(f, r, 1), 0))
    pb.add_equation((interp(diff(f, r), r, 0), 0))
    return pb, dict(f=f, R=R, rr=rr, basis=b)


def newton(pb, rr, **kw):
    kw = {"tolerance": 1e-13, "damping": 1.0, "max_iterations": 30, **kw}
    return NewtonSolver(pb, ncc={rr: lambda r: r}, **kw)


def lane_emden_n3(N, **kw):
    """n = 1 from cos(pi r / 2), R = 3, then continuation n = 1 -> 2 -> 3."""
    guess = None
    sols = []
    for n in (1, 2, 3):
        pb, o = lane_emden(n, N)
        g = guess or {o["f"]: lambda r: np.cos(np.pi * r / 2), o["R"]: lambda r: 3.0 + 0 * r}
        sol = newton(pb, o["rr"], **kw).solve(g)
        sols.append(sol)
        guess = sol.as_guess()
    return sols[-1], sols


def cubic(N):
    """u'' = 2 u^3 on [0, 1], u(0) = 1, u(1) = 1/2: exact solution u = 1 / (1 + x)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (0, 1), dealias=1)
    D = Domain(cs, [b])
    D0 = D.without(x)
    u, t1, t2 = Field("u", D), Field("t1", D0), Field("t2", D0)
    pb = NLBVP([u, t1, t2], name="u'' = 2u^3")
    pb.add_equation((diff(diff(u, x), x) + lift(t1, b, -1) + lift(t2, b, -2), 2 * power(u, 3)))
    pb.add_equation((interp(u, x, 0), 1))
    pb.add_equation((interp(u, x, 1), 0.5))
    return pb, dict(u=u, basis=b)


# Bratu on [-1, 1]: u'' + lam e^u = 0, u(+-1) = 0. Closed form u = 2 ln(cosh a / cosh(a x)) with
# lam = 2 a^2 / cosh(a)^2; the fold is at a tanh a = 1 (a* = 1.19967864...), i.e.
# lam_c = 0.87845767978129... On [0, 1] the same problem has lam_c = 4 * 0.8784... = 3.51383071912516
# (the classical value, e.g. Jacobsen & Schmitt, J. Differ. Equ. 184 (2002)); both follow from the
# closed form above, computed here rather than copied.
def bratu_a_star():
    from scipy.optimize import brentq
    return brentq(lambda a: a * np.tanh(a) - 1.0, 0.5, 2.0, xtol=1e-15)


def bratu_lambda_c():
    a = bratu_a_star()
    return 2 * a * a / np.cosh(a) ** 2


def bratu_exact(lam, x, branch="lower"):
    """Closed-form Bratu solution on [-1, 1] on the lower (a < a*) or upper (a > a*) branch."""
    from scipy.optimize import brentq
    a_s = bratu_a_star()
    g = lambda a: 2 * a * a / np.cosh(a) ** 2 - lam  # noqa: E731
    a = brentq(g, 1e-12, a_s, xtol=1e-15) if branch == "lower" else brentq(g, a_s, 50.0, xtol=1e-15)
    return 2 * np.log(np.cosh(a) / np.cosh(a * x)), a


def bratu(N, lam_value):
    from conatus.ir import Parameter, exp
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    D0 = D.without(x)
    u, t1, t2 = Field("u", D), Field("t1", D0), Field("t2", D0)
    lam = Parameter("lam", lam_value)
    pb = NLBVP([u, t1, t2], name="Bratu")
    pb.add_equation((diff(diff(u, x), x) + lam * exp(u) + lift(t1, b, -1) + lift(t2, b, -2), 0))
    pb.add_equation((interp(u, x, -1), 0))
    pb.add_equation((interp(u, x, 1), 0))
    return pb, dict(u=u, lam=lam, basis=b)
