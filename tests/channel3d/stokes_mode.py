"""Manufactured Stokes-resolvent mode for the pressure-treatment comparison (T4):
u = (i kx, ., i kz) v'/k^2, v = cos^2(pi y/2), p = sin(pi y/2) + y^2 on one mode
exp(i(kx x + kz z)); divergence-free, u(+-1) = 0. r = c u - (u'' - k^2 u)/Re + grad p."""
from __future__ import annotations

import numpy as np
import sympy as sp

y = sp.symbols("y")


def exact(kx: float, kz: float, c: float, Re: float):
    k2 = kx * kx + kz * kz
    v = sp.cos(sp.pi * y / 2) ** 2
    u = sp.I * kx * sp.diff(v, y) / k2
    w = sp.I * kz * sp.diff(v, y) / k2
    p = sp.sin(sp.pi * y / 2) + y**2
    vel = [u, v, w]
    grad = [sp.I * kx * p, sp.diff(p, y), sp.I * kz * p]
    r = [c * q - (sp.diff(q, y, 2) - k2 * q) / Re + g for q, g in zip(vel, grad)]
    f = lambda e: sp.lambdify(y, e, "numpy")  # noqa: E731
    return [f(q) for q in vel], f(p), [f(q) for q in r]


def cheb_coeffs(fun, N: int) -> np.ndarray:
    """Chebyshev-T interpolation coefficients on N Gauss-Lobatto-free (first-kind) nodes."""
    from numpy.polynomial import chebyshev as C
    re = C.chebinterpolate(lambda t: np.real(fun(t) + 0 * t), N - 1)
    im = C.chebinterpolate(lambda t: np.imag(fun(t) + 0j * t), N - 1)
    return re + 1j * im


def tau_resolvent(N: int, kx: float, kz: float, c: float, Re: float, assembly: str = "direct"):
    """The same resolvent problem with the plan's generalized tau (ComplexFourier x Cheb x
    ComplexFourier, one excited mode): returns (plan, fields, solution)."""
    from conatus.discretize import discretize
    from conatus.ir import (LBVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain, Field,
                            Parameter, VectorField, div, dot, grad, integ, interp, lap, lift,
                            unit_vector)
    cs = CartesianCoordinates("x", "y", "z")
    x, yc, z = cs.coords
    Lx, Lz = 2 * np.pi / kx, 2 * np.pi / kz
    bx = ComplexFourier(x, 4, (0, Lx), dealias=1.5)
    by = Chebyshev(yc, N, (-1, 1), dealias=1.5)
    bz = ComplexFourier(z, 4, (0, Lz), dealias=1.5)
    D = Domain(cs, [bx, by, bz])
    Dxz, D0 = D.without(yc), Domain(cs, [])
    u, p, f = VectorField("u", D), Field("p", D), VectorField("f", D)
    tp, tu1, tu2 = Field("tp", D0), VectorField("tu1", Dxz), VectorField("tu2", Dxz)
    c_, Re_ = Parameter("c", c), Parameter("Re", Re)
    ey = unit_vector(yc)
    pb = LBVP([u, p, tp, tu1, tu2], name="Stokes resolvent (tau)")
    pb.add_equation((div(u) + dot(ey, lift(tu2, by, -1)) + tp, 0))
    pb.add_equation((c_ * u - lap(u) / Re_ + grad(p) + lift(tu1, by, -1) + lift(tu2, by, -2), f))
    pb.add_equation((interp(u, yc, "left"), 0))
    pb.add_equation((interp(u, yc, "right"), 0))
    pb.add_equation((integ(p), 0))
    plan = discretize(pb, assembly=assembly)
    _, _, r = exact(kx, kz, c, Re)

    def fdata(X, Y, Z):
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        e = np.exp(1j * (kx * X + kz * Z))
        return np.stack([r[i](Y) * e for i in range(3)])
    sol = plan.solve({f: fdata})
    return plan, dict(u=u, p=p, f=f), sol
