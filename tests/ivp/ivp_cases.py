"""IVP test problems written through the IR."""
from __future__ import annotations

import numpy as np

from conatus.discretize import discretize
from conatus.ir import (EVEN, IVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain,
                        Field, Parameter, RealFourier, Reflection, SymmetrySubspace, VectorField,
                        diff, div, dot, dt, grad, integ, interp, lap, lift, unit_vector)


def heat_1d(N: int = 24, a: float = 1.0):
    """u_t - u_xx = -a u on (-1, 1), u(+-1) = 0 (explicit reaction exercises F).
    Exact: u = exp(-(pi^2 + a) t) sin(pi x)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = Chebyshev(x, N, (-1, 1), dealias=1.5)
    D = Domain(cs, [b])
    u = Field("u", D)
    t1, t2 = Field("t1", D.without(x)), Field("t2", D.without(x))
    pb = IVP([u, t1, t2], name="heat 1D")
    pb.add_equation((dt(u) - lap(u) + lift(t1, b, -1) + lift(t2, b, -2), -a * u))
    pb.add_equation((interp(u, x, "left"), 0))
    pb.add_equation((interp(u, x, "right"), 0))
    exact = lambda X, t: np.exp(-(np.pi ** 2 + a) * t) * np.sin(np.pi * X)  # noqa: E731
    return discretize(pb), u, exact


def burgers_1d(N: int = 64, nu: float = 0.1, A: float = 0.5, basis: str = "real"):
    """u_t - nu u_xx = -u u_x on (0, 2 pi). Cole–Hopf: phi = 1 + A e^{-nu t} cos x,
    u = -2 nu phi_x / phi = 2 nu A e^{-nu t} sin x / (1 + A e^{-nu t} cos x)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    B = RealFourier if basis == "real" else ComplexFourier
    b = B(x, N, (0, 2 * np.pi), dealias=1.5)
    D = Domain(cs, [b])
    u = Field("u", D, dtype=float if basis == "real" else complex)
    pb = IVP([u], name="Burgers 1D")
    pb.add_equation((dt(u) - nu * lap(u), -u * diff(u, x)))

    def exact(X, t):
        e = A * np.exp(-nu * t)
        return 2 * nu * e * np.sin(X) / (1 + e * np.cos(X))
    return discretize(pb), u, exact


def ks_1d(N: int = 128, L: float = 32 * np.pi):
    """Kuramoto–Sivashinsky u_t + u_xx + u_xxxx = -u u_x on (0, L)."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    b = RealFourier(x, N, (0, L), dealias=1.5)
    D = Domain(cs, [b])
    u = Field("u", D)
    pb = IVP([u], name="Kuramoto-Sivashinsky")
    pb.add_equation((dt(u) + lap(u) + lap(lap(u)), -u * diff(u, x)))
    return discretize(pb), u


def rayleigh_benard_2d(Nx: int, Nz: int, Ra: float, Pr: float = 1.0, Lx: float = 2.0,
                       symmetric: bool = False, dealias: float = 1.5):
    """Boussinesq RB in thermal-diffusion units, as the EVP case (tests/evp/cases.py):
    z in (-1/2, 1/2), rigid walls, T = +-1/2 on the walls,
    dt(u) - Pr lap u + grad p - Pr Ra T ez = -u.grad u,  dt(T) - lap T = -u.grad T."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=dealias)
    bz = Chebyshev(z, Nz, (-0.5, 0.5), dealias=dealias)
    D = Domain(cs, [bx, bz])
    Dx = D.without(z)
    u, T, p = VectorField("u", D), Field("T", D), Field("p", D)
    tp = Field("tp", Domain(cs, []))
    tT1, tT2 = Field("tT1", Dx), Field("tT2", Dx)
    tu1, tu2 = VectorField("tu1", Dx), VectorField("tu2", Dx)
    Ra_, Pr_ = Parameter("Ra", Ra), Parameter("Pr", Pr)
    ez = unit_vector(z)
    G = None
    if symmetric:     # x -> -x about x = 0: u_x odd (vector rule), u_z, T, p even
        G = SymmetrySubspace(Reflection(x, about=0), u=EVEN, T=EVEN, p=EVEN, tp=EVEN,
                             tT1=EVEN, tT2=EVEN, tu1=EVEN, tu2=EVEN)
    pb = IVP([u, T, p, tp, tT1, tT2, tu1, tu2], name="Rayleigh-Benard 2D", symmetry=G)
    # Generalized tau (lifts into T_{N-1}, T_{N-2}): momentum row N-1 carries tu1, and the
    # top continuity row (empty in the k = 0 group with a second-order formulation) carries
    # ez.tu2, as in the Dedalus first-order formulation trace(grad_u) + tau_p; tp = gauge.
    pb.add_equation((div(u) + dot(ez, lift(tu2, bz, -1)) + tp, 0))
    pb.add_equation((dt(T) - lap(T) + lift(tT1, bz, -1) + lift(tT2, bz, -2), -dot(u, grad(T))))
    pb.add_equation((dt(u) - Pr_ * lap(u) + grad(p) - Pr_ * Ra_ * T * ez
                     + lift(tu1, bz, -1) + lift(tu2, bz, -2), -dot(u, grad(u))))
    pb.add_equation((interp(T, z, "left"), 0.5))
    pb.add_equation((interp(T, z, "right"), -0.5))
    pb.add_equation((interp(u, z, "left"), 0))
    pb.add_equation((interp(u, z, "right"), 0))
    pb.add_equation((integ(p), 0))
    return discretize(pb), dict(u=u, T=T, p=p, x=x, z=z)


def rayleigh_benard_2d_psi_theta(Nx: int, Nz: int, Ra: float, Pr: float = 1.0, Lx: float = 2.0,
                                 dealias: float = 1.5, linear: bool = False):
    """Streamfunction-temperature RB, same formulation as the hand-written comparator
    (benchmarks/specialized/rb2d_cuda/README.md): thermal-diffusion units, z in (0, 1),
    u = psi_z, w = -psi_x, omega = lap psi, T = 1 - z + theta, rigid isothermal walls:
      dt(lap psi) - Pr lap lap psi + Ra Pr theta_x = -(u omega_x + w omega_z),
      dt(theta) - lap theta + psi_x = -(u theta_x + w theta_z),
      psi = psi_z = theta = 0 on z = 0, 1.
    ``linear=True`` drops the advection terms (growth-rate check against the EVP)."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=dealias)
    bz = Chebyshev(z, Nz, (0, 1), dealias=dealias)
    D = Domain(cs, [bx, bz])
    Dx = D.without(z)
    psi, th = Field("psi", D), Field("theta", D)
    tps = [Field(f"tpsi{i}", Dx) for i in range(1, 5)]
    tth = [Field(f"tth{i}", Dx) for i in range(1, 3)]
    Ra_, Pr_ = Parameter("Ra", Ra), Parameter("Pr", Pr)
    pb = IVP([psi, th, *tps, *tth], name="Rayleigh-Benard 2D (psi-theta)")
    u, w, om = diff(psi, z), -diff(psi, x), lap(psi)
    Npsi = 0 if linear else -(u * diff(om, x) + w * diff(om, z))
    Nth = 0 if linear else -(u * diff(th, x) + w * diff(th, z))
    pb.add_equation((dt(lap(psi)) - Pr_ * lap(lap(psi)) + Ra_ * Pr_ * diff(th, x)
                     + sum(lift(t, bz, -i) for i, t in enumerate(tps, 1)), Npsi))
    pb.add_equation((dt(th) - lap(th) + diff(psi, x)
                     + sum(lift(t, bz, -i) for i, t in enumerate(tth, 1)), Nth))
    for side in ("left", "right"):
        pb.add_equation((interp(psi, z, side), 0))
        pb.add_equation((interp(diff(psi, z), z, side), 0))
        pb.add_equation((interp(th, z, side), 0))
    return discretize(pb), dict(psi=psi, theta=th, x=x, z=z)
