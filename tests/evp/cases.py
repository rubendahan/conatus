"""Gate problems of written through the IR: Orr–Sommerfeld and Rayleigh–Bénard onset."""
from __future__ import annotations

import numpy as np
import scipy.optimize as so

from conatus.discretize import discretize
from conatus.ir import (EVEN, EVP, IVP, ODD, CartesianCoordinates, Chebyshev, ComplexFourier,
                        Domain, Field, Parameter, Reflection, SymmetrySubspace, VectorField, diff,
                        div, dot, dt, grad, integ, interp, lap, lift, unit_vector)
from conatus.ir import expr as E
from conatus.linearize import linearize
from conatus.solvers import solve_dense

# ---------------------------------------------------------------- Orr–Sommerfeld (v, 4th order)
ORSZAG_C = 0.23752649 + 0.00373967j        # Orszag 1971, Re = 1e4, alpha = 1
RE_C, ALPHA_C = 5772.22, 1.02056


def orr_sommerfeld(N: int, Re: float = 1e4, alpha: float = 1.0):
    """Plane Poiseuille, v-formulation; eigenvalue c:  (L + c M) v = 0."""
    cs = CartesianCoordinates("y")
    y = cs["y"]
    b = Chebyshev(y, N, (-1, 1), dealias=1)
    D = Domain(cs, [b])
    v, U, Upp = Field("v", D), Field("U", D), Field("Upp", D)
    taus = [Field(f"tau{i}", D.without(y)) for i in range(1, 5)]
    c = Parameter("c", dtype=complex)
    ia = 1j * alpha

    def L2(f):
        return diff(diff(f, y), y) - alpha ** 2 * f

    lhs = ((1 / (ia * Re)) * L2(L2(v)) - U * L2(v) + Upp * v + c * L2(v)
           + sum(lift(t, b, -(i + 1)) for i, t in enumerate(taus)))
    pb = EVP([v] + taus, eigenvalue=c, name="Orr-Sommerfeld")
    pb.add_equation((lhs, 0))
    for side in (-1, 1):
        pb.add_equation((interp(v, y, side), 0))
        pb.add_equation((interp(diff(v, y), y, side), 0))
    return discretize(pb, ncc={U: lambda Y: 1 - Y ** 2, Upp: lambda Y: -2 + 0 * Y})


def os_leading_c(N: int, Re: float, alpha: float) -> complex:
    gm = orr_sommerfeld(N, Re, alpha).matrices()
    r = solve_dense(gm.L, gm.M)
    phys = np.abs(r.values) < 2          # spurious tau modes have |c| ~ 1e3..1e4 (drift test)
    w = r.values[phys]
    return complex(w[np.argmax(w.imag)])


def os_critical(N: int = 64) -> tuple[float, float]:
    """(Re_c, alpha_c): minimum over alpha of the neutral Re (Im c = 0)."""
    def re_neutral(alpha):
        return so.brentq(lambda Re: os_leading_c(N, Re, alpha).imag, 5000, 7000, xtol=1e-6)
    opt = so.minimize_scalar(re_neutral, bounds=(0.98, 1.06), method="bounded",
                             options={"xatol": 1e-6})
    return float(opt.fun), float(opt.x)


# ---------------------------------------------------------------- Rayleigh–Bénard (Boussinesq)
RA_C, K_C = 1707.762, 3.1163      # rigid-rigid (Reid & Harris 1958; Chandrasekhar)


def boussinesq(k: float, Nz: int = 24):
    """Nonlinear Boussinesq IVP, z in (-1/2, 1/2), rigid walls, thermal-diffusion units."""
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = ComplexFourier(x, 4, (0, 2 * np.pi / k), dealias=1)
    bz = Chebyshev(z, Nz, (-0.5, 0.5), dealias=1)
    D = Domain(cs, [bx, bz])
    Dx = D.without(z)
    u, T, p = VectorField("u", D), Field("T", D), Field("p", D)
    tp = Field("tp", Domain(cs, []))
    tT1, tT2 = Field("tT1", Dx), Field("tT2", Dx)
    tu1, tu2 = VectorField("tu1", Dx), VectorField("tu2", Dx)
    Ra, Pr = Parameter("Ra", 1700.0), Parameter("Pr", 1.0)
    ez = unit_vector(z)
    ivp = IVP([u, T, p, tp, tT1, tT2, tu1, tu2], name="Boussinesq RB")
    ivp.add_equation((div(u) + tp, 0))
    ivp.add_equation((dt(T) - lap(T) + lift(tT1, bz, -1) + lift(tT2, bz, -2),
                      -dot(u, grad(T))))
    ivp.add_equation((dt(u) - Pr * lap(u) + grad(p) - Pr * Ra * T * ez
                      + lift(tu1, bz, -1) + lift(tu2, bz, -2), -dot(u, grad(u))))
    ivp.add_equation((interp(T, z, "left"), 0.5))
    ivp.add_equation((interp(T, z, "right"), -0.5))
    ivp.add_equation((interp(u, z, "left"), 0))
    ivp.add_equation((interp(u, z, "right"), 0))
    ivp.add_equation((integ(p), 0))
    T0 = Field("T0", Domain(cs, [bz]))
    return ivp, dict(cs=cs, z=z, u=u, T=T, T0=T0)


def rb_onset_evp(k: float, Nz: int = 24, symmetric: bool = False):
    ivp, f = boussinesq(k, Nz)
    s = Parameter("s", dtype=complex)
    G = None
    if symmetric:     # the onset mode: w and T even about mid-depth, u_x and p odd
        G = SymmetrySubspace(Reflection(f["z"], about=0), du=ODD, dT=EVEN, dp=ODD, dtp=ODD,
                             dtT1=ODD, dtT2=EVEN, dtu1=EVEN, dtu2=ODD, T0=ODD)
    evp = linearize(ivp, around={f["u"]: E.Zero((f["cs"],)), f["T"]: f["T0"]}, eigenvalue=s,
                    symmetry=G)
    return evp, f["T0"]


def rb_pencil(k: float, Nz: int = 24, symmetric: bool = False):
    """(L0, L1, M) with L(Ra) = L0 + Ra L1 exactly (Ra enters linearly), group k > 0."""
    evp, T0 = rb_onset_evp(k, Nz, symmetric)
    mats = []
    for ra in (0.0, 1.0):
        plan = discretize(evp, ncc={T0: lambda X, Z: -Z}, params={"Ra": ra})
        g = next(g for g in plan.groups if g[0].k > 0)
        mats.append(plan.matrices(g))
    return mats[0].L, mats[1].L - mats[0].L, mats[0].M, mats[0]


def rb_growth(L0, L1, M, Ra: float) -> float:
    r = solve_dense(L0 + Ra * L1, M)
    return float(np.max(r.values.real))


def rb_critical(Nz: int = 24, symmetric: bool = False) -> tuple[float, float]:
    cache = {}

    def ra_neutral(k):
        if k not in cache:
            L0, L1, M, _ = rb_pencil(k, Nz, symmetric)
            cache[k] = so.brentq(lambda Ra: rb_growth(L0, L1, M, Ra), 1500, 2000, xtol=1e-8)
        return cache[k]
    opt = so.minimize_scalar(ra_neutral, bounds=(2.9, 3.3), method="bounded",
                             options={"xatol": 1e-5})
    return float(opt.fun), float(opt.x)
