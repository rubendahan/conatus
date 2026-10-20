"""Flows in cylindrical geometry: Hagen–Poiseuille pipe flow and Taylor–Couette flow between
rotating cylinders, each as an IVP and as the linear-stability EVP (the natural validation).

Regular bases at the axis (DiskBasis: Zernike) and on the annulus (AnnulusBasis); vector
components are (phi, r, z); never 1/(r + eps) (docs/site/tutorials/code/t08_polar.py).
"""
from __future__ import annotations

import numpy as np

from ..ir import (EVP, IVP, AnnulusBasis, ComplexFourier, CylindricalCoordinates, DiskBasis, Domain, Field,
                  Parameter, RealFourier, VectorField, div, dot, dt, grad, integ, interp, lap, lift,
                  unit_vector)
from .._deprecation import renames as _renames   # not a public name of this module (V5)
from .._deprecation import former_names as _former_names
from ._ready import Ready, ready

__all__ = ["pipe", "pipe_evp", "taylor_couette", "taylor_couette_evp"]

_MT2003 = ("Meseguer & Trefethen (2003), J. Comput. Phys. 186, 178, Table 1, Re = 1000: "
           "(m, k) = (1, 1) centre mode -0.0708640053 + 0.8467498288i, wall mode "
           "-0.0911426036 + 0.4691428789i (their sign convention of the frequency: the values "
           "here are the conjugates); (m, k) = (1, 0): -0.0146819706. Linearly stable at every Re.")
_TC_REF = ("eta = 0.5, outer cylinder at rest (mu = 0), axisymmetric modes: Re_c = 68.19 at "
           "k_c = 3.16 (Recktenwald, Luecke & Mueller 1993, Phys. Rev. E 48, 4444, Table I).")


def pipe(*, Nphi: int, Nr: int, Nz: int, Lz: float, Re: float, pressure_gradient: float,
         dealias: float) -> Ready:
    """Pressure-driven pipe flow (Hagen–Poiseuille), incompressible Navier–Stokes, IVP.

    Equations (pipe radius 1, z periodic of length Lz)::

        dt(u) - lap(u)/Re + grad(p) = -u.grad(u) + G e_z,    div(u) = 0,    u = 0 at r = 1.

    Units: radius R, centreline velocity U_c of the laminar flow, Re = U_c R / nu (Meseguer &
    Trefethen 2003). Laminar solution W = (G Re / 4)(1 - r^2): G = 4/Re gives W = 1 - r^2.
    Bases: RealFourier(phi, Nphi) x DiskBasis(r, Nr, radius 1) x ComplexFourier(z, Nz, (0, Lz))
    (the axial basis of a cylinder is ComplexFourier; a real field stores m >= 0, every k_z).
    The no-slip tau is lifted in the derivative basis of order 2 (``lam=2``), the pressure gauge
    ``tp`` with integ(p) = 0 (Dedalus disk example, tests/polar/test_polar_ivp_dedalus.py).

    Validation reference: linear stability of the laminar flow, :func:`pipe_evp`:
    Meseguer & Trefethen (2003), J. Comput. Phys. 186,
    178, Table 1, Re = 1000: (m, k) = (1, 1) centre mode -0.0708640053 + 0.8467498288i, wall mode
    -0.0911426036 + 0.4691428789i (conjugates of the tabulated values: opposite sign convention
    of the frequency); (m, k) = (1, 0): -0.0146819706. Linearly stable at every Re.
    """
    cc = CylindricalCoordinates("phi", "r", "z")
    phi, r, z = cc["phi"], cc["r"], cc["z"]
    bphi = RealFourier(phi, Nphi, (0, 2 * np.pi), dealias=dealias)
    br = DiskBasis(r, Nr, 1.0, dealias=dealias)
    bz = ComplexFourier(z, Nz, (0, Lz), dealias=dealias)             # axial: complex (required)
    D = Domain(cc, [bphi, br, bz])
    u, p = VectorField("u", D), Field("p", D)
    tu = VectorField("tu", D.without(r))
    tp = Field("tp", Domain(cc, []))
    Re_, G = Parameter("Re", Re), Parameter("G", pressure_gradient)
    pb = IVP([u, p, tu, tp], name="pipe flow")
    pb.add_equation((dt(u) - lap(u) / Re_ + grad(p) + lift(tu, br, -1, lam=2),
                     -dot(u, grad(u)) + G * unit_vector(z)))
    pb.add_equation((div(u) + tp, 0))
    pb.add_equation((interp(u, r, "right"), 0))                # no slip at r = 1
    pb.add_equation((integ(p), 0))
    return ready(pb, [Re_, G], [bphi, br, bz], None,
                 notes=f"Laminar W = (G Re / 4)(1 - r^2) = {pressure_gradient * Re / 4:g} (1 - r^2).",
                 reference=_MT2003)


@_renames("conatus.problems.cylinders:pipe_evp")
def pipe_evp(*, N: int, Re: float, m: int, k: float) -> Ready:
    """Linear stability of Hagen–Poiseuille flow U0 = (1 - r^2) e_z, primitive variables.

    Normal modes exp(i m phi + i k z + s t), k the axial wavenumber (``alpha=`` is its deprecated
    v0.1 name)::

        s u + U0.grad(u) + u.grad(U0) + grad(p) - lap(u)/Re = 0,   div(u) = 0,   u = 0 at r = 1,

    units of :func:`pipe` (Re = U_c R / nu). Eigenvalue s = growth rate + i frequency. Only the
    group ``plan.group(m, k)`` is meaningful: phi and z bases are ComplexFourier of sizes
    2|m| + 4 and 4 (structural: they hold m and k), the z period is 2 pi / k. DiskBasis(r,
    N), dealias 1 (linear; U0 is exact). No-slip tau placed by ``add_bc(..., into=0)``.

    Validation reference: Meseguer & Trefethen (2003), J. Comput. Phys. 186,
    178, Table 1, Re = 1000: (m, k) = (1, 1) centre mode -0.0708640053 + 0.8467498288i, wall mode
    -0.0911426036 + 0.4691428789i (conjugates of the tabulated values: opposite sign convention
    of the frequency); (m, k) = (1, 0): -0.0146819706. Linearly stable at every Re.
    (tests/problems/test_problems_cylinders.py, N = 32.)
    """
    cc = CylindricalCoordinates("phi", "r", "z")
    phi, r, z = cc["phi"], cc["r"], cc["z"]
    br = DiskBasis(r, N, 1.0, dealias=1)
    bphi = ComplexFourier(phi, 2 * abs(m) + 4, (0, 2 * np.pi), dealias=1)
    bz = ComplexFourier(z, 4, (0, 2 * np.pi / k), dealias=1)
    D = Domain(cc, [bphi, br, bz])
    u, p = VectorField("u", D, dtype=complex), Field("p", D, dtype=complex)
    tp = Field("tp", Domain(cc, []), dtype=complex)            # pressure gauge
    W = Field("W", Domain(cc, [br]))                           # axial base velocity (NCC)
    U0 = W * unit_vector(z)
    s = Parameter("s", dtype=complex)
    Re_ = Parameter("Re", Re)
    pb = EVP([u, p, tp], eigenvalue=s, name="Hagen-Poiseuille stability")
    pb.add_equation((s * u + dot(U0, grad(u)) + dot(u, grad(U0)) + grad(p) - lap(u) / Re_, 0))
    pb.add_equation((div(u) + tp, 0))
    pb.add_bc(interp(u, r, "right"), into=0)                   # no slip; tau into momentum
    pb.add_equation((integ(p), 0))
    return ready(pb, [Re_], [bphi, br, bz], {W: lambda P, R, Z: 1 - R * R},
                 notes=f"Use plan.group({m}, {k}). Eigenvalue s = sigma + i omega.",
                 reference=_MT2003, extras={"m": m, "k": k})


def _couette_profile(eta: float, mu: float) -> tuple[float, float, float, float]:
    """(r_i, r_o, A, B) of circular Couette flow V = A r + B / r, gap 1, inner wall speed 1.
    ``eta = r_i / r_o`` in (0, 1) (eta = 1 divided by zero, eta <= 0 gave a negative radius)."""
    if not 0 < eta < 1:
        from ..errors import OptionError
        raise OptionError(f"eta = {eta!r}: the radius ratio r_i / r_o must be in (0, 1)",
                          fix="eta=0.5 (Recktenwald et al. 1993)")
    ri, ro = eta / (1 - eta), 1 / (1 - eta)
    Om_i = 1 / ri                                              # inner angular velocity
    A = Om_i * (mu - eta ** 2) / (1 - eta ** 2)
    B = Om_i * ri ** 2 * (1 - mu) / (1 - eta ** 2)
    return ri, ro, A, B


def taylor_couette(*, Nphi: int, Nr: int, Nz: int, Lz: float, Re: float, eta: float, mu: float,
                   dealias: float) -> Ready:
    """Taylor–Couette flow between rotating cylinders: perturbation u of circular Couette flow
    U0 = V(r) e_phi, V = A r + B / r, incompressible Navier–Stokes, IVP::

        dt(u) + U0.grad(u) + u.grad(U0) + grad(p) - lap(u)/Re = -u.grad(u),   div(u) = 0,
        u = 0 at r = r_i and r = r_o (the wall motion is carried by U0).

    Units: gap d = r_o - r_i = 1, inner wall speed Omega_i r_i = 1, Re = Omega_i r_i d / nu;
    radius ratio eta = r_i / r_o, rotation ratio mu = Omega_o / Omega_i. Bases:
    RealFourier(phi, Nphi) x AnnulusBasis(r, Nr, (r_i, r_o)) x ComplexFourier(z, Nz, (0, Lz)); two
    taus lifted in the derivative basis of order 2 (``lam=2``), the radial part of the second
    one also into continuity (``dot(rvec, lift(t2, br, -1, lam=1))``: the plain second-order form
    of tutorial t08 is singular in the group m = 0, k_z = 0); pressure gauge ``tp``. ``ncc``
    holds U0 and rvec = r e_r. The linearized pencil in a group (m, k) has the spectrum of
    :func:`taylor_couette_evp` (tests/problems/test_problems_cylinders.py).

    Validation reference: linear onset, :func:`taylor_couette_evp`: eta = 0.5, outer cylinder at rest (mu = 0),
    axisymmetric modes: Re_c = 68.19 at k_c = 3.16 (Recktenwald, Luecke & Mueller 1993, Phys.
    Rev. E 48, 4444, Table I).
    """
    ri, ro, A, B = _couette_profile(eta, mu)
    cc = CylindricalCoordinates("phi", "r", "z")
    phi, r, z = cc["phi"], cc["r"], cc["z"]
    bphi = RealFourier(phi, Nphi, (0, 2 * np.pi), dealias=dealias)
    br = AnnulusBasis(r, Nr, (ri, ro), dealias=dealias)
    bz = ComplexFourier(z, Nz, (0, Lz), dealias=dealias)             # axial: complex (required)
    D = Domain(cc, [bphi, br, bz])
    u, p = VectorField("u", D), Field("p", D)
    t1, t2 = VectorField("t1", D.without(r)), VectorField("t2", D.without(r))
    tp = Field("tp", Domain(cc, []))
    U0 = VectorField("U0", Domain(cc, [br]))                   # circular Couette flow (NCC)
    rv = VectorField("rvec", Domain(cc, [br]))                 # r e_r (NCC)
    Re_, eta_, mu_ = Parameter("Re", Re), Parameter("eta", eta), Parameter("mu", mu)
    pb = IVP([u, p, t1, t2, tp], name="Taylor-Couette")
    pb.add_equation((dt(u) + dot(U0, grad(u)) + dot(u, grad(U0)) + grad(p) - lap(u) / Re_
                     + lift(t1, br, -1, lam=2) + lift(t2, br, -2, lam=2), -dot(u, grad(u))))
    # r e_r . t2 also closes continuity (generalized tau, as e_y . tu2 in the channel): without
    # it the group (m, k_z) = (0, 0) is singular (pressure-tau null mode, RankError)
    pb.add_equation((div(u) + dot(rv, lift(t2, br, -1, lam=1)) + tp, 0))
    pb.add_equation((interp(u, r, "left"), 0))
    pb.add_equation((interp(u, r, "right"), 0))
    pb.add_equation((integ(p), 0))

    def base(P, R, Z):                                         # (phi, r, z) components
        zero = 0 * (P + R + Z)
        return np.array([A * R + B / R + zero, zero, zero])
    ncc = {U0: base, rv: lambda P, R, Z: np.array([0 * (P + R + Z), R + 0 * (P + Z), 0 * (P + R + Z)])}
    return ready(pb, [Re_, eta_, mu_], [bphi, br, bz], ncc,
                 notes=f"r_i = {ri:.6g}, r_o = {ro:.6g}, V = {A:.6g} r + {B:.6g} / r. "
                       "eta and mu enter through U0 and the annulus (their Parameters are "
                       "informative).",
                 reference=_TC_REF, extras={"ri": ri, "ro": ro, "A": A, "B": B})


def taylor_couette_evp(*, N: int, Re: float, eta: float, mu: float, m: int, k: float) -> Ready:
    """Linear stability of circular Couette flow: normal modes exp(i m phi + i k z + s t) of the
    equations of :func:`taylor_couette` (same units, u = 0 on both cylinders).

    Eigenvalue s = growth rate + i frequency. Only ``plan.group(m, k)`` is meaningful: phi and z
    bases are ComplexFourier of sizes 2|m| + 4 and 4 (structural), z period 2 pi / k;
    AnnulusBasis(r, N), dealias 1 (linear; U0 = A r + B/r is resolved to round-off by the NCC
    expansion, see the plan's NCC report).

    Validation reference: eta = 0.5, outer cylinder at rest (mu = 0),
    axisymmetric modes: Re_c = 68.19 at k_c = 3.16 (Recktenwald, Luecke & Mueller 1993, Phys.
    Rev. E 48, 4444, Table I).
    Tutorial t08 finds 68.19 with N = 24 (tests/problems/test_problems_cylinders.py).
    """
    ri, ro, A, B = _couette_profile(eta, mu)
    cc = CylindricalCoordinates("phi", "r", "z")
    phi, r, z = cc["phi"], cc["r"], cc["z"]
    br = AnnulusBasis(r, N, (ri, ro), dealias=1)
    bphi = ComplexFourier(phi, 2 * abs(m) + 4, (0, 2 * np.pi), dealias=1)
    bz = ComplexFourier(z, 4, (0, 2 * np.pi / k), dealias=1)
    D = Domain(cc, [bphi, br, bz])
    Db = D.without(r)
    u, p = VectorField("u", D, dtype=complex), Field("p", D, dtype=complex)
    t1, t2 = VectorField("t1", Db, dtype=complex), VectorField("t2", Db, dtype=complex)
    tp = Field("tp", Domain(cc, []), dtype=complex)
    U0 = VectorField("U0", Domain(cc, [br]))
    s = Parameter("s", dtype=complex)
    Re_ = Parameter("Re", Re)
    pb = EVP([u, p, t1, t2, tp], eigenvalue=s, name="Taylor-Couette stability")
    pb.add_equation((s * u + dot(U0, grad(u)) + dot(u, grad(U0)) + grad(p) - lap(u) / Re_
                     + lift(t1, br, -1, lam=2) + lift(t2, br, -2, lam=2), 0))
    pb.add_equation((div(u) + tp, 0))
    pb.add_equation((interp(u, r, "left"), 0))
    pb.add_equation((interp(u, r, "right"), 0))
    pb.add_equation((integ(p), 0))

    def base(P, R, Z):
        zero = 0 * (P + R + Z)
        return np.array([A * R + B / R + zero, zero, zero])
    return ready(pb, [Re_], [bphi, br, bz], {U0: base},
                 notes=f"Use plan.group({m}, {k}). r_i = {ri:.6g}, r_o = {ro:.6g}.",
                 reference=_TC_REF, extras={"m": m, "k": k, "ri": ri, "ro": ro, "A": A, "B": B})


# 0.1 showed the decorator as `renames`: it still resolves, with a DeprecationWarning (L8).
__getattr__ = _former_names(__name__, {"renames": ("conatus._deprecation.renames", _renames)})
