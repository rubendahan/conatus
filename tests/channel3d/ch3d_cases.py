"""3D plane channel: incompressible Navier–Stokes, Fourier(x) x Chebyshev(y) x Fourier(z).

Units: half-height h = 1, y in (-1, 1), velocities scaled by the laminar centreline velocity,
Re = U_c h / nu. The mean flow is driven by a *declared* choice (no default):
  * ``pressure_gradient=G``: constant body force G ex (laminar solution U = (G Re / 2)(1 - y^2),
    so G = 2/Re gives U = 1 - y^2);
  * ``flow_rate=Q``: scalar unknown ``P`` (mean pressure gradient) fixed by the constraint
    (1/(Lx Lz)) integ(u_x) = 2 Q, i.e. Q is the bulk velocity (laminar Poiseuille: Q = 2/3).
Pressure by generalized tau, as in the 2D RB case: the top continuity
row carries ey.tu2, ``tp`` is the pressure gauge. See"""
from __future__ import annotations

import numpy as np

from conatus.discretize import discretize
from conatus.ir import (IVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain, Field,
                        Parameter, RealFourier, VectorField, diff, div, dot, dt, grad, integ,
                        interp, lap, lift, unit_vector)


def channel_3d(Nx: int, Ny: int, Nz: int, Re: float, Lx: float = 2 * np.pi, Lz: float = np.pi,
               *, pressure_gradient: float | None = None, flow_rate: float | None = None,
               dealias: float = 1.5, linear_about_poiseuille: bool = False,
               assembly: str = "direct", grid: str = "floor", fourier: str = "real"):
    """Discretized channel problem. Exactly one of ``pressure_gradient`` / ``flow_rate``.
    ``linear_about_poiseuille=True``: perturbation about U = 1 - y^2 (NCC), quadratic term
    dropped (growth-rate checks against the Orr–Sommerfeld EVP and Squire).
    ``assembly``: group-matrix assembly of the plan (``"direct"`` or ``"wavenumber_interpolation"``).
    ``grid``: dealiased grid rule of the three bases (``"floor"`` or ``"smooth7"``, see bases.py).
    ``fourier``: ``"real"`` (RealFourier: each (kx, kz) group carries the 4 cos/sin products, one
    coupled real system of 4x the size) or ``"complex"`` (ComplexFourier: one complex system per
    mode exp(i(kx x + kz z)), both signs stored)."""
    if fourier not in ("real", "complex"):
        raise ValueError(f"fourier must be 'real' or 'complex', got {fourier!r}")
    FB = RealFourier if fourier == "real" else ComplexFourier
    if (pressure_gradient is None) == (flow_rate is None):
        raise ValueError("declare exactly one of pressure_gradient= or flow_rate= (no default)")
    cs = CartesianCoordinates("x", "y", "z")
    x, y, z = cs.coords
    bx = FB(x, Nx, (0, Lx), dealias=dealias, grid=grid)
    by = Chebyshev(y, Ny, (-1, 1), dealias=dealias, grid=grid)
    bz = FB(z, Nz, (0, Lz), dealias=dealias, grid=grid)
    D = Domain(cs, [bx, by, bz])
    Dxz, D0 = D.without(y), Domain(cs, [])
    u, p = VectorField("u", D), Field("p", D)
    tp = Field("tp", D0)
    tu1, tu2 = VectorField("tu1", Dxz), VectorField("tu2", Dxz)
    Re_ = Parameter("Re", Re)
    ex, ey = unit_vector(x), unit_vector(y)
    unknowns = [u, p, tp, tu1, tu2]
    P = Field("P", D0) if flow_rate is not None else None
    if P is not None:
        unknowns.append(P)
    pb = IVP(unknowns, name="Channel 3D")

    U, ncc = None, {}
    mom = dt(u) - lap(u) / Re_ + grad(p) + lift(tu1, by, -1) + lift(tu2, by, -2)
    if linear_about_poiseuille:
        U = Field("U", Domain(cs, [by]))
        ncc = {U: lambda X, Y, Z: 1 - Y**2}
        mom = mom + U * diff(u, x) + dot(u, ey) * diff(U, y) * ex
        rhs = 0
    else:
        rhs = -dot(u, grad(u))
    if pressure_gradient is not None:
        rhs = rhs + pressure_gradient * ex
    else:
        mom = mom - P * ex

    pb.add_equation((div(u) + dot(ey, lift(tu2, by, -1)) + tp, 0))
    pb.add_equation((mom, rhs))
    pb.add_equation((interp(u, y, "left"), 0))
    pb.add_equation((interp(u, y, "right"), 0))
    pb.add_equation((integ(p), 0))
    if P is not None:
        pb.add_equation((integ(dot(ex, u)) / (Lx * Lz), 2.0 * flow_rate))
    plan = discretize(pb, ncc=ncc or None, assembly=assembly)
    return plan, dict(u=u, p=p, x=x, y=y, z=z, P=P, U=U)


def vector_init(fx, fy, fz):
    """Initial-data callable for a vector field from three callables of (X, Y, Z)."""
    def f(X, Y, Z):
        X, Y, Z = np.broadcast_arrays(X, Y, Z)
        return np.stack([fx(X, Y, Z), fy(X, Y, Z), fz(X, Y, Z)])
    return f


# ---------------------------------------------------------------------------------------------
# Poloidal–toroidal (scalar) formulation. §Formulation
# scalaire. Same physics as channel_3d, written by the user as a standard IR problem:
#   u = curl curl(phi ey) + curl(psi ey) + U(y) ex + W(y) ez
#     = (dx dy phi + dz psi + U,  -lap_h phi,  dz dy phi - dx psi + W),   lap_h = dx^2 + dz^2,
# divergence-free by construction (no pressure). ey.curl curl and ey.curl of the momentum
# equation give, for every (kx, kz) != (0, 0):
#   lap_h [dt lap phi - lap^2 phi / Re] = dy(dx Nx + dz Nz) - lap_h Ny   (order 4 in y)
#   lap_h [dt psi     - lap psi / Re]   = dz Nx - dx Nz                  (order 2 in y)
# with N = -div(u u) (divergence form, 6 products). Walls: phi = dy phi = psi = 0.
# The mean mode is separate: dt U - U''/Re = <Nx> + drive, dt W - W''/Re = <Nz>, U = W = 0 at
# the walls, <.> = xz-average. At (kx, kz) = (0, 0) phi and psi carry no velocity: the term
# <phi> (resp. <psi>), non-zero only in that group, pins them to 0 -- still a standard IR term
# (partial integration), no special case in the compiler.
# Taus: lifted in the equations' own row spaces, C^(4) (phi) and C^(2) (psi, U, W): the last
# 4 / 2 rows of the ultraspherical equations carry the taus -- the classic Olver–Townsend tau,
# same discretization as the hand-written comparator (Galerkin basis, truncated C^(4)/C^(2)
# equations). T-lifts (as in channel_3d) give spurious unstable eigenvalues here.
def channel_3d_pt(Nx: int, Ny: int, Nz: int, Re: float, Lx: float = 2 * np.pi,
                  Lz: float = np.pi, *, pressure_gradient: float | None = None,
                  flow_rate: float | None = None, dealias: float = 1.5,
                  linear_about_poiseuille: bool = False, assembly: str = "direct",
                  grid: str = "floor", tau_lift: str = "derivative",
                  wall_velocity: tuple[float, float] | None = None, linear_about: str | None = None):
    """Discretized poloidal–toroidal channel (RealFourier x, z). Exactly one drive, as in
    :func:`channel_3d`. ``wall_velocity=(U_bottom, U_top)``: moving walls, U(-1) = U_bottom,
    U(+1) = U_top (plane Couette: (-1, 1) with ``pressure_gradient=0.0`` declared; laminar
    U = y); default None = fixed walls. ``linear_about="couette"``: perturbation about U0 = y
    (NCC; Orr–Sommerfeld/Squire checks), or a callable U0(y): perturbation about that mean profile
(Oseen preconditioner about a solution's mean flow); exclusive with ``linear_about_poiseuille``. Returns (plan, fields) with fields phi, psi, U, W, P, x, y, z and
    ``u``: the velocity expression (for initial data, diagnostics and comparisons).
    ``tau_lift``: ``"derivative"`` (taus in C^(4) / C^(2), default) or ``"T"`` (taus in T; kept
    only to exhibit the spurious unstable eigenvalues it creates)."""
    if tau_lift not in ("derivative", "T"):
        raise ValueError(f"tau_lift must be 'derivative' or 'T', got {tau_lift!r}")
    l4, l2 = (4, 2) if tau_lift == "derivative" else (0, 0)
    if (pressure_gradient is None) == (flow_rate is None):
        raise ValueError("declare exactly one of pressure_gradient= or flow_rate= (no default)")
    cs = CartesianCoordinates("x", "y", "z")
    x, y, z = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=dealias, grid=grid)
    by = Chebyshev(y, Ny, (-1, 1), dealias=dealias, grid=grid)
    bz = RealFourier(z, Nz, (0, Lz), dealias=dealias, grid=grid)
    D = Domain(cs, [bx, by, bz])
    Dxz, Dy, D0 = D.without(y), Domain(cs, [by]), Domain(cs, [])
    phi, psi = Field("phi", D), Field("psi", D)
    U, W = Field("U", Dy), Field("W", Dy)
    tph = [Field(f"tphi{i}", Dxz) for i in range(1, 5)]
    tps = [Field(f"tpsi{i}", Dxz) for i in range(1, 3)]
    tU = [Field(f"tU{i}", D0) for i in range(1, 3)]
    tW = [Field(f"tW{i}", D0) for i in range(1, 3)]
    unknowns = [phi, psi, U, W, *tph, *tps, *tU, *tW]
    P = Field("P", D0) if flow_rate is not None else None
    if P is not None:
        unknowns.append(P)
    Re_ = Parameter("Re", Re)
    pb = IVP(unknowns, name="Channel 3D (poloidal-toroidal)")

    def dxx(f):
        return diff(diff(f, x), x)

    def lap_h(f):
        return dxx(f) + diff(diff(f, z), z)

    def mean_xz(f):
        return integ(f, (x, z)) / (Lx * Lz)

    ux = diff(diff(phi, x), y) + diff(psi, z) + U
    uy = -lap_h(phi)
    uz = diff(diff(phi, z), y) - diff(psi, x) + W
    u = [ux, uy, uz]
    X = [x, y, z]

    def N_div(vel):
        """N_i = -d_j(u_i u_j): divergence form, 6 distinct products."""
        return [-sum(diff(vel[i] * vel[j], X[j]) for j in range(3)) for i in range(3)]

    ncc = {}
    if linear_about_poiseuille and linear_about is not None:
        raise ValueError("linear_about_poiseuille and linear_about are exclusive")
    if not (linear_about in (None, "couette") or callable(linear_about)):
        raise ValueError(f"linear_about must be None, 'couette' or a profile U0(y), got {linear_about!r}")
    if linear_about_poiseuille or linear_about:
        # perturbation about U0 (NCC): N_lin = -(U0 dx u + v U0' ex), on the LHS;
        # U0 = 1 - y^2 (Poiseuille) or U0 = y (Couette)
        U0 = Field("U0", Dy)
        if linear_about_poiseuille:
            prof = lambda X_, Y_, Z_: 1 - Y_**2  # noqa: E731
        elif linear_about == "couette":
            prof = lambda X_, Y_, Z_: Y_ + 0 * X_ * Z_  # noqa: E731
        else:                        # a mean profile U0(y) given by the caller (e.g. a solution's)
            prof = lambda X_, Y_, Z_: linear_about(Y_) + 0 * X_ * Z_  # noqa: E731
        ncc = {U0: prof}
        Nl = [-(U0 * diff(ux, x) + uy * diff(U0, y)), -U0 * diff(uy, x), -U0 * diff(uz, x)]
        lin_phi = -(diff(diff(Nl[0], x) + diff(Nl[2], z), y) - lap_h(Nl[1]))
        lin_psi = -(diff(Nl[0], z) - diff(Nl[2], x))
        rhs_phi = rhs_psi = rhs_U = rhs_W = 0
    else:
        N = N_div(u)
        lin_phi = lin_psi = 0
        rhs_phi = diff(diff(N[0], x) + diff(N[2], z), y) - lap_h(N[1])
        rhs_psi = diff(N[0], z) - diff(N[2], x)
        rhs_U, rhs_W = mean_xz(N[0]), mean_xz(N[2])

    lifts_phi = sum(lift(t, by, -i, lam=l4) for i, t in enumerate(tph, start=1))
    lifts_psi = sum(lift(t, by, -i, lam=l2) for i, t in enumerate(tps, start=1))
    pb.add_equation((lap_h(dt(lap(phi))) - lap_h(lap(lap(phi))) / Re_ + lin_phi
                     + mean_xz(phi) + lifts_phi, rhs_phi))
    pb.add_equation((lap_h(dt(psi)) - lap_h(lap(psi)) / Re_ + lin_psi
                     + mean_xz(psi) + lifts_psi, rhs_psi))
    momU = dt(U) - diff(diff(U, y), y) / Re_ + lift(tU[0], by, -1, lam=l2) + lift(tU[1], by, -2, lam=l2)
    momW = dt(W) - diff(diff(W, y), y) / Re_ + lift(tW[0], by, -1, lam=l2) + lift(tW[1], by, -2, lam=l2)
    if pressure_gradient is not None:
        rhs_U = rhs_U + pressure_gradient
    else:
        momU = momU - P
    pb.add_equation((momU, rhs_U))
    pb.add_equation((momW, rhs_W))
    wall_u = (0.0, 0.0) if wall_velocity is None else tuple(float(v) for v in wall_velocity)
    for side, uw in zip(("left", "right"), wall_u):
        pb.add_equation((interp(phi, y, side), 0))
        pb.add_equation((interp(diff(phi, y), y, side), 0))
        pb.add_equation((interp(psi, y, side), 0))
        pb.add_equation((interp(U, y, side), uw))
        pb.add_equation((interp(W, y, side), 0))
    if P is not None:
        pb.add_equation((integ(U, y), 2.0 * flow_rate))
    plan = discretize(pb, ncc=ncc or None, assembly=assembly)
    return plan, dict(phi=phi, psi=psi, U=U, W=W, P=P, x=x, y=y, z=z, u=u)


def pt_velocity_grid(plan, f, coeffs: dict) -> np.ndarray:
    """Velocity (3, Nx, Ny, Nz) on the N-point grid from poloidal–toroidal coefficient arrays
    ``coeffs`` = {name: array} (reference evaluator; diagnostics and tests only)."""
    from conatus.reference.evaluate import FieldData
    data = {f[k]: plan._as_grid(f[k], FieldData(f[k], coeffs[k], layout="coeff"))
            for k in ("phi", "psi", "U", "W")}
    dom = plan.domain
    return np.stack([plan._eval_grid(c, dom, data) for c in f["u"]])


def pt_initial(plan, f, phi0, psi0, U0, W0) -> tuple[dict, np.ndarray]:
    """Initial data {Field: callable} of the poloidal–toroidal problem and the matching
    primitive velocity on the N-point grid (to start channel_3d from the same state)."""
    data = {f["phi"]: phi0, f["psi"]: psi0, f["U"]: U0, f["W"]: W0}
    grids = {k: plan._as_grid(k, v) for k, v in data.items()}
    u = np.stack([plan._eval_grid(c, plan.domain, grids) for c in f["u"]])
    return data, u
