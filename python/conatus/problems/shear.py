"""Wall-bounded plane shear flows: pressure-driven channel and plane Couette flow.

Each flow comes as a 3D Navier–Stokes IVP (``channel``, ``plane_couette``) and as the
Orr–Sommerfeld EVP of its laminar profile (``channel_evp``, ``plane_couette_evp``): the EVP is
the natural validation (Orszag 1971 for the channel; Romanov 1973 for Couette).
"""
from __future__ import annotations

from ..ir import (EVP, IVP, CartesianCoordinates, Chebyshev, Domain, Field, Parameter, RealFourier,
                  VectorField, diff, div, dot, dt, grad, integ, interp, lap, lift, unit_vector)
from ._ready import Ready, ready

__all__ = ["channel", "channel_evp", "plane_couette", "plane_couette_evp"]


def _walls_3d(Nx: int, Ny: int, Nz: int, Lx: float, Lz: float, dealias: float):
    """Coordinates, bases and unknowns shared by the two 3D wall-bounded IVPs (y = wall-normal)."""
    cs = CartesianCoordinates("x", "y", "z")
    x, y, z = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=dealias)
    by = Chebyshev(y, Ny, (-1, 1), dealias=dealias)
    bz = RealFourier(z, Nz, (0, Lz), dealias=dealias)
    D = Domain(cs, [bx, by, bz])
    u, p = VectorField("u", D), Field("p", D)
    tp = Field("tp", Domain(cs, []))                           # pressure gauge (a constant)
    tu1, tu2 = VectorField("tu1", D.without(y)), VectorField("tu2", D.without(y))
    return cs, (x, y, z), (bx, by, bz), (u, p, tp, tu1, tu2)


def channel(*, Nx: int, Ny: int, Nz: int, Lx: float, Lz: float, Re: float,
            pressure_gradient: float, dealias: float) -> Ready:
    """3D plane channel (Poiseuille) flow, incompressible Navier–Stokes, as an IVP.

    Equations (half-height h = 1, walls at y = -1, +1; x streamwise, z spanwise, both periodic)::

        dt(u) - lap(u)/Re + grad(p) = -u.grad(u) + G e_x,     div(u) = 0,
        u = 0 at y = -1 and y = +1.

    Units: length h, velocity U_c = laminar centreline velocity, Re = U_c h / nu. The flow is
    driven by the declared constant pressure gradient G (``pressure_gradient``): the laminar
    solution is U = (G Re / 2)(1 - y^2), so G = 2/Re gives U = 1 - y^2 (centreline velocity 1).
    Pressure by generalized tau (the top continuity row carries e_y.tu2, ``tp`` fixes the
    pressure level with integ(p) = 0), as in docs/site/tutorials/code/t02a_rayleigh_benard_cpu.py.
    Bases: RealFourier(x, Nx, (0, Lx)) x Chebyshev(y, Ny, (-1, 1)) x RealFourier(z, Nz, (0, Lz)).

    Validation reference: the laminar state is linearly stable below Re_c = 5772.22 (Orszag 1971,
    J. Fluid Mech. 50, 689); least-stable 2D mode at Re = 10000, alpha = 1:
    c = 0.23752649 + 0.00373967i. Checked on the Orr–Sommerfeld form, :func:`channel_evp`
    (tests/problems/test_problems_shear.py).
    """
    cs, (x, y, z), (bx, by, bz), (u, p, tp, tu1, tu2) = _walls_3d(Nx, Ny, Nz, Lx, Lz, dealias)
    Re_, G = Parameter("Re", Re), Parameter("G", pressure_gradient)
    ex, ey = unit_vector(x), unit_vector(y)
    pb = IVP([u, p, tp, tu1, tu2], name="plane channel 3D")
    pb.add_equation((div(u) + dot(ey, lift(tu2, by, -1)) + tp, 0))
    pb.add_equation((dt(u) - lap(u) / Re_ + grad(p) + lift(tu1, by, -1) + lift(tu2, by, -2),
                     -dot(u, grad(u)) + G * ex))
    pb.add_equation((interp(u, y, "left"), 0))                 # no slip, bottom wall
    pb.add_equation((interp(u, y, "right"), 0))                # no slip, top wall
    pb.add_equation((integ(p), 0))                             # pressure level
    return ready(pb, [Re_, G], [bx, by, bz], None,
                 notes=f"Laminar profile U = (G Re / 2)(1 - y^2) = {pressure_gradient * Re / 2:g} "
                       "(1 - y^2). Initial data: solver.set_state({u: ...}).",
                 reference="Orszag (1971): Re_c = 5772.22; c(Re = 1e4, alpha = 1) = "
                           "0.23752649 + 0.00373967i (see channel_evp).")


def channel_evp(*, N: int, Re: float, alpha: float) -> Ready:
    """Orr–Sommerfeld EVP of plane Poiseuille flow U = 1 - y^2 (2D modes).

    Normal modes v(y) exp(i alpha (x - c t)) of the wall-normal velocity, eigenvalue the complex
    wave speed c (growth rate alpha Im c)::

        (D^2 - alpha^2)^2 v / (i alpha Re) - U (D^2 - alpha^2) v + U'' v + c (D^2 - alpha^2) v = 0,
        v = v' = 0 at y = -1, +1.

    Units as in :func:`channel` (h = 1, U_c = 1, Re = U_c h / nu). Chebyshev(y, N, (-1, 1)),
    dealias 1 (a linear problem has no product to dealias; the NCC U is exact at N >= 3).
    ``ncc`` holds U; pass it to discretize. Spurious (tau) modes appear at large |c|: select the
    physical ones (|c| < 2, or a drift test between two N; tutorial t01).

    Validation reference: Orszag (1971), J. Fluid Mech. 50, 689, Table 3: at Re = 10000,
    alpha = 1, the least-stable mode is c = 0.23752649 + 0.00373967i; Re_c = 5772.22 at
    alpha_c = 1.02056 (tests/problems/test_problems_shear.py: c to 1e-8 at N = 64).
    """
    cs = CartesianCoordinates("y")
    y = cs["y"]
    by = Chebyshev(y, N, (-1, 1), dealias=1)
    D = Domain(cs, [by])
    v, U = Field("v", D, dtype=complex), Field("U", D)          # U: base flow (NCC)
    c = Parameter("c", dtype=complex)                          # eigenvalue: wave speed
    Re_, a = Parameter("Re", Re), Parameter("alpha", alpha)

    def L2(f):                                                 # D^2 - alpha^2
        return diff(diff(f, y), y) - a ** 2 * f

    pb = EVP([v], eigenvalue=c, name="Orr-Sommerfeld (plane Poiseuille)")
    pb.add_equation((L2(L2(v)) / (1j * a * Re_) - U * L2(v) + diff(diff(U, y), y) * v
                     + c * L2(v), 0))
    pb.add_bc(interp(v, y, -1), interp(v, y, 1),                        # v = 0 at the walls
              interp(diff(v, y), y, -1), interp(diff(v, y), y, 1))      # v' = 0 at the walls
    return ready(pb, [Re_, a], [by], {U: lambda Y: 1 - Y ** 2},
                 notes="Eigenvalue c (wave speed); growth rate alpha Im(c).",
                 reference="Orszag (1971) Table 3: c(Re = 1e4, alpha = 1) = 0.23752649 + "
                           "0.00373967i; Re_c = 5772.22, alpha_c = 1.02056.")


def plane_couette(*, Nx: int, Ny: int, Nz: int, Lx: float, Lz: float, Re: float,
                  dealias: float) -> Ready:
    """3D plane Couette flow, incompressible Navier–Stokes, as an IVP.

    Equations (half-gap h = 1, walls at y = -1, +1 moving at u = -e_x and +e_x)::

        dt(u) - lap(u)/Re + grad(p) = -u.grad(u),     div(u) = 0,
        u = -e_x at y = -1,   u = +e_x at y = +1.

    Units: length h, velocity U_w = wall speed, Re = U_w h / nu. Laminar solution U = y e_x.
    Same generalized-tau pressure treatment as :func:`channel`; no pressure gradient.

    Validation reference: plane Couette flow is linearly stable at every Re (Romanov 1973,
    Funct. Anal. Appl. 7, 137). No tabulated eigenvalue is reproduced here: the check
    (:func:`plane_couette_evp`, tests/problems/test_problems_shear.py) is Im c < 0 for the whole
    converged spectrum up to Re = 1e5, the centre-symmetry c -> -conj(c) of the spectrum, and
    convergence in N of the least-stable mode.
    """
    cs, (x, y, z), (bx, by, bz), (u, p, tp, tu1, tu2) = _walls_3d(Nx, Ny, Nz, Lx, Lz, dealias)
    Re_ = Parameter("Re", Re)
    ex, ey = unit_vector(x), unit_vector(y)
    pb = IVP([u, p, tp, tu1, tu2], name="plane Couette 3D")
    pb.add_equation((div(u) + dot(ey, lift(tu2, by, -1)) + tp, 0))
    pb.add_equation((dt(u) - lap(u) / Re_ + grad(p) + lift(tu1, by, -1) + lift(tu2, by, -2),
                     -dot(u, grad(u))))
    pb.add_equation((interp(u, y, "left"), -1 * ex))           # bottom wall moves at -U_w
    pb.add_equation((interp(u, y, "right"), ex))               # top wall moves at +U_w
    pb.add_equation((integ(p), 0))
    return ready(pb, [Re_], [bx, by, bz], None,
                 notes="Laminar profile U = y e_x. Initial data: solver.set_state({u: ...}).",
                 reference="Linearly stable for all Re (Romanov 1973); see plane_couette_evp.")


def plane_couette_evp(*, N: int, Re: float, alpha: float) -> Ready:
    """Orr–Sommerfeld EVP of plane Couette flow U = y (2D modes), same form as
    :func:`channel_evp` (U'' = 0): eigenvalue c, v = v' = 0 at y = -1, +1, Re = U_w h / nu.

    Validation reference: linearly stable for all Re (Romanov 1973): every converged
    eigenvalue has Im c < 0, and the spectrum is symmetric under c -> -conj(c) (the reflection
    y -> -y, x -> -x). No tabulated digit is reproduced (tests/problems/test_problems_shear.py).
    """
    cs = CartesianCoordinates("y")
    y = cs["y"]
    by = Chebyshev(y, N, (-1, 1), dealias=1)
    D = Domain(cs, [by])
    v, U = Field("v", D, dtype=complex), Field("U", D)
    c = Parameter("c", dtype=complex)
    Re_, a = Parameter("Re", Re), Parameter("alpha", alpha)

    def L2(f):
        return diff(diff(f, y), y) - a ** 2 * f

    pb = EVP([v], eigenvalue=c, name="Orr-Sommerfeld (plane Couette)")
    pb.add_equation((L2(L2(v)) / (1j * a * Re_) - U * L2(v) + c * L2(v), 0))   # U'' = 0
    pb.add_bc(interp(v, y, -1), interp(v, y, 1), interp(diff(v, y), y, -1),
              interp(diff(v, y), y, 1))
    return ready(pb, [Re_, a], [by], {U: lambda Y: Y + 0 * Y},
                 notes="Eigenvalue c (wave speed); growth rate alpha Im(c).",
                 reference="Romanov (1973): linearly stable, Im c < 0 for every mode.")
