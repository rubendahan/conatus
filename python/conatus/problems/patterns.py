"""Periodic model problems: Kuramoto–Sivashinsky, Swift–Hohenberg and 2D Kolmogorov flow.

All periodic (RealFourier): their linear growth rates about the trivial or laminar state are the
validation (exact for KS and SH; Chandler & Kerswell 2013 for Kolmogorov).
"""
from __future__ import annotations

import numpy as np

from ..ir import IVP, CartesianCoordinates, Domain, Field, Parameter, RealFourier, diff, dt, integ, lap
from ._ready import Ready, ready

__all__ = ["kuramoto_sivashinsky", "swift_hohenberg", "kolmogorov"]


def kuramoto_sivashinsky(*, N: int, L: float, dealias: float) -> Ready:
    """Kuramoto–Sivashinsky equation on the periodic line (0, L), IVP::

        dt(u) + u_xx + u_xxxx = -u u_x.

    Linear terms implicit (LHS), the quadratic term explicit (RHS). RealFourier(x, N, (0, L)).
    The size L is the only parameter (L = 22: the standard chaotic case).

    Validation reference: the linear growth rate of exp(i k x) about u = 0 is exactly
    sigma = k^2 - k^4 (tests/problems/test_problems_patterns.py, every k = 2 pi j / L). L = 22:
    equilibria E1, E2 of energy 0.2609, 0.4382 and travelling wave TW1 of speed 0.73699
    (Cvitanovic, Davidchack & Siminos 2010, SIAM J. Appl. Dyn. Syst. 9, 1, Tables 5.1-5.2;
    tutorial t05 reproduces them from this equation).
    """
    cs = CartesianCoordinates("x")
    x = cs["x"]
    bx = RealFourier(x, N, (0, L), dealias=dealias)
    u = Field("u", Domain(cs, [bx]))
    L_ = Parameter("L", L)                                     # informative: L sets the basis
    pb = IVP([u], name=f"Kuramoto-Sivashinsky L={L:g}")
    pb.add_equation((dt(u) + lap(u) + lap(lap(u)), -u * diff(u, x)))
    return ready(pb, [L_], [bx], None,
                 notes="Energy (1/L) int u^2/2 dx in Cvitanovic et al.'s tables.",
                 reference="sigma(k) = k^2 - k^4 exactly; L = 22: E1 energy 0.2609, TW1 speed "
                           "0.73699 (Cvitanovic, Davidchack & Siminos 2010).")


def swift_hohenberg(*, Nx: int, Ny: int, Lx: float, Ly: float, r: float,
                    dealias: float) -> Ready:
    """Swift–Hohenberg equation on a doubly periodic box, IVP (Swift & Hohenberg 1977)::

        dt(u) - r u + (1 + lap)^2 u = -u^3.

    Critical wavenumber 1 (length unit = 1 / q_c), time unit as written; r = distance from
    onset. RealFourier(x, Nx, (0, Lx)) x RealFourier(y, Ny, (0, Ly)). The cubic term is
    dealiased exactly by dealias = 2 (a quadratic one by 3/2): it is your argument.

    Validation reference: exp(i k.x) grows at s = r - (1 - |k|^2)^2 about u = 0, so onset is
    r_c = 0 at |k| = 1 (Cross & Hohenberg 1993, Rev. Mod. Phys. 65, 851);
    tests/problems/test_problems_patterns.py checks every Fourier group exactly. Rolls: A^2 = 4 r / 3
    + r^2 / 144 + O(r^3) (examples/patterns/swift_hohenberg.py).
    """
    cs = CartesianCoordinates("x", "y")
    x, y = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=dealias)
    by = RealFourier(y, Ny, (0, Ly), dealias=dealias)
    u, r_ = Field("u", Domain(cs, [bx, by])), Parameter("r", r)
    pb = IVP([u], name="Swift-Hohenberg 2D")
    pb.add_equation((dt(u) - r_ * u + u + 2 * lap(u) + lap(lap(u)), -u * u * u))
    return ready(pb, [r_], [bx, by], None,
                 notes="(1 + lap)^2 u written out as u + 2 lap(u) + lap(lap(u)).",
                 reference="s(k) = r - (1 - |k|^2)^2; onset r_c = 0 at |k| = 1.")


def kolmogorov(*, N: int, n: int, Re: float, dealias: float) -> Ready:
    """2D Kolmogorov flow on the torus [0, 2 pi]^2 (Chandler & Kerswell 2013), IVP in
    streamfunction form (u = psi_y, v = -psi_x, vorticity -lap(psi))::

        dt(lap psi) - lap(lap psi)/Re + tau = -psi_y (lap psi)_x + psi_x (lap psi)_y + n cos(n y),
        integ(psi) = 0,

    i.e. dt(u) + u.grad(u) + grad(p) = lap(u)/Re + sin(n y) e_x. ``tau`` is a constant (its
    only role: the (0, 0) mode of the first equation, where lap psi = 0, gives tau = 0).
    Laminar state psi = -(Re / n^3) cos(n y), u = (Re / n^2) sin(n y) e_x. RealFourier(N) in x
    and y. ``ncc`` holds the forcing f = n cos(n y). Same equations as
    conatus.ecs.kolmogorov.kolmogorov_solver (which returns a solver).

    No ``kolmogorov_evp``: the laminar streamfunction varies along a Fourier direction, and
    discretize refuses such an NCC (it couples wavenumbers; DiscretizationError). The linear
    stability is computed on the IVP instead, from the Jacobian of its right-hand side:
    ``conatus.ecs.kolmogorov.linear_stability_dense(IVPSolver(plan, ...), laminar_state(...))``.

    Validation reference: laminar flow loses stability at Re_lin = 9.9669 for n = 4
    (Chandler & Kerswell 2013, J. Fluid Mech. 722, 554, Fig. 3 caption): the leading eigenvalue
    changes sign, real, between Re = 9.96685 and 9.96695 at N = 24
    (tests/problems/test_problems_patterns.py, as tests/ecs/test_kolmogorov.py). Rescaled to the forcing wavelength, Re_MS = Re^2 / n^3 tends to
    sqrt(2) as the domain grows (Meshalkin & Sinai 1961): 9.9669^2 / 64 = 1.552 here, where
    the smallest wavenumber is 1/4 of the forcing's.
    """
    cs = CartesianCoordinates("x", "y")
    x, y = cs.coords
    bx = RealFourier(x, N, (0, 2 * np.pi), dealias=dealias)
    by = RealFourier(y, N, (0, 2 * np.pi), dealias=dealias)
    D = Domain(cs, [bx, by])
    psi, tau, f = Field("psi", D), Field("tau", Domain(cs, [])), Field("f", D)
    Re_ = Parameter("Re", Re)
    q = lap(psi)
    pb = IVP([psi, tau], name=f"Kolmogorov 2D n={n}")
    pb.add_equation((dt(q) - lap(q) / Re_ + tau,
                     -diff(psi, y) * diff(q, x) + diff(psi, x) * diff(q, y) + f))
    pb.add_equation((integ(psi), 0))
    return ready(pb, [Re_], [bx, by], {f: lambda X, Y: n * np.cos(n * Y)},
                 notes=f"Laminar psi = -(Re/n^3) cos(n y) = {-Re / n ** 3:.6g} cos({n} y).",
                 reference="Re_lin = 9.9669 for n = 4 (Chandler & Kerswell 2013).",
                 extras={"n": n})
