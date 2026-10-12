"""Ready problems in the native layer (``cn.problems.native``): equations as on paper.

Each function returns a :class:`conatus.Model` (no resolution, no tau, no lift: the native layer
places them and ``explain()`` lists every decision); the user then calls ``simulate(N=..., ...)``,
``linearize(...)``, ``eigenvalues(N)`` or ``discretize(N)``. They are the same problems as the
expert functions of :mod:`conatus.problems` (same equations, units, parameters and validation
references, given in each docstring), and their discrete matrices are **identical bit for bit**
to the expert ones at equal resolution (tests/problems/test_problems_native.py and
``test_problems_native2.py``), up to the order of
the unknowns and equations.

    import conatus as cn
    m = cn.problems.native.rayleigh_benard(Lx=2.0, Ra=2500, Pr=1)
    sim = m.simulate(N=(64, 32), scheme="RK443", dt=1e-3)

Linear-stability problems are the linearization of a native model about its base state: those
functions (``rayleigh_benard_evp``, ``pipe_evp``, ``taylor_couette_evp``,
``rotating_shell_convection_evp``) return the
:class:`conatus.model.Linearization` (``.eigenvalues(N, ...)``; time-periodic base:
``.floquet(N, ...)``, ``.optimal_gain(N, ...)``). Their tau formulation is the native one, so they
are compared with the expert eigenproblems by their eigenvalues and gains (round-off agreement,
given in each docstring), not by their matrices.
"""
from __future__ import annotations

import math

import numpy as np

from .._deprecation import renames

from ..model import Box, Linearization, Model, Pipe, Shell, TaylorCouette, periodic, walls
from ..model.conditions import free_slip

__all__ = ["rayleigh_benard", "kuramoto_sivashinsky", "swift_hohenberg", "channel", "plane_couette",
           "orr_sommerfeld", "pipe", "taylor_couette", "couette_profile", "kolmogorov", "rotating_shell_convection", "rotating_shell_convection_evp",
           "rayleigh_benard_evp", "pipe_evp", "taylor_couette_evp"]


def rayleigh_benard(*, Lx: float, Ra: float, Pr: float) -> Model:
    """2D Rayleigh–Benard convection between rigid plates at z = -1/2 (T = +1/2) and +1/2
    (T = -1/2), x periodic of length Lx; units and validation of
    :func:`conatus.problems.rayleigh_benard` (Ra_c = 1707.762 at k_c = 3.117, Reid & Harris 1958;
    growth rate 5.5425059 at Ra = 2500, k = pi, Pr = 1)."""
    m = Model(Box(x=periodic(Lx), z=walls(-0.5, 0.5)), name="Rayleigh-Benard 2D", Ra=Ra, Pr=Pr)
    m.velocity("u")
    m.pressure("p")
    m.scalar("T")
    m.equations("dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u@grad(u)",
                "div(u) = 0",
                "dt(T) - lap(T) = -u@grad(T)")
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m


def kuramoto_sivashinsky(*, L: float) -> Model:
    """Kuramoto–Sivashinsky dt(u) + u_xx + u_xxxx = -u u_x on the periodic line (0, L); validation
    of :func:`conatus.problems.kuramoto_sivashinsky` (sigma = k^2 - k^4; L = 22: E1, TW1 of
    Cvitanovic, Davidchack & Siminos 2010)."""
    m = Model(Box(x=periodic(L)), name=f"Kuramoto-Sivashinsky L={L:g}")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    return m


def swift_hohenberg(*, Lx: float, Ly: float, r: float) -> Model:
    """Swift–Hohenberg dt(u) - r u + (1 + lap)^2 u = -u^3 on a doubly periodic box; validation of
    :func:`conatus.problems.swift_hohenberg` (s = r - (1 - |k|^2)^2). The cubic term is
    dealiased exactly by the native default factor 2 = (3+1)/2 (degree 3)."""
    m = Model(Box(x=periodic(Lx), y=periodic(Ly)), name="Swift-Hohenberg 2D", r=r)
    m.scalar("u")
    m.equations("dt(u) - r*u + u + 2*lap(u) + lap(lap(u)) = -u*u*u")
    return m


def channel(*, Lx: float, Lz: float, Re: float, pressure_gradient: float) -> Model:
    """3D plane channel between walls y = -1, +1, driven by the pressure gradient G e_x; units and
    validation of :func:`conatus.problems.channel` (laminar U = (G Re/2)(1 - y^2); Re_c = 5772.22,
    Orszag 1971)."""
    m = Model(Box(x=periodic(Lx), y=walls(-1, 1), z=periodic(Lz)), name="plane channel 3D",
              Re=Re, G=pressure_gradient)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) - lap(u)/Re + grad(p) = -u@grad(u) + G*ex", "div(u) = 0")
    m.at("walls", u=0)
    return m


def plane_couette(*, Lx: float, Lz: float, Re: float) -> Model:
    """3D plane Couette flow, walls y = -1, +1 moving at u = -e_x and +e_x; units and validation of
    :func:`conatus.problems.plane_couette` (laminar U = y e_x; linearly stable, Romanov 1973)."""
    m = Model(Box(x=periodic(Lx), y=walls(-1, 1), z=periodic(Lz)), name="plane Couette 3D", Re=Re)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) - lap(u)/Re + grad(p) = -u@grad(u)", "div(u) = 0")
    m.at("bottom", u=(-1, 0, 0))
    m.at("top", u=(1, 0, 0))
    return m


def orr_sommerfeld(*, Re: float, alpha: float, flow: str) -> Model:
    """Orr–Sommerfeld EVP of plane Poiseuille (``flow="poiseuille"``, U = 1 - y^2) or Couette
    (``flow="couette"``, U = y) flow, eigenvalue the wave speed c; validation of
    :func:`conatus.problems.channel_evp` (Orszag 1971: c = 0.23752649 + 0.00373967i at Re = 1e4,
    alpha = 1) and :func:`conatus.problems.plane_couette_evp`."""
    profiles = {"poiseuille": lambda y: 1 - y ** 2, "couette": lambda y: y + 0 * y}
    if flow not in profiles:
        from ..errors import OptionError
        raise OptionError(f"orr_sommerfeld(flow={flow!r})", fix="flow='poiseuille' or 'couette'")
    m = Model(Box(y=walls(-1, 1)), name=f"Orr-Sommerfeld ({flow})", Re=Re, alpha=alpha)
    m.scalar("v")
    m.given(U=profiles[flow])
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    if flow == "poiseuille":
        m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    else:                       # U'' = 0 exactly: not written (its NCC would be round-off, 1e-14)
        m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    return m


def pipe(*, length: float, Re: float, pressure_gradient: float) -> Model:
    """Pressure-driven pipe flow of radius 1, z periodic of length ``length``; units and
    validation of :func:`conatus.problems.pipe` (laminar W = (G Re/4)(1 - r^2); Meseguer &
    Trefethen 2003, Table 1)."""
    m = Model(Pipe(1.0, length), name="pipe flow", Re=Re, G=pressure_gradient)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) - lap(u)/Re + grad(p) = -u@grad(u) + G*ez", "div(u) = 0")
    m.at("wall", u=0)
    return m


def couette_profile(eta: float, mu: float) -> tuple[float, float, float, float]:
    """(r_i, r_o, A, B) of circular Couette flow V = A r + B/r, gap 1, inner wall speed 1 (the
    units of :func:`conatus.problems.taylor_couette`; the same function, one formula)."""
    from .cylinders import _couette_profile
    return _couette_profile(eta, mu)


def taylor_couette(*, eta: float, mu: float, length: float, Re: float) -> Model:
    """Perturbation u of circular Couette flow U0 = (A r + B/r) e_phi between cylinders r_i, r_o
    (gap 1, inner wall speed 1, eta = r_i/r_o, mu = Omega_o/Omega_i), z periodic of length
    ``length``; equations, units and validation of :func:`conatus.problems.taylor_couette`
    (eta = 0.5, mu = 0: Re_c = 68.19 at k_c = 3.16, Recktenwald, Luecke & Mueller 1993)::

        dt(u) + U0@grad(u) + u@grad(U0) + grad(p) - lap(u)/Re = -u@grad(u),  div(u) = 0,
        u = 0 on both cylinders (the wall motion is carried by U0)."""
    ri, ro, A, B = couette_profile(eta, mu)
    m = Model(TaylorCouette(ri, ro, length), name="Taylor-Couette", Re=Re)
    m.velocity("u")
    m.pressure("p")
    m.given(U0=lambda r: np.stack([A * r + B / r, 0 * r, 0 * r]), rank=1)
    m.equations("dt(u) + U0@grad(u) + u@grad(U0) + grad(p) - lap(u)/Re = -u@grad(u)",
                "div(u) = 0")
    m.at("inner", u=0)
    m.at("outer", u=0)
    return m





def _top(top: str):
    """Value of the top condition of the truncated boundary layers (explicit choice, no default)."""
    if top not in ("stress_free", "no_slip"):
        from ..errors import OptionError
        raise OptionError(f"top = {top!r}", fix="'stress_free' or 'no_slip' (explicit choice)")
    return free_slip if top == "stress_free" else 0






def _coriolis(theta: float, f: float) -> tuple[dict, str]:
    """Parameters and text of the Coriolis term (4 pi/f) Ohat x u, Ohat = (0, -cos theta,
    sin theta), by components (``cross`` of a constant vector is not discretizable on the left in
    Cartesian geometry yet); no rotation (f = inf): no term."""
    if not math.isfinite(f):
        return {}, ""
    return (dict(Co=4 * math.pi / f, Oy=-math.cos(theta), Oz=math.sin(theta)),
            " + Co*(Oy*(u@ez) - Oz*(u@ey))*ex + Co*Oz*(u@ex)*ey - Co*Oy*(u@ex)*ez")








# --- Kolmogorov flow, rotating shell convection ------------------------------------------------

def kolmogorov(*, n: int, Re: float) -> Model:
    """2D Kolmogorov flow on the torus [0, 2 pi]^2 in streamfunction form (u = dy(psi),
    v = -dx(psi)), forcing sin(n y) e_x; equations, units and validation of
    :func:`conatus.problems.kolmogorov` (Chandler & Kerswell 2013, J. Fluid Mech. 722, 554:
    Re_lin = 9.9669 for n = 4)::

        dt(lap psi) - lap(lap psi)/Re = -dy(psi) dx(lap psi) + dx(psi) dy(lap psi) + n cos(n y),
        integ(psi) = 0.

    The written gauge integ(psi) = 0 fixes the mean of psi; the native layer adds its multiplier
    (a constant tau_psi) to the vorticity equation, whose mean mode reads tau_psi = mean of the
    right-hand side = 0. Laminar state psi = -(Re/n^3) cos(n y). L, M and the right-hand sides
    are bit-identical to the expert problem (tests/problems/test_problems_native2.py)."""
    m = Model(Box(x=periodic(2 * np.pi), y=periodic(2 * np.pi)), name=f"Kolmogorov 2D n={n}",
              Re=Re)
    m.scalar("psi")
    m.given(f=lambda y: n * np.cos(n * y))
    m.equations("dt(lap(psi)) - lap(lap(psi))/Re = -dy(psi)*dx(lap(psi)) + dx(psi)*dy(lap(psi)) + f",
                "integ(psi) = 0")
    return m


def _shell(eta: float, Ra: float, E: float, Pr: float) -> Model:
    ri = eta / (1 - eta)                                      # gap D = r_o - r_i = 1
    ro = 1 / (1 - eta)
    m = Model(Shell(ri, ro), name="rotating shell convection", Ra=Ra, E=E, Pr=Pr, ro=ro)
    m.velocity("u")
    m.pressure("p")
    m.scalar("T")
    # components (phi, theta, r): e_z = cos(theta) e_r - sin(theta) e_theta, r e_r, grad T0
    m.given(ez=lambda phi, theta: np.stack(np.broadcast_arrays(0 * theta + 0 * phi,
                                                               -np.sin(theta) + 0 * phi,
                                                               np.cos(theta) + 0 * phi)), rank=1)
    m.given(r_er=lambda r: np.stack([0 * r, 0 * r, r]), rank=1)
    m.given(gT0=lambda r: np.stack([0 * r, 0 * r, -ri * ro / r ** 2]), rank=1)
    m.equations("dt(u) - lap(u) + grad(p)/E - (Ra/(E*ro))*r_er*T = -u@grad(u) - explicit(2*cross(ez, u)/E)",
                "div(u) = 0",
                "dt(T) + u@gT0 - lap(T)/Pr = -u@grad(T)")
    m.at("inner", u=0, T=0)
    m.at("outer", u=0, T=0)
    return m


@renames("conatus.problems.native:rotating_shell_convection")
def rotating_shell_convection(*, eta: float, Ra: float, E: float, Pr: float) -> Model:
    """Rotating Boussinesq convection in a spherical shell (Christensen et al. 2001 geometry and
    units, non-magnetic): temperature perturbation T about the conduction profile
    T0 = r_i r_o/r - r_i, gap 1, radius ratio eta = r_i/r_o (``radius_ratio=`` is its deprecated
    v0.1 name), gravity r/r_o; equations, units and validation of
    :func:`conatus.problems.rotating_shell_convection` (Christensen et al. 2001, PEPI 128, 25,
    case 0: E_kin = 58.348 +- 0.050, drift 0.1824 +- 0.005)::

        dt(u) - lap(u) + grad(p)/E - (Ra/(E r_o)) T r e_r = -u.grad(u) - (2/E) e_z x u,
        div(u) = 0,   dt(T) + u.grad(T0) - lap(T)/Pr = -u.grad(T),
        u = 0, T = 0 at r_i and r_o.

    The Coriolis term is explicit (``explicit(...)``, as in the expert problem: its rate 2/E
    bounds the time step). The native shell recipe (first-order form, tau_2 in lam=2) is the
    expert one: L and M bit-identical, right-hand sides equal to round-off
    (tests/problems/test_problems_native2.py). Initial perturbation of the benchmark:
    ``conatus.problems.rotating_shell_convection(...).extras["initial_theta"]``."""
    return _shell(eta, Ra, E, Pr)


@renames("conatus.problems.native:rotating_shell_convection_evp")
def rotating_shell_convection_evp(*, eta: float, Ra: float, E: float,
                                  Pr: float) -> Linearization:
    """Onset of rotating shell convection: linearization of :func:`rotating_shell_convection`
    about the conduction state (u = 0, T = 0), normal modes exp(i m phi + s t), s in viscous units
    (Im s E = drift frequency in units of Omega)::

        lin = cn.problems.native.rotating_shell_convection_evp(eta=0.35, Ra=55.904299,
                                                               E=1e-3, Pr=1.0)
        res = lin.eigenvalues((12, 49, 32), m=4, method="shift-invert", target=-23j, n_modes=4)

    Same eigenvalues as :func:`conatus.problems.rotating_shell_convection_evp` (E-scaled rows,
    implicit Coriolis) to round-off (Nr = 24, Lmax = 15: 6e-14 on the leading mode). Validation
    reference: eta = 0.35, E = 1e-3, Pr = 1: m_c = 4, Ra_c = 55.904299, omega_c =
    -0.0230918 Omega (expert, (Nr, Lmax) = (32, 48)); Skene & Tobias 2024, Table 1: 55.9, 4,
    -0.0231; Barik et al. 2023: 55.904296."""
    return _shell(eta, Ra, E, Pr).linearize(about={"u": 0, "T": 0})


# --- normal-mode problems of the canonical flows ------------------------------------------------

def rayleigh_benard_evp(*, k: float, Ra: float, Pr: float) -> Linearization:
    """Onset of Rayleigh–Benard convection: linearization of :func:`rayleigh_benard` about the
    conduction state T = -z, x a complex axis of length 2 pi/k (one wave), normal modes
    exp(i k x + s t): ``lin.eigenvalues(N, k=k)``. Same eigenvalues as
    :func:`conatus.problems.rayleigh_benard_evp` to round-off (matrices equal to 4e-16, the
    expert writes i k by hand). Validation: Reid & Harris (1958), Ra_c = 1707.762 at k_c = 3.117;
    s = 5.5425059 at Ra = 2500, k = pi, Pr = 1."""
    m = Model(Box(x=periodic(2 * np.pi / k, complex=True), z=walls(-0.5, 0.5)),
              name="Rayleigh-Benard 2D", Ra=Ra, Pr=Pr)
    m.velocity("u")
    m.pressure("p")
    m.scalar("T")
    m.equations("dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u@grad(u)",
                "div(u) = 0",
                "dt(T) - lap(T) = -u@grad(T)")
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m.linearize(about={"u": 0, "T": lambda z: -z})


@renames("conatus.problems.native:pipe_evp")
def pipe_evp(*, k: float, Re: float) -> Linearization:
    """Linear stability of Hagen–Poiseuille flow U0 = (1 - r^2) e_z: linearization of
    :func:`pipe` (length 2 pi/k, pressure gradient 4/Re, so that the laminar flow has
    centreline velocity 1) about U0; ``lin.eigenvalues((N_phi, N_r, N_z), m=m, kz=k)``.
    ``k`` is the axial wavenumber (``alpha=`` is its deprecated v0.1 name).
    Same eigenvalues as :func:`conatus.problems.pipe_evp` to round-off (2e-13 on the ten least
    stable modes, N_r = 32). Validation: Meseguer & Trefethen (2003), Table 1, Re = 1000,
    (m, k) = (1, 1): -0.0708640053 + 0.8467498288i (frequency sign convention opposite)."""
    m = pipe(length=2 * np.pi / k, Re=Re, pressure_gradient=4 / Re)
    return m.linearize(about={"u": lambda r: np.stack([0 * r, 0 * r, 1 - r * r])})


def taylor_couette_evp(*, eta: float, mu: float, k: float, Re: float) -> Linearization:
    """Linear stability of circular Couette flow: linearization of :func:`taylor_couette` (length
    2 pi/k) about u = 0 (the perturbation of the Couette profile);
    ``lin.eigenvalues((N_phi, N_r, N_z), m=m, kz=k)``. Same eigenvalues as
    :func:`conatus.problems.taylor_couette_evp` to round-off (1e-14 on the four least stable
    modes, N_r = 24 to 40; the native annulus recipe carries the divergence tau, the expert one
    does not: both converge to the same spectrum). Validation: eta = 0.5, mu = 0: Re_c = 68.19 at
    k_c = 3.16 (Recktenwald, Luecke & Mueller 1993)."""
    return taylor_couette(eta=eta, mu=mu, length=2 * np.pi / k, Re=Re).linearize(about={"u": 0})
