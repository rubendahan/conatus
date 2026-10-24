"""Thermal convection: Rayleigh–Benard between rigid plates and rotating convection in a
spherical shell (Christensen et al. 2001 benchmark geometry), each as an IVP and as the EVP of
the linear onset (the natural validation)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..ir import (EVP, IVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain, Field, Parameter,
                  RealFourier, ShellBasis, SphereBasis, SphericalCoordinates, VectorField, cross, diff, div,
                  dot, dt, grad, integ, interp, lap, lift, outer, unit_vector)
from .._deprecation import renames as _renames   # not a public name of this module (V5)
from .._deprecation import former_names as _former_names
from ._ready import Ready, ready

__all__ = ["rayleigh_benard", "rayleigh_benard_evp", "rotating_shell_convection",
           "rotating_shell_convection_evp"]

_RB_REFERENCE = ("Rigid-rigid onset Ra_c = 1707.762 at k_c = 3.117, independent of Pr "
                 "(Reid & Harris 1958, Phys. Fluids 1, 102; Chandrasekhar 1961, Table III).")


def rayleigh_benard(*, Nx: int, Nz: int, Lx: float, Ra: float, Pr: float,
                    dealias: float) -> Ready:
    """2D Rayleigh–Benard convection between rigid, fixed-temperature plates (Boussinesq), IVP.

    Equations (z in [-1/2, 1/2], x periodic of length Lx)::

        dt(u) - Pr lap(u) + grad(p) - Pr Ra T e_z = -u.grad(u),     div(u) = 0,
        dt(T) - lap(T) = -u.grad(T),
        u = 0 and T = +1/2 at z = -1/2 (hot bottom),  u = 0 and T = -1/2 at z = +1/2.

    Units: length = gap d, time = thermal diffusion time d^2/kappa, temperature = the imposed
    difference; Ra = g alpha DeltaT d^3 / (nu kappa), Pr = nu / kappa. Conduction state
    T = -z, u = 0. Pressure by generalized tau, gauge ``tp`` with integ(p) = 0 (tutorial t02a).

    Validation reference: Reid & Harris (1958): Ra_c = 1707.762, k_c = 3.117 (checked with
    :func:`rayleigh_benard_evp`, tests/problems/test_problems_convection.py). At Ra = 2500, k = pi,
    Pr = 1 the onset roll grows at sigma = 5.5425059 (tutorial t02a: IVP = EVP to 1e-5).
    """
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, Nx, (0, Lx), dealias=dealias)
    bz = Chebyshev(z, Nz, (-0.5, 0.5), dealias=dealias)
    D = Domain(cs, [bx, bz])
    u, T, p = VectorField("u", D), Field("T", D), Field("p", D)
    tp = Field("tp", Domain(cs, []))                           # pressure gauge
    tT1, tT2 = Field("tT1", D.without(z)), Field("tT2", D.without(z))
    tu1, tu2 = VectorField("tu1", D.without(z)), VectorField("tu2", D.without(z))
    Ra_, Pr_ = Parameter("Ra", Ra), Parameter("Pr", Pr)
    ez = unit_vector(z)
    pb = IVP([u, T, p, tp, tT1, tT2, tu1, tu2], name="Rayleigh-Benard 2D")
    pb.add_equation((div(u) + dot(ez, lift(tu2, bz, -1)) + tp, 0))
    pb.add_equation((dt(T) - lap(T) + lift(tT1, bz, -1) + lift(tT2, bz, -2), -dot(u, grad(T))))
    pb.add_equation((dt(u) - Pr_ * lap(u) + grad(p) - Pr_ * Ra_ * T * ez
                     + lift(tu1, bz, -1) + lift(tu2, bz, -2), -dot(u, grad(u))))
    pb.add_equation((interp(T, z, "left"), 0.5))              # hot bottom
    pb.add_equation((interp(T, z, "right"), -0.5))            # cold top
    pb.add_equation((interp(u, z, "left"), 0))                # no slip
    pb.add_equation((interp(u, z, "right"), 0))
    pb.add_equation((integ(p), 0))
    return ready(pb, [Ra_, Pr_], [bx, bz], None,
                 notes="Start from the conduction state T = -z plus a perturbation "
                       "(solver.set_state({T: lambda X, Z: -Z + ...})).",
                 reference=_RB_REFERENCE + " Growth rate at Ra = 2500, k = pi, Pr = 1: "
                           "sigma = 5.5425059 (tutorial 2, Rayleigh-Benard: IVP = EVP to 1e-5).")


def rayleigh_benard_evp(*, N: int, Ra: float, Pr: float, k: float) -> Ready:
    """Linear onset of Rayleigh–Benard convection: normal modes exp(i k x + s t) of the
    conduction state, same equations and units as :func:`rayleigh_benard` (z in [-1/2, 1/2])::

        s u - Pr (D^2 - k^2) u + i k p = 0,
        s w - Pr (D^2 - k^2) w + D p - Pr Ra T = 0,
        s T - (D^2 - k^2) T - w = 0,          (dT0/dz = -1)
        i k u + D w = 0,
        u = w = T = 0 at z = -1/2 and z = +1/2.

    Eigenvalue s = growth rate (real at onset). The wavenumber k is a Parameter, so
    ``ParametricEVP(r.problem, [Ra, k])`` and ``critical(...)`` vary it without rebuilding
    (k > 0: k = 0 has no pressure gauge here). Chebyshev(z, N), dealias 1 (linear, constant
    coefficients). Taus as in the IVP: the second velocity tau of w is also lifted into the
    continuity equation (generalized tau).

    Validation reference: Reid & Harris (1958): Ra_c = 1707.762 at k_c = 3.117
    (tests/problems/test_problems_convection.py: critical() over k, N = 24).
    """
    cs = CartesianCoordinates("z")
    z = cs["z"]
    bz = Chebyshev(z, N, (-0.5, 0.5), dealias=1)
    D, D0 = Domain(cs, [bz]), Domain(cs, [])
    u, w, T, p = (Field(n, D, dtype=complex) for n in ("u", "w", "T", "p"))
    tu1, tu2, tw1, tw2, tT1, tT2 = (Field(n, D0, dtype=complex)
                                    for n in ("tu1", "tu2", "tw1", "tw2", "tT1", "tT2"))
    s = Parameter("s", dtype=complex)                          # eigenvalue: growth rate
    Ra_, Pr_, k_ = Parameter("Ra", Ra), Parameter("Pr", Pr), Parameter("k", k)

    def L2(f):                                                 # Laplacian of a normal mode
        return diff(diff(f, z), z) - k_ ** 2 * f

    pb = EVP([u, w, T, p, tu1, tu2, tw1, tw2, tT1, tT2], eigenvalue=s,
             name="Rayleigh-Benard onset")
    pb.add_equation((1j * k_ * u + diff(w, z) + lift(tw2, bz, -1), 0))
    pb.add_equation((s * u - Pr_ * L2(u) + 1j * k_ * p + lift(tu1, bz, -1) + lift(tu2, bz, -2), 0))
    pb.add_equation((s * w - Pr_ * L2(w) + diff(p, z) - Pr_ * Ra_ * T
                     + lift(tw1, bz, -1) + lift(tw2, bz, -2), 0))
    pb.add_equation((s * T - L2(T) - w + lift(tT1, bz, -1) + lift(tT2, bz, -2), 0))
    for f in (u, w, T):                                        # rigid, fixed-temperature plates
        pb.add_equation((interp(f, z, "left"), 0))
        pb.add_equation((interp(f, z, "right"), 0))
    return ready(pb, [Ra_, Pr_, k_], [bz], None,
                 notes="Eigenvalue s = growth rate in thermal-diffusion units.",
                 reference=_RB_REFERENCE)


def _christensen_initial_theta(ri: float, ro: float) -> Callable:
    """Christensen et al. (2001) eq. (9) temperature perturbation, amplitude A = 0.1, m = 4."""
    def theta0(P, T, R):
        x = 2 * R - ri - ro
        return (210 * 0.1 / np.sqrt(17920 * np.pi) * (1 - 3 * x ** 2 + 3 * x ** 4 - x ** 6)
                * np.sin(T) ** 4 * np.cos(4 * P))
    return theta0


@_renames("conatus.problems.convection:rotating_shell_convection")
def rotating_shell_convection(*, Nphi: int, Ntheta: int, Nr: int, eta: float,
                              Ra: float, E: float, Pr: float, dealias: float) -> Ready:
    """Rotating Boussinesq convection in a spherical shell (Christensen et al. 2001 geometry and
    units, non-magnetic), IVP.

    Gap D = r_o - r_i = 1, radius ratio r_i / r_o = ``eta`` (``radius_ratio=`` is its deprecated
    v0.1 name); viscous time D^2/nu; temperature
    perturbation theta = T - T0 about the conduction profile T0 = r_i r_o / r - r_i
    (T = 1 at r_i, 0 at r_o); gravity g ~ r / r_o. Momentum divided by E = nu / (Omega D^2)::

        dt(u) - lap(u) + grad(p)/E - (Ra / (E r_o)) theta r e_r = -u.grad(u) - (2/E) e_z x u,
        div(u) = 0,
        dt(theta) + u.grad(T0) - lap(theta)/Pr = -u.grad(theta),
        u = 0 and theta = 0 at r = r_i and r = r_o (no slip, fixed temperature).

    Ra = alpha g_0 DeltaT D / (nu Omega) (Christensen's modified Rayleigh number), Pr = nu/kappa.
    First-order tau form: tau_1 lifted in the derivative basis ``lam=1`` inside the gradient,
    tau_2 one basis higher (``lam=2``) in the equation. Lifting both in ``lam=1`` gives a spurious
    growing tau mode from N_r ~ 48 (regression: the development test suite).
    The Coriolis term is explicit (RHS): its rate 2/E bounds the time step of an explicit scheme. NCC data
    ``rvec`` (r e_r), ``ez`` (e_z = cos(theta) e_r - sin(theta) e_theta) and ``gT0``
    (grad T0). Bases: RealFourier(phi, Nphi) x SphereBasis(theta, Ntheta) x ShellBasis(r, Nr).
    ``extras["initial_theta"]``: the benchmark's initial perturbation, eq. (9) (m = 4, A = 0.1).

    Validation reference: Christensen et al. (2001), PEPI 128, 25, case 0 (eta = 0.35,
    E = 1e-3, Pr = 1, Ra = 100): E_kin = 58.348 +- 0.050, drift frequency 0.1824 +- 0.005
    (Table 4); this formulation gave E_kin = 58.347, drift 0.1820
    (preliminary). Linear onset at these E, Pr:
    Ra_c = 55.904299, m_c = 4 (see :func:`rotating_shell_convection_evp`).
    """
    ri = eta / (1 - eta)                                       # D = ro - ri = 1
    ro = 1 / (1 - eta)
    sc = SphericalCoordinates("phi", "theta", "r")
    r = sc["r"]
    bphi = RealFourier(sc["phi"], Nphi, (0, 2 * np.pi), dealias=dealias)
    bt = SphereBasis(sc["theta"], Ntheta, dealias=dealias)
    br = ShellBasis(r, Nr, (ri, ro), dealias=dealias)
    D = Domain(sc, [bphi, bt, br])
    u, p, th = VectorField("u", D), Field("p", D), Field("theta", D)
    S = D.without(r)
    tu1, tu2, tT1, tT2 = VectorField("tu1", S), VectorField("tu2", S), Field("tT1", S), Field("tT2", S)
    tp = Field("tp", Domain(sc, []))
    rv = VectorField("rvec", Domain(sc, [br]))                 # r e_r            (NCC)
    ez = VectorField("ez", Domain(sc, [bt]))                   # rotation axis    (NCC)
    gT0 = VectorField("gT0", Domain(sc, [br]))                 # grad T0          (NCC)
    Ra_, E_, Pr_ = Parameter("Ra", Ra), Parameter("E", E), Parameter("Pr", Pr)
    L1u, L1T = lift(tu1, br, -1, lam=1), lift(tT1, br, -1, lam=1)
    pb = IVP([u, p, th, tu1, tu2, tT1, tT2, tp], name="rotating shell convection")
    pb.add_equation((dt(u) - div(grad(u) + outer(rv, L1u)) + grad(p) / E_ + lift(tu2, br, -1, lam=2)
                     - (Ra_ / (E_ * ro)) * (rv * th),
                     -dot(u, grad(u)) - (2 / E_) * cross(ez, u)))
    pb.add_equation((div(u) + dot(rv, L1u) + tp, 0))
    pb.add_equation((dt(th) + dot(u, gT0) - (1 / Pr_) * div(grad(th) + rv * L1T)
                     + lift(tT2, br, -1, lam=2), -dot(u, grad(th))))
    for side in ("left", "right"):
        pb.add_equation((interp(u, r, side), 0))
        pb.add_equation((interp(th, r, side), 0))
    pb.add_equation((integ(p), 0))
    ncc = {rv: lambda P, T, R: np.array([0 * R, 0 * R, R + 0 * P]),
           ez: lambda P, T, R: np.array([0 * T, -np.sin(T), np.cos(T) + 0 * P]),
           gT0: lambda P, T, R: np.array([0 * R, 0 * R, -ri * ro / R ** 2 + 0 * P])}
    return ready(pb, [Ra_, E_, Pr_], [bphi, bt, br], ncc,
                 notes=f"r_i = {ri:.6g}, r_o = {ro:.6g} (D = 1). Coriolis explicit: rate 2/E. "
                       "Vector components are (phi, theta, r).",
                 reference="Christensen et al. (2001) case 0 (0.35, E = 1e-3, Pr = 1, Ra = 100): "
                           "E_kin = 58.348 +- 0.050, drift 0.1824 +- 0.005 (Table 4).",
                 extras={"initial_theta": _christensen_initial_theta(ri, ro), "ri": ri, "ro": ro})


@_renames("conatus.problems.convection:rotating_shell_convection_evp")
def rotating_shell_convection_evp(*, Nr: int, Lmax: int, m: int, eta: float,
                                  Ra: float, E: float, Pr: float) -> Ready:
    """Linear onset of rotating shell convection: normal modes exp(i m phi + s t) about the
    conduction state, equations and units of :func:`rotating_shell_convection` multiplied back
    by E::

        E (s u - lap u) + 2 e_z x u + grad p - (Ra / r_o) T r e_r = 0,   div u = 0,
        s T + u.grad(T0) - lap(T)/Pr = 0,   u = T = 0 at r_i and r_o.

    Eigenvalue s in viscous units (Im s / (1/E) = drift frequency in units of Omega). Only the
    group ``m`` is meaningful: the phi basis is ComplexFourier(phi, 2|m| + 4) (structural: large
    enough to hold m); SphereBasis(theta, Lmax + 1), ShellBasis(r, Nr), dealias 1 (linear). Use
    ``plan.group(m)``; e_z couples all degrees l of one m.

    Validation reference: eta = 0.35, E = 1e-3, Pr = 1: m_c = 4, Ra_c = 55.904299,
    omega_c = -0.0230918 Omega at (Nr, Lmax) = (32, 48)
    (Skene & Tobias 2025 Table 1: 55.9, 4, -0.0231; Barik et al. 2023: 55.904296).
    """
    ri = eta / (1 - eta)
    ro = 1 / (1 - eta)
    sc = SphericalCoordinates("phi", "theta", "r")
    r = sc["r"]
    bphi = ComplexFourier(sc["phi"], 2 * abs(m) + 4, (0, 2 * np.pi), dealias=1)
    bt = SphereBasis(sc["theta"], Lmax + 1, dealias=1)
    br = ShellBasis(r, Nr, (ri, ro), dealias=1)
    D = Domain(sc, [bphi, bt, br])
    u, p, T = VectorField("u", D, dtype=complex), Field("p", D, dtype=complex), Field("T", D, dtype=complex)
    S = D.without(r)
    tu1, tu2 = VectorField("tu1", S, dtype=complex), VectorField("tu2", S, dtype=complex)
    tT1, tT2 = Field("tT1", S, dtype=complex), Field("tT2", S, dtype=complex)
    tp = Field("tp", Domain(sc, []), dtype=complex)
    rv = VectorField("rvec", Domain(sc, [br]))
    ez = VectorField("ez", Domain(sc, [bt]))
    gT0 = VectorField("gT0", Domain(sc, [br]))
    s = Parameter("s", dtype=complex)
    Ra_, E_, Pr_ = Parameter("Ra", Ra), Parameter("E", E), Parameter("Pr", Pr)
    L1u, L1T = lift(tu1, br, -1, lam=1), lift(tT1, br, -1, lam=1)
    pb = EVP([u, p, T, tu1, tu2, tT1, tT2, tp], eigenvalue=s, name="rotating shell onset")
    pb.add_equation((E_ * (s * u - div(grad(u) + outer(rv, L1u))) + 2 * cross(ez, u) + grad(p)
                     - (Ra_ / ro) * (rv * T) + lift(tu2, br, -1, lam=2), 0))
    pb.add_equation((div(u) + dot(rv, L1u) + tp, 0))
    pb.add_equation((s * T + dot(u, gT0) - (1 / Pr_) * div(grad(T) + rv * L1T)
                     + lift(tT2, br, -1, lam=2), 0))
    for side in ("left", "right"):
        pb.add_equation((interp(u, r, side), 0))
        pb.add_equation((interp(T, r, side), 0))
    pb.add_equation((integ(p), 0))
    ncc = {rv: lambda P, Th, R: np.array([0 * R, 0 * R, R + 0 * P]),
           ez: lambda P, Th, R: np.array([0 * Th, -np.sin(Th), np.cos(Th) + 0 * P]),
           gT0: lambda P, Th, R: np.array([0 * R, 0 * R, -ri * ro / R ** 2 + 0 * P])}
    return ready(pb, [Ra_, E_, Pr_], [bphi, bt, br], ncc,
                 notes=f"r_i = {ri:.6g}, r_o = {ro:.6g}. Use plan.group({m}).",
                 reference="0.35, E = 1e-3, Pr = 1: m_c = 4, Ra_c = 55.904299, omega_c = "
                           "-0.0230918 Omega (Skene & Tobias 2025; Barik et al. 2023).",
                 extras={"m": m, "ri": ri, "ro": ro})


# 0.1 showed the decorator as `renames`: it still resolves, with a DeprecationWarning (L8).
__getattr__ = _former_names(__name__, {"renames": ("conatus._deprecation.renames", _renames)})
