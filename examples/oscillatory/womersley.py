"""Womersley flow in a plane channel: exact profile (LBVP and IVP) and Floquet stability (Hill).

1. Oscillating pressure gradient. Lengths in half-widths h, time in 1/omega, velocity in P/omega
   (P: amplitude of -dp/dx / rho), Womersley number Wo = h sqrt(omega / nu):

       u_t - u_yy / Wo^2 = cos t,        u(-1) = u(1) = 0.

   The periodic solution is u = Re[U(y) e^{it}] with U = -i (1 - cosh(k y) / cosh k), k = sqrt(i) Wo
   (the plane-channel form of Womersley 1955, J. Physiol. 127, 553-563, eq. for the pipe with
   replaced by cosh). The harmonic U is an LBVP (i U - U'' / Wo^2 = 1) solved to round-off; the IVP
   started from the exact state is compared after one period (temporal order of the scheme).

2. Pulsatile Poiseuille flow (Orr–Sommerfeld units: centreline velocity 1, Re = U_c h / nu):
   U(y, t) = 1 - y^2 + eps Re[W(y) e^{i omega t}], W the same Womersley profile normalized to
   W(0) = 1. Floquet modes v = e^{mu t} sum_n v_n(y) e^{i n omega t} of

       (d/dt + i alpha U) (v'' - alpha^2 v) - i alpha U'' v - (v'''' - 2 alpha^2 v'' + alpha^4 v) / Re = 0

   come from the general Hill operator (floquet.hill_from_plan) built on the same IR problem as the
   steady Orr–Sommerfeld EVP. Checks: eps = 0 gives Orszag's eigenvalue (1971, J. Fluid Mech. 50,
   689-703, Table 3: c = 0.23752649 + 0.00373967 i at Re = 1e4, alpha = 1), and the shift
   mu(eps) - mu(0) is O(eps^2) (the harmonics +-1 only act at second order). A published Floquet
   value for this flow (von Kerczek 1982; Pier & Schmid 2017) is not reproduced yet.

Run:  python examples/oscillatory/womersley.py
"""
import numpy as np

from conatus.expert import (EVP, IVP, LBVP, IVPSolver, Parameter, channel, cos, diff, dt,
                            discretize, grid, hill_from_plan, interp, secant_root)

ORSZAG_C = 0.23752649 + 0.00373967j          # Orszag (1971), Re = 1e4, alpha = 1


def womersley_exact(Y, t, Wo):
    """u(y, t) = Re[-i (1 - cosh(k y)/cosh k) e^{it}], k = sqrt(i) Wo."""
    k = np.sqrt(1j) * Wo
    return (-1j * (1 - np.cosh(k * Y) / np.cosh(k)) * np.exp(1j * t)).real


def harmonic_lbvp(N: int, Wo: float):
    """i U - U'' / Wo^2 = 1, U(+-1) = 0 (complex LBVP); returns (y grid, U, max residual)."""
    ch = channel(N, base=None)
    y, U = ch.y, ch.field("U", dtype=complex)
    pb = LBVP([U], name="Womersley harmonic")
    pb.add_equation((1j * U - diff(diff(U, y), y) / Wo ** 2, 1))
    pb.add_bc(interp(U, y, -1), interp(U, y, 1))
    sol = discretize(pb).solve()
    return grid(ch.basis, dealias=False), sol.grid("U"), sol.max_residual


def oscillating_channel(N: int, Wo: float):
    """IVP u_t - u_yy / Wo^2 = cos t with no-slip walls; t enters the RHS as a parameter."""
    ch = channel(N, base=None)
    y, u, t = ch.y, ch.field("u"), Parameter("t", 0.0)
    pb = IVP([u], name="oscillating channel")
    pb.add_equation((dt(u) - diff(diff(u, y), y) / Wo ** 2, cos(t)))
    pb.add_bc(interp(u, y, -1), interp(u, y, 1))
    return discretize(pb), u, grid(ch.basis, dealias=False)


def one_period_error(N: int, Wo: float, scheme: str, steps: int) -> float:
    plan, u, Y = oscillating_channel(N, Wo)
    s = IVPSolver(plan, scheme, 2 * np.pi / steps, time_parameter="t")
    s.set_state({u: lambda y: womersley_exact(y, 0.0, Wo)})
    s.run(steps)
    return float(np.abs(s.grid("u") - womersley_exact(Y, s.t, Wo)).max())


def pulsatile_hill(N: int, Re: float, alpha: float, omega: float, eps: float, Nh: int):
    """Hill operator of Orr–Sommerfeld around U = 1 - y^2 + eps Re[W e^{i omega t}]."""
    ch = channel(N, base=None)
    y, v, U = ch.y, ch.field("v"), ch.field("U")
    s = Parameter("s", dtype=complex)                           # growth rate (EVP convention)

    def L2(f):
        return diff(diff(f, y), y) - alpha ** 2 * f

    pb = EVP([v], eigenvalue=s, name="Orr-Sommerfeld (growth rate)")
    pb.add_equation((s * L2(v) + 1j * alpha * U * L2(v) - 1j * alpha * diff(diff(U, y), y) * v
                     - L2(L2(v)) / Re, 0))
    pb.add_bc(interp(v, y, -1), interp(v, y, 1), interp(diff(v, y), y, -1), interp(diff(v, y), y, 1))
    k = np.sqrt(1j * omega * Re)                                # Wo = sqrt(omega Re) in these units
    W = lambda Y: (1 - np.cosh(k * Y) / np.cosh(k)) / (1 - 1 / np.cosh(k))   # noqa: E731, W(0) = 1
    harmonics = {0: lambda Y: 1 - Y ** 2 + 0j,                  # steady Poiseuille part
                 1: lambda Y: 0.5 * eps * W(Y), -1: lambda Y: 0.5 * eps * np.conj(W(Y))}
    return hill_from_plan(lambda ncc: discretize(pb, ncc=ncc), {U: harmonics}, Nh=Nh,
                          omega=omega, label=f"pulsatile Poiseuille eps={eps}")


def floquet_exponent(eps: float, *, N=64, Re=1e4, alpha=1.0, omega=0.1, Nh=8):
    """Floquet exponent continued from the steady Orr–Sommerfeld mode (seed mu = -i alpha c)."""
    H = pulsatile_hill(N, Re, alpha, omega, eps, Nh)
    root = secant_root(H, -1j * alpha * ORSZAG_C, tol=1e-12, max_iterations=60)  # raises if unconverged
    return root.mu, H


if __name__ == "__main__":
    Wo = 4.0
    print(f"1. Womersley harmonic (LBVP), Wo = {Wo}:")
    for N in (16, 24, 32):
        Y, U, res = harmonic_lbvp(N, Wo)
        ex = -1j * (1 - np.cosh(np.sqrt(1j) * Wo * Y) / np.cosh(np.sqrt(1j) * Wo))
        print(f"   N={N}: max|U - U_exact| = {np.abs(U - ex).max():.1e}, residual {res:.1e}")
    assert np.abs(U - ex).max() < 5e-15 and res < 1e-14         # Womersley (1955), closed form
    print("   IVP over one period from the exact state (N = 32): error vs steps")
    for scheme in ("RK443", "SBDF2"):
        errs = [one_period_error(32, Wo, scheme, n) for n in (100, 200, 400)]
        orders = np.log2(np.array(errs[:-1]) / errs[1:])
        print(f"   {scheme}: " + ", ".join(f"{e:.2e}" for e in errs) + f"  observed orders {np.round(orders, 2)}")
        assert orders[-1] > {"RK443": 2.6, "SBDF2": 1.75}[scheme], (scheme, orders)   # 2.85, 1.95 observed
    print("2. Pulsatile Poiseuille, Floquet (Re = 1e4, alpha = 1, omega = 0.1):")
    mu0, H = floquet_exponent(0.0)
    print(f"   eps = 0: mu = {mu0:.10f}, c = i mu / alpha = {1j * mu0:.8f} (Orszag {ORSZAG_C})")
    assert abs(1j * mu0 - ORSZAG_C) < 1e-8, mu0                   # Orszag (1971), his 8 digits
    for eps in (0.0125, 0.025, 0.05):
        mu, _ = floquet_exponent(eps)
        print(f"   eps = {eps}: mu = {mu:.10f}, |mu - mu0| / eps^2 = {abs(mu - mu0) / eps ** 2:.4e}")
    for Nh in (4, 8, 12):
        print(f"   eps = 0.05, Nh = {Nh}: mu = {floquet_exponent(0.05, Nh=Nh)[0]:.12f}")
    for N in (48, 64, 80):
        print(f"   eps = 0.05, N = {N}: mu = {floquet_exponent(0.05, N=N)[0]:.12f}")
    print(H.explain())
