"""Non-modal stability of plane Poiseuille flow: transient growth and the continuous adjoint.

Base flow U(y) = 1 − y² between no-slip walls y = ±1 (centreline Reynolds number Re). One Fourier
mode e^{i(kx x + kz z)} of the linearized Navier–Stokes equations in primitive variables:

    dt u = (Δ/Re − i kx U) u − U' v − i kx p,   dt v = ... − ∂y p,   dt w = ... − i kz p,
    0 = i kx u + ∂y v + i kz w,                 Δ = ∂y² − kx² − kz².

Every norm is the kinetic energy E = ½ ∫ (|u|² + |v|² + |w|²) dy with the exact Chebyshev Gram.
Butler & Farrell (1992, Phys. Fluids A 4, 1637) and Reddy & Henningson (1993, JFM 252, 209):
at Re = 5000, (kx, kz) = (0, 2.044), the energy can grow by G_max = 4897 at t = 379 although
every eigenmode decays. The streaks come from the lift-up coupling −U' v (Squire): non-normality.

Run:  python examples/shear/nonmodal_poiseuille.py
"""
import numpy as np

from conatus.expert import Y, compare_adjoints, max_growth, parallel_shear_flow, state_space

BUTLER_FARRELL = {"G_max": 4897, "t_max": 379}       # Re = 5000, kx = 0, kz = 2.044


def generator(Re, kx, kz, N):
    """Energy-orthonormal generator of one mode (QZ reduction of the tau DAE, gap reported)."""
    system = parallel_shear_flow(1 - Y ** 2, kx=kx, kz=kz, Re=Re, walls="no-slip")
    return system, state_space(system.discretize(N), infinite_tol=1e-10)


def optimal_growth(Re=5000.0, kx=0.0, kz=2.044, N=64):
    _, ss = generator(Re, kx, kz, N)
    return max_growth(ss, (300.0, 460.0), xtol=1e-6, n_scan=40)


if __name__ == "__main__":
    # 1. transient growth: every mode decays, yet the energy grows ~5000-fold
    for N in (48, 64):
        G, t = optimal_growth(N=N)
        print(f"N = {N}: G_max = {G:.6f} at t = {t:.4f}   (Butler & Farrell: "
              f"{BUTLER_FARRELL['G_max']} at t = {BUTLER_FARRELL['t_max']})")
    system, ss = generator(5000.0, 0.0, 2.044, 64)
    print(ss.explain())
    print(f"largest growth rate {np.linalg.eigvals(ss.A).real.max():.3e} < 0: asymptotically stable")

    # 2. continuous (optimize-then-discretize) versus discrete adjoint of the TS mode
    ts = parallel_shear_flow(1 - Y ** 2, kx=1.0, kz=0.0, Re=2000.0, walls="no-slip")
    adj = ts.formal_adjoint(ts.bcs)          # declared adjoint BCs, Green concomitant checked
    print(adj.show_equations())
    print(ts.show_green_identity(adj))
    for N in (32, 48, 64):
        print(compare_adjoints(ts, ts.bcs, N, -0.02 - 0.312j, infinite_tol=1e-10).explain())
