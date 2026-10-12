"""Kelvin–Helmholtz instability of the tanh mixing layer (Michalke 1964), inviscid and viscous.

Base flow U(y) = tanh(y) (lengths in units of the half-thickness, velocities in units of the
half velocity difference). A normal mode v(y) exp(i alpha (x - c t)) of the cross-stream velocity
obeys the Rayleigh equation (inviscid) or the Orr–Sommerfeld equation (viscous):

    (U - c)(v'' - alpha^2 v) - U'' v = (1 / (i alpha Re)) (v'''' - 2 alpha^2 v'' + alpha^4 v)

with the temporal growth rate omega_i = alpha Im(c). Michalke (1964, J. Fluid Mech. 19, 543-556,
Table 1 / Fig. 3) finds the maximum omega_i = 0.1897 at alpha = 0.4446 (c_r = 0 by symmetry).

The unbounded layer is truncated to y in [-L, L] with v = 0 (and v' = 0 when viscous) at y = +-L.
The eigenfunction decays like exp(-alpha |y|), so the truncation error is ~exp(-2 alpha L); both
L and N are explicit and their convergence is printed.

Run:  python examples/shear/kelvin_helmholtz.py
"""
import numpy as np
import scipy.optimize as so

from conatus.expert import EVP, Parameter, ParametricEVP, channel, diff, interp, verify

MICHALKE = {"alpha": 0.4446, "omega_i": 0.1897}      # Michalke (1964), Table 1


def mixing_layer(N: int, L: float, viscous: bool):
    """Rayleigh (viscous=False) or Orr–Sommerfeld (viscous=True) pencil in alpha (and Re)."""
    ch = channel(N, base=None, interval=(-L, L))                # Chebyshev in y on [-L, L]
    y, v, U = ch.y, ch.field("v"), ch.field("U")
    c, alpha, Re = Parameter("c", dtype=complex), Parameter("alpha", 0.44), Parameter("Re", 1e3)

    def L2(f):                                                  # d^2/dy^2 - alpha^2
        return diff(diff(f, y), y) - alpha ** 2 * f

    Upp = diff(diff(U, y), y)                                   # U'' from the NCC, not by hand
    # (U - c) L2 v - U'' v [- L2 L2 v / (i alpha Re)] = 0; the eigenvalue c multiplies L2 v
    rayleigh = U * L2(v) - c * L2(v) - Upp * v
    pb = EVP([v], eigenvalue=c, name="Orr-Sommerfeld (tanh)" if viscous else "Rayleigh (tanh)")
    if viscous:
        pb.add_equation((rayleigh - L2(L2(v)) / (1j * alpha * Re), 0))
        pb.add_bc(interp(v, y, -L), interp(v, y, L), interp(diff(v, y), y, -L), interp(diff(v, y), y, L))
        params = [alpha, Re]
    else:
        pb.add_equation((rayleigh, 0))
        pb.add_bc(interp(v, y, -L), interp(v, y, L))
        params = [alpha]
    # U'' is computed from the sampled tanh: on [-15, 15] at N = 192 its Chebyshev tail is 1.1e-5
    # (the interpolation error of tanh, 1e-10, amplified by d^2/dy^2; largest at |y| = L where
    # U'' ~ 0 and the mode ~ exp(-alpha L)); accepted explicitly up to 2e-5. The gallery
    # test checks the growth rate against Michalke, and __main__ its convergence in N and L.
    return ParametricEVP(pb, params, ncc={U: np.tanh}, ncc_tail_tol=2e-5), pb


def most_unstable(P: ParametricEVP, **fixed):
    """max over alpha of omega_i = alpha Im(c): full QZ once, then the KH mode is tracked in alpha."""
    # KH mode = largest Im(c) among |c| < 2 (|U| <= 1 bounds physical phase speeds, Howard's
    # semicircle); the viscous tau-spurious modes have |c| ~ 1e3 and are excluded (count logged)
    seed = P.least_stable(growth="imag", select=lambda s: abs(s) < 2, alpha=0.44, **fixed)

    def minus_omega_i(a):
        m = P.track(seed.sigma, seed.x, left=False, alpha=a, **fixed)
        return -a * m.sigma.imag

    opt = so.minimize_scalar(minus_omega_i, bounds=(0.40, 0.49), method="bounded",
                             options={"xatol": 1e-7})
    mode = P.track(seed.sigma, seed.x, alpha=float(opt.x), **fixed)
    return float(opt.x), mode


if __name__ == "__main__":
    print("Inviscid (Rayleigh), convergence in N and L:")
    for L, N in [(10, 192), (15, 192), (15, 256), (20, 384)]:
        P, pb = mixing_layer(N, L, viscous=False)
        a, m = most_unstable(P)
        print(f"  L={L:2d} N={N:3d}: alpha_max = {a:.6f}, omega_i = {a * m.sigma.imag:.8f}, "
              f"c = {m.sigma:.2e}, residual {m.residual:.1e}")
    print(f"  Michalke (1964): alpha_max = {MICHALKE['alpha']}, omega_i = {MICHALKE['omega_i']}")
    print(pb.explain())
    print(verify(P, m).explain())
    print("Viscous (Orr–Sommerfeld), L=15, N=256: omega_i -> inviscid value as Re -> infinity")
    for Re in (1e2, 1e3, 1e4):
        P, _ = mixing_layer(256, 15, viscous=True)
        a, m = most_unstable(P, Re=Re)
        print(f"  Re = {Re:7.0f}: alpha_max = {a:.6f}, omega_i = {a * m.sigma.imag:.8f}, residual {m.residual:.1e}")
