"""Salt fingers: stationary double-diffusive onset in a layer with free walls (Stern 1960;
Baines & Gill 1969, J. Fluid Mech. 37, 289-306).

Warm salty water above cold fresh water: temperature is stabilizing (Rayleigh number R_T > 0)
and salt, which diffuses tau = kappa_S / kappa_T < 1 times slower, is destabilizing (R_S > 0).
Units: depth d, thermal diffusion time d^2 / kappa_T; normal mode exp(i k x + sigma t), D = d/dz:

    sigma u = Pr (D^2 - k^2) u - i k p
    sigma w = Pr (D^2 - k^2) w - D p + Pr (R_T T - R_S S)       (heat and salt both make it lighter)
    sigma T =     (D^2 - k^2) T - w                              (stable temperature gradient)
    sigma S = tau (D^2 - k^2) S - w                              (unstable salinity gradient)
    0       = i k u + D w

Free walls: w = Du = T = S = 0. With sin(pi z) modes and sigma = 0, the stationary (finger)
threshold is  R_S / tau - R_T = min_k (pi^2 + k^2)^3 / k^2 = 27 pi^4 / 4  at k^2 = pi^2 / 2,
independent of Pr (Baines & Gill 1969, §2).

Run:  python examples/convection/salt_fingers.py
"""
import numpy as np

from conatus.expert import EVP, Parameter, ParametricEVP, channel, critical, diff, interp

RA_FREE = 27 * np.pi ** 4 / 4          # 657.511..., the free-wall Rayleigh–Bénard number


def finger_layer(N: int, R_T: float, tau: float, Pr: float = 7.0):
    """Linear stability pencil in (R_S, k) at fixed R_T, tau, Pr (free walls)."""
    ch = channel(N, base=None, interval=(0.0, 1.0))            # Chebyshev in z on [0, 1]
    z = ch.y
    u, w, p, T, S = (ch.field(n) for n in ("u", "w", "p", "T", "S"))
    sigma = Parameter("sigma", dtype=complex)
    R_S, k = Parameter("R_S", 1e3), Parameter("k", 2.2)
    Dz = lambda q: diff(q, z)                                   # noqa: E731
    lap = lambda q: Dz(Dz(q)) - k ** 2 * q                      # noqa: E731
    pb = EVP([u, w, p, T, S], eigenvalue=sigma, name=f"salt fingers (R_T={R_T:g}, tau={tau:g}, Pr={Pr:g})")
    pb.add_equation((sigma * u - Pr * lap(u) + 1j * k * p, 0))
    pb.add_equation((sigma * w - Pr * lap(w) + Dz(p) - Pr * (R_T * T - R_S * S), 0))
    pb.add_equation((sigma * T - lap(T) + w, 0))
    pb.add_equation((sigma * S - tau * lap(S) + w, 0))
    pb.add_equation((1j * k * u + Dz(w), 0))
    # free walls; add_bc lifts the two taus of each second-order equation into modes N-1, N-2
    for i, cond in enumerate((Dz(u), w, T, S)):
        pb.add_bc(interp(cond, z, 0.0), interp(cond, z, 1.0), into=i)
    return ParametricEVP(pb, [R_S, k]), pb


def finger_onset(N: int, R_T: float, tau: float, Pr: float = 7.0):
    """Critical R_S (growth Re sigma = 0) minimized over k; exact: tau (R_T + 27 pi^4 / 4)."""
    P, _ = finger_layer(N, R_T, tau, Pr)
    ex = tau * (R_T + RA_FREE)
    return critical(P, "R_S", bracket=(0.8 * ex, 1.25 * ex), growth="real",
                    over={"k": (1.8, 2.7)}, xtol=1e-11), ex


if __name__ == "__main__":
    print("Stationary finger threshold R_S,c = tau (R_T + 27 pi^4 / 4), k_c = pi / sqrt(2)")
    for R_T, tau in [(0.0, 0.01), (1e3, 0.01), (1e4, 0.1), (1e5, 1 / 3)]:
        for N in (16, 24):
            cp, ex = finger_onset(N, R_T, tau)
            print(f"  R_T={R_T:8.0f} tau={tau:.3f} N={N}: R_S,c = {cp.value:.8f} (exact {ex:.8f}, "
                  f"rel. {cp.value / ex - 1:+.1e}), k_c = {cp.over['k']:.6f} "
                  f"(exact {np.pi / np.sqrt(2):.6f}), residual {cp.mode.residual:.1e}")
    print(cp.explain())
