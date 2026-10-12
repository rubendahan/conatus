"""Onset of rotating Rayleigh–Bénard convection on the f-plane (Chandrasekhar 1961, Ch. III).

A Boussinesq layer 0 < z < 1 heated from below rotates about the vertical with Taylor number
Ta = (2 Omega d^2 / nu)^2. Units: depth d, thermal diffusion time d^2 / kappa. A normal mode
exp(i k x + sigma t) independent of y (the horizontal plane is isotropic, so this loses no
generality) has all three velocity components; with D = d/dz and Pr = nu / kappa:

    sigma u = Pr (D^2 - k^2) u - i k p + Pr sqrt(Ta) v          (Coriolis: -2 Omega x u)
    sigma v = Pr (D^2 - k^2) v         - Pr sqrt(Ta) u
    sigma w = Pr (D^2 - k^2) w - D p   + Pr Ra T
    sigma T =    (D^2 - k^2) T + w                              (linear conduction profile)
    0       = i k u + D w

Free (stress-free) walls: w = Du = Dv = T = 0. Onset is stationary for Pr > 0.6766 and then
(Chandrasekhar 1961, §26, closed form)

    Ra(k) = [(pi^2 + k^2)^3 + pi^2 Ta] / k^2,   Ra_c = min_k Ra(k).

Rigid walls (u = v = w = T = 0) have no closed form (Chandrasekhar's Table, Ch. III); we report
our values with their convergence in N.

Run:  python examples/convection/rotating_rayleigh_benard.py
"""
import numpy as np
import scipy.optimize as so

from conatus.expert import EVP, Parameter, ParametricEVP, channel, critical, diff, interp, sqrt


def rotating_layer(N: int, walls: str, Pr: float = 1.0):
    """Linear stability pencil in (Ra, k, Ta) at fixed Pr; walls = 'free' or 'rigid'. The Coriolis
    parameter sqrt(Ta) is not a Laurent polynomial in Ta: it is declared as a derived pencil
    coordinate, so one pencil serves every Ta (it was rebuilt per Ta before)."""
    ch = channel(N, base=None, interval=(0.0, 1.0))            # Chebyshev in z on [0, 1]
    z = ch.y
    u, v, w, p, T = (ch.field(n) for n in ("u", "v", "w", "p", "T"))
    sigma = Parameter("sigma", dtype=complex)
    Ra, k, Ta = Parameter("Ra", 1e3), Parameter("k", 3.0), Parameter("Ta", 1e3)
    f = sqrt(Ta)                                                # Coriolis parameter in these units
    Dz = lambda q: diff(q, z)                                   # noqa: E731
    lap = lambda q: Dz(Dz(q)) - k ** 2 * q                      # noqa: E731
    pb = EVP([u, v, w, p, T], eigenvalue=sigma, name=f"rotating RB ({walls} walls)")
    pb.add_equation((sigma * u - Pr * lap(u) + 1j * k * p - Pr * f * v, 0))
    pb.add_equation((sigma * v - Pr * lap(v) + Pr * f * u, 0))
    pb.add_equation((sigma * w - Pr * lap(w) + Dz(p) - Pr * Ra * T, 0))
    pb.add_equation((sigma * T - lap(T) - w, 0))
    pb.add_equation((1j * k * u + Dz(w), 0))
    # two conditions per second-order equation; add_bc lifts their taus into modes N-1, N-2
    wall = (lambda q: Dz(q)) if walls == "free" else (lambda q: q)   # stress-free or no-slip
    for i, cond in enumerate((wall(u), wall(v), w, T)):
        pb.add_bc(interp(cond, z, 0.0), interp(cond, z, 1.0), into=i)
    return ParametricEVP(pb, [Ra, k, Ta], derived=[f]), pb


def ra_free_exact(Ta: float) -> tuple[float, float]:
    """Chandrasekhar's closed form for free walls: (Ra_c, k_c). dRa/dk = 0 gives, with x = k^2,
    2 x^3 + 3 pi^2 x^2 - pi^6 - pi^2 Ta = 0 (one positive root)."""
    x = so.brentq(lambda x: 2 * x ** 3 + 3 * np.pi ** 2 * x ** 2 - np.pi ** 6 - np.pi ** 2 * Ta,
                  1e-6, 1e4, xtol=1e-15)
    return ((np.pi ** 2 + x) ** 3 + np.pi ** 2 * Ta) / x, float(np.sqrt(x))


_PENCILS: dict = {}


def onset(N: int, Ta: float, walls: str, k_bounds: tuple[float, float],
          ra_bracket: tuple[float, float]):
    """Critical Ra minimized over k: neutral curve by tracked Newton on Re(sigma), Brent in k.
    One pencil per (N, walls), reused for every Ta."""
    if (N, walls) not in _PENCILS:
        _PENCILS[N, walls] = rotating_layer(N, walls)[0]
    P = _PENCILS[N, walls]
    # the stationary onset mode is the least stable one; critical() re-checks it by a full QZ
    return critical(P, "Ra", bracket=ra_bracket, growth="real", over={"k": k_bounds},
                    values={"Ta": Ta}, xtol=1e-11), P


if __name__ == "__main__":
    print("Free walls: Ra_c(Ta) against Chandrasekhar's closed form (Pr = 1, stationary onset)")
    for Ta in (0.0, 1e2, 1e3, 1e4, 1e5):
        ex, kx = ra_free_exact(Ta)
        for N in (16, 24):
            cp, P = onset(N, Ta, "free", (0.8 * kx, 1.25 * kx), (0.8 * ex, 1.25 * ex))
            print(f"  Ta={Ta:8.0f} N={N}: Ra_c = {cp.value:.10f} (exact {ex:.10f}, rel. {cp.value / ex - 1:+.1e}), "
                  f"k_c = {cp.over['k']:.6f} (exact {kx:.6f}), sigma = {cp.mode.sigma:.1e}, residual {cp.mode.residual:.1e}")
    print(cp.explain())
    print("Rigid walls (no closed form; to compare with Chandrasekhar's table), convergence in N:")
    for Ta in (0.0, 1e2, 1e3, 1e4, 1e5):
        ex, kx = ra_free_exact(Ta)                  # free-wall values only bracket the search
        kb = max(kx, 3.1)
        for N in (24, 32, 48):
            cp, _ = onset(N, Ta, "rigid", (0.7 * kb, 1.3 * kb), (0.5 * ex, 3.0 * ex))
            print(f"  Ta={Ta:8.0f} N={N}: Ra_c = {cp.value:.8f}, k_c = {cp.over['k']:.6f}, residual {cp.mode.residual:.1e}")
