"""Stratified Kelvin–Helmholtz: Taylor–Goldstein equation for the tanh/tanh layer (Hazel 1972).

Base state (half-thickness and half velocity difference as units): U(y) = tanh(y) and buoyancy
frequency N^2(y) = J sech^2(y) (density profile tanh(y), same thickness as the shear layer: Hazel's
R = 1). The local Richardson number Ri(y) = N^2 / U'^2 = J cosh^2(y) is smallest at the centre, so J
is the minimum Richardson number. Linearized inviscid Boussinesq equations for a normal mode
exp(i alpha (x - c t)), written for the cross-stream velocity v and the buoyancy b:

    (U - c)(v'' - alpha^2 v) - U'' v - i alpha b = 0        (vorticity)
    i alpha (U - c) b + N^2 v                  = 0        (buoyancy: b_t + U b_x + N^2 v = 0)

Eliminating b gives the Taylor–Goldstein equation (U - c)(v'' - a^2 v) - U'' v + N^2 v / (U - c) = 0;
the pair above keeps the eigenvalue c linear. Miles (1961) and Howard (1961): no instability when
Ri >= 1/4 everywhere. For R = 1 the neutral curve is J = alpha (1 - alpha) (Hazel 1972, J. Fluid
Mech. 51, 39-61, R = 1 case; also Drazin & Reid 2004, section 44), whose maximum J = 1/4 at
alpha = 1/2 makes the Miles–Howard bound sharp.

Numerics (the whole line, no truncation, no extrapolation):
- y in (-inf, inf) through the tanh map y = L artanh(xi) (MappedChebyshev, ir/maps.py): the metric
  dxi/dy = (1 - xi^2)/L is a polynomial, so the operators stay banded. The far field e^{-alpha|y|}
  becomes (1 -+ xi)^{alpha L / 2}; L = 2 / alpha makes it analytic (declared choice; convergence in
  L is shown by the study).
- Near neutrality c_i -> 0 and the critical layer U = c reaches the real axis: on the real line the
  mode becomes singular and the discretized continuous spectrum (U(y) = c) mixes with it.
  The path is therefore shifted into the complex plane, y(xi) - i delta (``contour``):
  the unstable mode's critical point atanh(c) lies above the path, so its eigenvalue is unchanged
  (Cauchy), while the continuous spectrum moves below the real axis. The mode is then followed
  analytically *through* c_i = 0 and the neutral J is a root of c_i(J), found by Newton
  (critical()), converged in N, in L and in delta: no extrapolation.
- Admissible modes (declared select): Howard's semicircle |c| <= 1, and |c_r| < 1/2 to stay away from
  the ends U = +-1 of the continuous spectrum.

Run:  python examples/shear/taylor_goldstein.py            (quick, ~1 min)
      python examples/shear/taylor_goldstein.py --study    (N up to 192, L and delta: via Slurm)
"""
import sys

import numpy as np

from conatus.expert import (EVP, CartesianCoordinates, Domain, Field, MappedChebyshev, Parameter,
                            ParametricEVP, TanhMap, critical, diff, interp)


def admissible(c):
    """Howard's semicircle (an unstable mode has |c| <= max|U| = 1), and away from the ends of the
    continuous spectrum U = +-1 (reached only at y = +-inf, where the shifted path meets the real
    axis: discretized-continuum modes accumulate there with |c_i| -> 0 as N grows). The unstable
    modes of this symmetric layer are stationary (c_r = 0)."""
    return (np.abs(c) < 1) & (np.abs(c.real) < 0.5)


def taylor_goldstein(N: int, alpha: float, *, L: float | None = None, contour: float = 0.4):
    """Pencil in (J, alpha) on the whole line (tanh map, scale L = 2/alpha unless given)."""
    L = 2 / alpha if L is None else L
    cs = CartesianCoordinates("y")
    y = cs["y"]
    D = Domain(cs, [MappedChebyshev(y, N, TanhMap(L, contour=contour), dealias=1)])
    v, b, U, S = Field("v", D), Field("b", D), Field("U", D), Field("S", D)   # S = sech^2 y
    c = Parameter("c", dtype=complex)
    J, a = Parameter("J", 0.2), Parameter("alpha", alpha)

    def L2(f):
        return diff(diff(f, y), y) - a ** 2 * f

    pb = EVP([v, b], eigenvalue=c, name="Taylor-Goldstein (tanh/tanh, R = 1)")
    pb.add_equation(((U - c) * L2(v) - diff(diff(U, y), y) * v - 1j * a * b, 0))
    pb.add_equation((1j * a * (U - c) * b + J * S * v, 0))
    pb.add_bc(interp(v, y, -np.inf), interp(v, y, np.inf), into=0)   # decay: taus in vorticity eq.
    # 1 - tanh^2 rather than 1/cosh^2: cosh overflows on the mapped (complex) grid (refused)
    # tanh and sech^2 on the mapped line are resolved to 2e-8 (alpha = 0.5) .. 2e-6 (alpha = 0.3,
    # L = 6.7) at N = 96, and U'' (computed from the sampled tanh, so its interpolation error
    # amplified) to 4e-5: accepted explicitly (the gallery tests check convergence in N and L:
    # c_i drifts by 1.2e-7 from N = 96 to 128)
    return ParametricEVP(pb, [J, a], ncc={U: np.tanh, S: lambda Y: 1 - np.tanh(Y) ** 2},
                         ncc_tail_tol=5e-5)


def growth(P, J: float, alpha: float):
    """Most unstable admissible mode at (J, alpha): full QZ, then tracked (residual reported)."""
    return P.least_stable(growth="imag", select=admissible, J=J, alpha=alpha)


def neutral_J(P, alpha: float, start: float = 0.15):
    """Neutral J at alpha: Newton on c_i(J) = 0 along the continued mode (critical())."""
    return critical(P, "J", bracket=(0.02, 0.35), start=start, growth="imag", select=admissible,
                    values={"alpha": alpha}, xtol=1e-13)


if __name__ == "__main__":
    study = "--study" in sys.argv
    Ns = (64, 96, 128, 160, 192) if study else (64, 96)
    print(taylor_goldstein(Ns[0], 0.5).problem.explain())       # N = 48 under-resolves dy(dy(U))
    print("c_i(J) at alpha = 1/2 (tanh map L = 4, contour 0.4), drift between the last two N:")
    for J in (0.0, 0.1, 0.2, 0.24):
        row = [growth(taylor_goldstein(N, 0.5), J, 0.5) for N in Ns[-2:]]
        print(f"  J = {J:.2f}: c = {row[-1].sigma:.10f} (residual {row[-1].residual:.0e}), "
              f"drift {abs(row[-1].sigma - row[-2].sigma):.1e}")
    for alpha in (0.5, 0.3) if study else (0.5,):
        print(f"Neutral J at alpha = {alpha} (Hazel: alpha (1 - alpha) = {alpha * (1 - alpha)}):")
        for N in Ns:
            cp = neutral_J(taylor_goldstein(N, alpha), alpha)
            print(f"  N = {N:3d}: J_n = {cp.value:.13f}, error {cp.value - alpha * (1 - alpha):+.1e}, "
                  f"c = {cp.mode.sigma:.1e}, residual {cp.mode.residual:.0e}", flush=True)
        if alpha == 0.5:                            # Miles (1961), Howard (1961): J_c = 1/4
            assert abs(cp.value - 0.25) < (2e-7 if N >= 96 else 1e-5), (N, cp.value)
    if study:
        print("Map scale L and contour delta at alpha = 1/2, N = 160 (path independence):")
        for L, delta in ((4.0, 0.3), (4.0, 0.5), (8.0, 0.4), (3.0, 0.4)):
            cp = neutral_J(taylor_goldstein(160, 0.5, L=L, contour=delta), 0.5)
            print(f"  L = {L}, delta = {delta}: J_n = {cp.value:.13f}, error {cp.value - 0.25:+.1e}")
        print(cp.explain())
