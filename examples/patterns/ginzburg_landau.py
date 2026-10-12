"""Complex Ginzburg–Landau equation: Benjamin–Feir–Newell and Eckhaus instabilities of plane waves.

    dt A = A + (1 + i c1) dx^2 A - (1 + i c3) |A|^2 A

(Aranson & Kramer 2002, Rev. Mod. Phys. 74, 99, eq. (1), with b = c1, c = c3). The plane wave
A = R exp(i (q x - omega t)), R^2 = 1 - q^2, omega = c3 R^2 + c1 q^2, is linearly unstable to long
waves (Benjamin–Feir–Newell) iff 1 + c1 c3 < 0 (Newell 1974; Aranson & Kramer 2002, Sec. II.B);
for 1 + c1 c3 > 0 it is Eckhaus-unstable iff q^2 > q_E^2 = (1 + c1 c3) / (3 + c1 c3 + 2 c3^2).

Written in the co-moving, co-rotating variable B = A exp(-i (q x - omega t)), the plane wave is the
constant state B = R and the equation keeps constant coefficients:

    dt B = (1 + i omega) B + (1 + i c1)(dx^2 B + 2 i q dx B - q^2 B) - (1 + i c3) |B|^2 B.

The IR linearizes the complex form: conj(dB) appears (|B|^2 B = B B conj(B)), so the linear
operator is R-linear, not C-linear, and couples the Fourier modes k and -k. The discretizer solves
it as the equivalent doubled system in (dB, conj dB), every equation followed by its conjugate, the
eigenvalue standing for d/dt (declared in explain(); before conj() in an LHS was refused
and this example was written in the real variables B = X + i Y: same spectrum, checked in
tests/discretize/test_conj_lhs.py). A perturbation exp(i k x + sigma t) lives in one Fourier group
(2x2 pencil); the long-wave threshold is the k -> 0 limit.

Run:  python examples/patterns/ginzburg_landau.py
"""
import numpy as np
import scipy.optimize as so

from conatus.expert import (IVP, CartesianCoordinates, ComplexFourier, Domain, Field, Parameter,
                            ParametricEVP, conj, critical, diff, discretize, dt, eigenvalues, linearize,
                            sqrt)


def cgle(k: float, c1: float, c3: float, q: float):
    """Linearized CGLE around the plane wave of wavenumber q, one perturbation wavenumber k."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [ComplexFourier(x, 4, (0, 2 * np.pi / k), dealias=1)])   # modes 0, +-k
    B = Field("B", D, dtype=complex)
    C1, C3, Q = Parameter("c1", c1), Parameter("c3", c3), Parameter("q", q)
    R2 = 1 - Q ** 2                                       # plane-wave amplitude squared
    w = C3 * R2 + C1 * Q ** 2                             # plane-wave frequency

    def disp(f):                                          # (1 + i c1)(f'' + 2 i q f' - q^2 f)
        return (1 + 1j * C1) * (diff(diff(f, x), x) + 2j * Q * diff(f, x) - Q ** 2 * f)
    ivp = IVP([B], name="CGLE (co-moving frame)")
    ivp.add_equation((dt(B) - (1 + 1j * w) * B - disp(B), -(1 + 1j * C3) * B * B * conj(B)))
    sigma = Parameter("sigma", dtype=complex)
    return linearize(ivp, around={B: sqrt(R2)}, eigenvalue=sigma)


def growth(k: float, c1: float, c3: float, q: float) -> tuple[float, float]:
    """Largest Re(sigma) over the groups +-k, and the largest eigen-residual."""
    plan = discretize(cgle(k, c1, c3, q))
    res = [eigenvalues(plan, g) for g in plan.groups if g[0].k != 0]
    return (max(float(r.values.real.max()) for r in res), max(float(r.residuals.max()) for r in res))


def bfn_threshold(k: float, c1: float) -> float:
    """c3 at which the plane wave q = 0 becomes unstable at wavenumber k (parametric pencil in c3)."""
    P = ParametricEVP(cgle(k, c1, -0.5 / c1, 0.0), [Parameter("c3", -0.5 / c1)],
                      group=lambda g: g[0].k > 0)
    return critical(P, "c3", bracket=(-3.0 / c1, -0.2 / c1), growth="real", xtol=1e-9).value


def eckhaus_threshold(k: float, c1: float, c3: float) -> float:
    """q at which the plane wave becomes unstable at small perturbation wavenumber k."""
    return so.brentq(lambda q: growth(k, c1, c3, q)[0], 1e-3, 0.99, xtol=1e-13)


if __name__ == "__main__":
    c1 = 2.0
    print(f"Benjamin–Feir–Newell, c1 = {c1}: predicted c3 = -1/c1 = {-1 / c1}")
    for k in (0.1, 0.05, 0.025):
        c3k = bfn_threshold(k, c1)
        print(f"  k = {k:5.3f}: c3_threshold = {c3k:.12f}, 1 + c1 c3 = {1 + c1 * c3k:.3e}"
              f"  (exact 2x2: -(2 + k^2 (1 + c1^2)) / (2 c1) = {-(2 + k * k * (1 + c1 * c1)) / (2 * c1):.12f})")
    for c3 in (-0.4, -0.6):
        g, r = growth(0.05, c1, c3, 0.0)
        print(f"  c3 = {c3}: 1 + c1 c3 = {1 + c1 * c3:+.1f}, max Re(sigma) at k=0.05 = {g:+.3e} (residual {r:.1e})")
    print("Eckhaus (1 + c1 c3 > 0): q_E^2 = (1 + c1 c3) / (3 + c1 c3 + 2 c3^2)")
    for c1, c3 in ((0.0, 0.0), (0.5, 0.5), (1.0, -0.5)):
        qE2 = (1 + c1 * c3) / (3 + c1 * c3 + 2 * c3 ** 2)
        for k in (0.02, 0.01):
            q = eckhaus_threshold(k, c1, c3)
            print(f"  c1={c1}, c3={c3}, k={k}: q^2 = {q * q:.10f}  (theory {qE2:.10f}, gap {q * q - qE2:.2e})")
    print(cgle(0.05, 2.0, -0.6, 0.0).explain())
