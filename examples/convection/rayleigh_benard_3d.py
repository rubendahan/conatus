"""Rayleigh–Bénard onset in 3D: the nonlinear Boussinesq equations, linearized by the IR.

Layer -1/2 < z < 1/2 between rigid isothermal walls, periodic in x and y (Fourier x Fourier x
Chebyshev), units of depth and thermal diffusion time:

    div u = 0
    dt T - lap T = -u . grad T
    dt u - Pr lap u + grad p - Pr Ra T e_z = -u . grad u

around the conduction state u = 0, T0 = -z. linearize() gives the eigenvalue problem; each
Fourier group (kx, ky) is one pencil. Onset: Ra_c = 1707.762 at |k| = 3.1163 (Reid & Harris 1958;
Chandrasekhar 1961, Ch. II, Table III). Isotropy: the growth rate depends on |k| only, so
(kx, ky) = (K, 0), (0, K) and (K cos 30°, K sin 30°) are all marginal at Ra_c.

Run:  python examples/convection/rayleigh_benard_3d.py
"""
import numpy as np
import scipy.optimize as so

from conatus.expert import (IVP, CartesianCoordinates, Chebyshev, ComplexFourier, Domain, Field, Parameter,
                            VectorField, Zero, div, dot, dt, eigenvalues, grad, integ, interp, lap, lift,
                            discretize, linearize, unit_vector)

RA_C, K_C = 1707.762, 3.1163           # rigid-rigid onset (Reid & Harris 1958)


def boussinesq_3d(kx: float, ky: float, Nz: int):
    """Nonlinear Boussinesq IVP on a box holding exactly the wavevector (kx, ky) (4 modes a side)."""
    cs = CartesianCoordinates("x", "y", "z")
    x, y, z = cs.coords
    # box lengths 2 pi / k hold the wavevector exactly (a zero component: any length, here 2 pi)
    bx = ComplexFourier(x, 4, (0, 2 * np.pi / (kx or 1.0)), dealias=1)
    by = ComplexFourier(y, 4, (0, 2 * np.pi / (ky or 1.0)), dealias=1)
    bz = Chebyshev(z, Nz, (-0.5, 0.5), dealias=1)
    D = Domain(cs, [bx, by, bz])
    Dh = D.without(z)
    u, T, p = VectorField("u", D), Field("T", D), Field("p", D)
    tp = Field("tp", Domain(cs, []))
    tT1, tT2 = Field("tT1", Dh), Field("tT2", Dh)
    tu1, tu2 = VectorField("tu1", Dh), VectorField("tu2", Dh)
    Ra, Pr = Parameter("Ra", RA_C), Parameter("Pr", 1.0)
    ez = unit_vector(z)
    ivp = IVP([u, T, p, tp, tT1, tT2, tu1, tu2], name="Boussinesq RB 3D")
    ivp.add_equation((div(u) + tp, 0))
    ivp.add_equation((dt(T) - lap(T) + lift(tT1, bz, -1) + lift(tT2, bz, -2), -dot(u, grad(T))))
    ivp.add_equation((dt(u) - Pr * lap(u) + grad(p) - Pr * Ra * T * ez
                      + lift(tu1, bz, -1) + lift(tu2, bz, -2), -dot(u, grad(u))))
    ivp.add_equation((interp(T, z, "left"), 0.5))
    ivp.add_equation((interp(T, z, "right"), -0.5))
    ivp.add_equation((interp(u, z, "left"), 0))
    ivp.add_equation((interp(u, z, "right"), 0))
    ivp.add_equation((integ(p), 0))
    T0 = Field("T0", Domain(cs, [bz]))
    evp = linearize(ivp, around={u: Zero((cs,)), T: T0}, eigenvalue=Parameter("s", dtype=complex))
    return evp, T0


def growth(kx: float, ky: float, Ra: float, Nz: int = 24):
    """Leading growth rate of the (kx, ky) group at Ra, with its residual."""
    evp, T0 = boussinesq_3d(kx, ky, Nz)
    plan = discretize(evp, ncc={T0: lambda X, Y, Z: -Z}, params={"Ra": Ra})
    g = next(g for g in plan.groups if np.isclose(g[0].k, kx) and np.isclose(g[1].k, ky))
    r = eigenvalues(plan, g)
    i = int(np.argmax(r.values.real))
    return r.values[i], r.residuals[i]


def neutral_ra(kx: float, ky: float, Nz: int = 24) -> float:
    """Ra where the leading (real) growth rate of the (kx, ky) group vanishes."""
    return so.brentq(lambda Ra: growth(kx, ky, Ra, Nz)[0].real, 1600.0, 1800.0, xtol=1e-9)


if __name__ == "__main__":
    K = K_C
    for kx, ky in [(K, 0.0), (0.0, K), (K * np.cos(np.pi / 6), K * np.sin(np.pi / 6))]:
        Ra = neutral_ra(kx, ky)
        s, res = growth(kx, ky, Ra)
        print(f"  k = ({kx:.4f}, {ky:.4f}), |k| = {np.hypot(kx, ky):.4f}: neutral Ra = {Ra:.8f} "
              f"(2D reference {RA_C} at the minimum), sigma = {s:.1e}, residual {res:.1e}")
    evp, _ = boussinesq_3d(K, 0.0, 24)
    print(evp.explain())
