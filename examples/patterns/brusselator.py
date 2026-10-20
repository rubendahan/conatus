"""Turing instability of the Brusselator (Prigogine & Lefever 1968).

    dt u = Du dx^2 u + a - (b + 1) u + u^2 v
    dt v = Dv dx^2 v + b u - u^2 v

The uniform state (u, v) = (a, b/a) loses stability to a stationary pattern of wavenumber k_c at

    b_c = (1 + a eta)^2,   k_c^2 = a / sqrt(Du Dv),   eta = sqrt(Du / Dv)

(Peña & Pérez-García 2001, Phys. Rev. E 64, 056213, Sec. II), provided b_c < 1 + a^2 (the Hopf
threshold of the k = 0 mode), i.e. Dv / Du large enough. The nonlinear IVP is written as in the
article and linearized by the IR around the uniform state; a perturbation exp(i k x + sigma t)
lives in one Fourier group (2x2 pencil). b enters linearly: the parametric pencil gives the neutral
b(k) by Newton on Re(sigma), minimized over k (the domain length 2 pi / k is rebuilt per k).

Run:  python examples/patterns/brusselator.py
"""
import numpy as np
import scipy.optimize as so

from conatus.expert import (IVP, CartesianCoordinates, ComplexFourier, Domain, Field, Parameter,
                            ParametricEVP, critical, dt, lap, linearize)


def brusselator(k: float, a: float, Du: float, Dv: float, b0: float):
    """Parametric pencil in b of the Brusselator linearized around (a, b/a), wavenumber k."""
    cs = CartesianCoordinates("x")
    x = cs["x"]
    D = Domain(cs, [ComplexFourier(x, 4, (0, 2 * np.pi / k), dealias=1)])   # modes 0, +-k
    u, v = Field("u", D), Field("v", D)
    A, B = Parameter("a", a), Parameter("b", b0)
    ivp = IVP([u, v], name="Brusselator")
    ivp.add_equation((dt(u) - Du * lap(u) + (B + 1) * u, A + u * u * v))
    ivp.add_equation((dt(v) - Dv * lap(v) - B * u, -u * u * v))
    evp = linearize(ivp, around={u: A, v: B / A}, eigenvalue=Parameter("sigma", dtype=complex))
    return ParametricEVP(evp, [B], group=lambda g: g[0].k > 0), evp


def turing_threshold(a: float, Du: float, Dv: float):
    """(b_c, k_c): neutral b(k) of the least stable mode, minimized over k."""
    b_guess = (1 + a * np.sqrt(Du / Dv)) ** 2
    crit = {}

    def neutral_b(k):
        P, _ = brusselator(k, a, Du, Dv, b_guess)
        crit[k] = critical(P, "b", bracket=(0.2 * b_guess, 1 + a * a), growth="real", xtol=1e-9)
        return crit[k].value

    kg = np.sqrt(a / np.sqrt(Du * Dv))
    opt = so.minimize_scalar(neutral_b, bounds=(0.7 * kg, 1.3 * kg), method="bounded",
                             options={"xatol": 1e-10})
    return neutral_b(float(opt.x)), float(opt.x), crit[float(opt.x)]


if __name__ == "__main__":
    for a, Du, Dv in ((2.0, 1.0, 8.0), (3.0, 1.0, 20.0), (1.5, 2.0, 16.0)):
        bc, kc, cp = turing_threshold(a, Du, Dv)
        bth, kth = (1 + a * np.sqrt(Du / Dv)) ** 2, np.sqrt(a / np.sqrt(Du * Dv))
        print(f"a={a}, Du={Du}, Dv={Dv}: b_c = {bc:.12f} (theory {bth:.12f}, gap {bc - bth:.1e}), "
              f"k_c = {kc:.8f} (theory {kth:.8f}), sigma = {cp.mode.sigma:.1e}, residual {cp.mode.residual:.1e}")
        assert abs(bc - bth) < 1e-12 and abs(kc - kth) < 1e-6, (a, bc, bth, kc, kth)   # closed form
    print(cp.explain())
    print(brusselator(kc, a, Du, Dv, bc)[1].explain())
