"""Weakly nonlinear theory derived from the equations of the model: Landau and Ginzburg–Landau
coefficients.

(a) Swift–Hohenberg, du/dt = r u − (1 + ∂x²)² u − u³ (Swift & Hohenberg 1977). Rolls u = A cos x:
    the pitchfork normal form on the even subspace gives dA/dt = r A − (3/4) A³, i.e. A² = 4r/3
    (Cross & Hohenberg 1993, RMP 65, §IV); the envelope equation has τ0 = 1, ξ0² = 4, g0 = 3.
(b) Rayleigh–Bénard between stress-free isothermal walls (ψ–θ form, thermal-diffusion units,
    z ∈ (0, 1)), onset Ra_c = 27π⁴/4 at k_c = π/√2 (Rayleigh 1916). The Landau equation and the
    second-order mean temperature give the heat flux of the rolls: Nu − 1 = 2 (Ra − Ra_c)/Ra_c + ...
    for every Pr (Malkus & Veronis 1958; the same number follows from the Lorenz 1963 truncation,
    exact at this order since the O(a²) correction is the sin 2πz mean mode only).

Every coefficient comes from ``normal_form``: symbolic 2nd/3rd Fréchet derivatives of the
right-hand side of the model, bordered solves on the discrete Jacobian. Nothing is derived by hand here.

Run:  python examples/bifurcation/weakly_nonlinear.py
"""
import numpy as np

import conatus.expert as C
from conatus.bifurcation import (amplitude_equation, cos_mode_observable, ginzburg_landau, ir_system,
                                 normal_form)


# ------------------------------------------------------------------ (a) Swift–Hohenberg
def swift_hohenberg(k: float, N: int = 16):
    """SH on one period 2π/k, even subspace (kills the translation mode), r free."""
    cs = C.CartesianCoordinates("x")
    x = cs["x"]
    D = C.Domain(cs, [C.RealFourier(x, N, (0, 2 * np.pi / k), dealias=2)])   # cubic: dealias 2
    u, r = C.Field("u", D), C.Parameter("r", 0.0)
    pb = C.IVP([u], name="Swift-Hohenberg", symmetry=C.SymmetrySubspace(C.Reflection(x, about=0), u=C.EVEN))
    L = lambda f: f + C.diff(C.diff(f, x), x)
    pb.add_equation((C.dt(u) - r * u + L(L(u)), -u * u * u))
    s = ir_system(pb, ["r"])
    return s, np.zeros(s.n)


def sh_coefficients():
    s, x0 = swift_hohenberg(1.0)
    le = amplitude_equation(s, x0, "r", kind="stationary", observable=cos_mode_observable(s, "u"))
    gl = ginzburg_landau(swift_hohenberg, 1.0, "r", kind="stationary", observable=cos_mode_observable(s, "u"), h=1e-2)
    return le, gl


# ------------------------------------------------------------------ (b) stress-free convection
def rb_free_slip(Nx: int, Nz: int, Pr: float):
    """ψ–θ Boussinesq rolls between stress-free walls, x-period 2π/k_c, reflection subspace
    (ψ odd, θ even in x). Ra free, starting at Ra_c = 27π⁴/4."""
    kc = np.pi / np.sqrt(2)
    cs = C.CartesianCoordinates("x", "z")
    x, z = cs.coords
    bz = C.Chebyshev(z, Nz, (0, 1), dealias=3 / 2)
    D = C.Domain(cs, [C.RealFourier(x, Nx, (0, 2 * np.pi / kc), dealias=3 / 2), bz])
    Dx = D.without(z)
    psi, th = C.Field("psi", D), C.Field("theta", D)
    tps = [C.Field(f"tpsi{i}", Dx) for i in range(1, 5)]
    tth = [C.Field(f"tth{i}", Dx) for i in range(1, 3)]
    Ra, Pr_ = C.Parameter("Ra", 27 * np.pi ** 4 / 4), C.Parameter("Pr", Pr)
    G = C.SymmetrySubspace(C.Reflection(x, about=0), psi=C.ODD, theta=C.EVEN,
                           **{t.name: C.ODD for t in tps}, **{t.name: C.EVEN for t in tth})
    pb = C.IVP([psi, th, *tps, *tth], name="RB stress-free (psi-theta)", symmetry=G)
    u, w, om = C.diff(psi, z), -C.diff(psi, x), C.lap(psi)
    pb.add_equation((C.dt(om) - Pr_ * C.lap(om) + Ra * Pr_ * C.diff(th, x)
                     + sum(C.lift(t, bz, -i) for i, t in enumerate(tps, 1)),
                     -(u * C.diff(om, x) + w * C.diff(om, z))))
    pb.add_equation((C.dt(th) - C.lap(th) + C.diff(psi, x)
                     + sum(C.lift(t, bz, -i) for i, t in enumerate(tth, 1)),
                     -(u * C.diff(th, x) + w * C.diff(th, z))))
    for side in ("left", "right"):
        pb.add_equation((C.interp(psi, z, side), 0))
        pb.add_equation((C.interp(C.diff(C.diff(psi, z), z), z, side), 0))
        pb.add_equation((C.interp(th, z, side), 0))
    s = ir_system(pb, ["Ra"])
    return s, np.zeros(s.n)


def nusselt_slope(Nx: int, Nz: int, Pr: float):
    """d Nu / dε at onset: Nu − 1 = −(a²/2) ∂zθ̄_{h2}(0) with a² = −μ ΔRa / a3 (normal form)."""
    s, x0 = rb_free_slip(Nx, Nz, Pr)
    nf = normal_form(s, x0, kind="zero", parameter="Ra", kernel_tol=1e-6)
    theta = s.solver.system.scatter(nf.h2)[1]            # (x modes, Chebyshev); row 0 = x-mean
    n = np.arange(theta.shape[1])
    dz0 = np.sum(theta[0] * 2 * (-1.0) ** (n + 1) * n ** 2)  # d/dz T_n(2z − 1) at z = 0
    Rac = 27 * np.pi ** 4 / 4
    return Rac * (-nf.mu / nf.a3) * (-0.5 * dz0), nf


if __name__ == "__main__":
    le, gl = sh_coefficients()
    print(le.explain(), gl.explain(), sep="\n")
    for Pr in (0.1, 1.0, 7.0):
        slope, nf = nusselt_slope(8, 16, Pr)
        print(f"stress-free RB, Pr = {Pr}: dNu/dε = {slope:.10f} (Malkus & Veronis 1958: 2), "
              f"Ra_c = {nf.params['Ra']:.10f}, |λ0| = {nf.checks['|λ_0|']:.1e}")
        # the script checks its references: Rayleigh (1916) Ra_c = 27 pi^4 / 4, Malkus & Veronis (1958) slope 2
        assert abs(nf.params["Ra"] - 27 * np.pi ** 4 / 4) < 1e-9 and abs(slope - 2) < 1e-8, (Pr, slope)
    assert abs(le.amplitude(0.1) ** 2 - 4 * 0.1 / 3) < 1e-13         # Cross & Hohenberg (1993): A^2 = 4r/3
