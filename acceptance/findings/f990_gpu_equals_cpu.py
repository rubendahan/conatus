"""F990 (GPU retest of 0.5): the same model on backend="cuda" gives the CPU result.

Oracle: the CPU run of the same model, same resolution, scheme and step (the algebra is the same;
only the order of floating-point operations differs), and for Burgers the Cole-Hopf closed form.
Cases: viscous Burgers on RealFourier (closed form, 200 RK443 steps) and Rayleigh-Benard 2D on
RealFourier x Chebyshev with its nonlinear terms (100 RK443 steps, default and fused evaluator).
Bound: 1e-11 relative to the largest value, a few hundred steps of rounding. SKIP without a GPU.
"""
import numpy as np
from _acceptance import check, passed, skipped, start

start(990, "backend='cuda' equals backend='cpu'")
import conatus as cn  # noqa: E402
from conatus.runtime import cuda  # noqa: E402

if not cuda.available():
    skipped("no CUDA device (conatus.runtime.cuda.available() is False)")

nu, a = 0.1, 0.5


def burgers(backend):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="Burgers", nu=nu)
    m.scalar("u")
    m.equations("dt(u) - nu*lap(u) = -u*dx(u)")
    sim = m.simulate(N=64, scheme="RK443", dt=1e-3, backend=backend)
    sim.initial(u=lambda x: 2 * nu * a * np.sin(x) / (1 + a * np.cos(x)))
    sim.run(steps=200)
    x = sim.coordinate("x")
    e = a * np.exp(-nu * sim.t)
    return np.asarray(sim.grid("u")), 2 * nu * e * np.sin(x) / (1 + e * np.cos(x))


def rb(backend, **kw):
    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
    m = cn.Model(box, name="RB", Ra=1e5, Pr=1)
    m.velocity("u")
    m.scalar("T")
    m.pressure("p")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=(32, 24), scheme="RK443", dt=1e-4, backend=backend, **kw)
    sim.initial(T=lambda x, z: -z + 0.05 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.run(steps=100)
    return np.asarray(sim.grid("T"))


def rel(g, c):
    return float(np.max(np.abs(g - c)) / np.max(np.abs(c)))


(gu, exact), (cu, _) = burgers("cuda"), burgers("cpu")
check(rel(gu, cu) < 1e-11, f"Burgers: GPU - CPU = {rel(gu, cu):.1e} relative")
check(np.max(np.abs(gu - exact)) < 1e-8, f"Burgers: GPU vs Cole-Hopf {np.max(np.abs(gu - exact)):.1e}")
tc, gaps = rb("cpu"), []
for kw in ({}, {"f_backend": "fused"}):
    gaps.append(rel(rb("cuda", **kw), tc))
    check(gaps[-1] < 1e-11, f"RB 2D {kw or 'default F'}: GPU - CPU = {gaps[-1]:.1e} relative")
passed(f"Burgers {rel(gu, cu):.1e}, RB 2D {gaps[0]:.1e} (fused F {gaps[1]:.1e})")
