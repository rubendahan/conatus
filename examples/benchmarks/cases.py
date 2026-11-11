"""Benchmark cases of the public suite: fixed problems, each with an accuracy check against an
oracle that does not come from Conatus (a closed form, a published value, or a quantity the
equations fix exactly).

Every case is a function ``case(size) -> dict`` that builds the problem with the public API, times
its phases with ``time.perf_counter`` and returns:

* ``timings``: ``setup_s`` (model, discretisation, factorisations, first step), ``run_s`` (the
  measured part) and, for an initial-value problem, ``steps`` and ``s_per_step`` (median over
  blocks of ``BLOCK`` steps, each block one ``sim.run(steps=BLOCK)`` call, so the per-call cost of
  ``run`` is included as a user pays it) with ``s_per_step_min`` and ``s_per_step_mean``;
* ``accuracy``: ``quantity``, ``value``, ``oracle``, ``error``, ``tolerance`` and ``passed``
  (``error <= tolerance``); the tolerance is written here, before any run, from the expected
  discretisation error, never fitted to a measured error;
* ``problem``: resolution, scheme, time step and what is computed.

``SIZES[name]`` lists the sizes a case accepts; ``"default"`` is what the suite runs.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

BLOCK = 10           # steps per timed block of an initial-value case


def _blocks(sim, steps: int) -> dict:
    """Run ``steps`` steps in blocks of BLOCK; per-step times of each block."""
    per = []
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        n = min(BLOCK, steps - done)
        t1 = time.perf_counter()
        sim.run(steps=n)
        per.append((time.perf_counter() - t1) / n)
        done += n
    run = time.perf_counter() - t0
    return dict(run_s=run, steps=steps, s_per_step=float(np.median(per)),
                s_per_step_min=float(np.min(per)), s_per_step_mean=run / steps, _per=per)


def _acc(quantity: str, value: Any, oracle: Any, error: float, tol: float, source: str) -> dict:
    return dict(quantity=quantity, value=value, oracle=oracle, oracle_source=source,
                error=float(error), tolerance=tol, passed=bool(np.isfinite(error) and error <= tol))


def _sha(sim) -> str:
    """SHA-256 of the final state's bytes: two versions that claim the same arithmetic must give
    the same hash (an equal error is a weaker check)."""
    import hashlib
    X = sim.solver.X
    X = X.get() if hasattr(X, "get") else X          # device state (backend="cuda"): explicit copy
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()


def _cx(z) -> Any:
    z = complex(z)
    return [z.real, z.imag]


# ------------------------------------------------------------------------------------------------
# IVP 1D, RealFourier: viscous Burgers, Cole-Hopf closed form
# ------------------------------------------------------------------------------------------------

def ivp1d_burgers(size: str) -> dict:
    """u_t + u u_x = nu u_xx on [0, 2 pi): phi = 1 + a exp(-nu t) cos x solves the heat equation,
    so u = -2 nu phi_x / phi = 2 nu a exp(-nu t) sin x / (1 + a exp(-nu t) cos x) exactly
    (Cole-Hopf). Error: max over the grid at t = T. Fourier coefficients decay like 0.27^k, so N =
    64 is resolved to round-off; RK443 at dt = 1e-3 leaves a time error far below 1e-8."""
    import conatus as cn
    N = {"default": 64, "large": 1024}[size]
    nu, a, dt, T = 0.1, 0.5, 1e-3, 1.0
    t0 = time.perf_counter()
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="Burgers", nu=nu)
    m.scalar("u")
    m.equations("dt(u) - nu*lap(u) = -u*dx(u)")

    def exact(x, t):
        e = a * np.exp(-nu * t)
        return 2 * nu * e * np.sin(x) / (1 + e * np.cos(x))

    sim = m.simulate(N=N, scheme="RK443", dt=dt)
    sim.initial(u=lambda x: exact(x, 0.0))
    sim.run(steps=BLOCK)
    setup = time.perf_counter() - t0
    steps = int(round(T / dt)) - BLOCK
    tim = _blocks(sim, steps)
    x = sim.coordinate("x")
    err = float(np.max(np.abs(sim.grid("u") - exact(x, sim.t))))
    return dict(problem=dict(N=N, scheme="RK443", dt=dt, T=T, nu=nu, a=a, basis="RealFourier"),
                timings=dict(setup_s=setup, **{k: v for k, v in tim.items() if k != "_per"}),
                accuracy=_acc("max |u - u_exact| at t = T", None, None, err, 1e-8,
                              "Cole-Hopf closed form") | dict(state_sha256=_sha(sim)))


# ------------------------------------------------------------------------------------------------
# IVP 2D, RealFourier x RealFourier: Taylor-Green vortex, exact decaying solution of Navier-Stokes
# ------------------------------------------------------------------------------------------------

def ivp2d_taylor_green(size: str) -> dict:
    """u = (sin x cos y, -cos x sin y) exp(-2 nu t) solves the 2D Navier-Stokes equations exactly
    (u.grad u is a gradient, balanced by the pressure). Error: max over the grid of |u - exact| at
    t = T; the solution is one Fourier mode, so the error is the RK443 error of exp(-2 nu t)."""
    import conatus as cn
    N = {"default": 64, "large": 256}[size]
    nu, dt, T = 0.05, 1e-2, 2.0
    t0 = time.perf_counter()
    box = cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi))
    m = cn.Model(box, name="Taylor-Green 2D", nu=nu)
    m.velocity("u")
    m.pressure("p")
    m.equations("""
        momentum:   dt(u) + grad(p) - nu*lap(u) = -u·grad(u)
        continuity: div(u) = 0
    """)

    def exact(x, y, t):
        f = np.exp(-2 * nu * t)
        return np.array([np.sin(x) * np.cos(y) * f, -np.cos(x) * np.sin(y) * f])

    sim = m.simulate(N=(N, N), scheme="RK443", dt=dt)
    sim.initial(u=lambda x, y: exact(x, y, 0.0))
    sim.run(steps=BLOCK)
    setup = time.perf_counter() - t0
    steps = int(round(T / dt)) - BLOCK
    tim = _blocks(sim, steps)
    X, Y = np.meshgrid(sim.coordinate("x"), sim.coordinate("y"), indexing="ij")
    err = float(np.max(np.abs(sim.grid("u") - exact(X, Y, sim.t))))
    return dict(problem=dict(N=[N, N], scheme="RK443", dt=dt, T=T, nu=nu,
                             basis="RealFourier x RealFourier"),
                timings=dict(setup_s=setup, **{k: v for k, v in tim.items() if k != "_per"}),
                accuracy=_acc("max |u - u_exact| at t = T", None, None, err, 1e-9,
                              "exact decaying solution of the Navier-Stokes equations")
                | dict(state_sha256=_sha(sim)))


# ------------------------------------------------------------------------------------------------
# Rayleigh-Benard growth rate (2D: RealFourier x Chebyshev; 3D: RealFourier^2 x Chebyshev)
# ------------------------------------------------------------------------------------------------

SIGMA_RB = 5.5425059   # k = pi, Ra = 2500, Pr = 1, rigid walls (tutorial 2a; psi-theta EVP, tests/ivp)


def _rb_model(three_d: bool):
    import conatus as cn
    axes = dict(x=cn.periodic(2.0), y=cn.periodic(2.0)) if three_d else dict(x=cn.periodic(2.0))
    box = cn.Box(**axes, z=cn.walls(-0.5, 0.5))
    m = cn.Model(box, name="Rayleigh-Benard", Ra=2500, Pr=1)
    u, T = m.velocity("u"), m.scalar("T")
    m.pressure("p")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    return m, u, T


def _rb_growth(three_d: bool, N: tuple, dt: float, t1: float, t2: float) -> dict:
    """Growth rate of the k = pi roll from the nonlinear IVP (small amplitude, nonlinear terms
    evaluated every step), against the EVP of the same equations at N_z = 24 (SIGMA_RB, which
    the suite also recomputes). Tolerance 1e-5: the tutorial's check at (16, 24), dt = 5e-4."""
    t0 = time.perf_counter()
    m, u, T = _rb_model(three_d)
    lin = m.linearize(about={u: 0, T: lambda z: -z})
    kk = (np.pi, 0.0) if three_d else np.pi
    sigma_evp = float(lin.eigenvalues(N=24, k=kk).values.real.max())
    t_evp = time.perf_counter() - t0
    sim = m.simulate(N=N, scheme="RK443", dt=dt)
    if three_d:
        sim.initial(T=lambda x, y, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))
    else:
        sim.initial(T=lambda x, z: -z + 1e-6 * np.cos(np.pi * x) * np.cos(np.pi * z))
    sim.run(steps=BLOCK)
    setup = time.perf_counter() - t0 - t_evp

    def amp():
        Tg = sim.grid("T")
        return np.abs(Tg - Tg.mean(axis=tuple(range(Tg.ndim - 1)))).max()

    n1 = int(round(t1 / dt)) - BLOCK
    n2 = int(round((t2 - t1) / dt))
    ta = _blocks(sim, n1)
    a1, s1 = amp(), sim.t
    tb = _blocks(sim, n2)
    a2, s2 = amp(), sim.t
    sigma = float(np.log(a2 / a1) / (s2 - s1))
    steps = ta["steps"] + tb["steps"]
    run = ta["run_s"] + tb["run_s"]
    per = ta["_per"] + tb["_per"]
    tim = dict(setup_s=setup, evp_s=t_evp, run_s=run, steps=steps,
               s_per_step=float(np.median(per)), s_per_step_min=float(np.min(per)),
               s_per_step_mean=run / steps)
    acc = _acc("growth rate sigma of the k = pi roll (IVP)", sigma, SIGMA_RB, abs(sigma - SIGMA_RB),
               1e-5, "EVP of the same equations, psi-theta formulation (tests/ivp/test_psi_theta.py)")
    acc["state_sha256"] = _sha(sim)
    acc["evp_sigma"] = sigma_evp
    acc["evp_error"] = abs(sigma_evp - SIGMA_RB)
    acc["passed"] = bool(acc["passed"] and acc["evp_error"] < 1e-6)
    return dict(problem=dict(N=list(N), scheme="RK443", dt=dt, Ra=2500, Pr=1, t1=t1, t2=t2,
                             basis=("RealFourier^2 x Chebyshev" if three_d
                                    else "RealFourier x Chebyshev")),
                timings=tim, accuracy=acc)


def ivp2d_rb(size: str) -> dict:
    N = {"default": (64, 32), "small": (16, 24), "large": (256, 64)}[size]
    return _rb_growth(False, N, 5e-4, 0.3, 0.8)


def ivp3d_rb(size: str) -> dict:
    N = {"default": (16, 16, 24), "large": (32, 32, 32)}[size]
    return _rb_growth(True, N, 5e-4, 0.3, 0.8)


# ------------------------------------------------------------------------------------------------
# LBVP: Poisson, RealFourier x Chebyshev, manufactured solution
# ------------------------------------------------------------------------------------------------

def lbvp_poisson(size: str) -> dict:
    """lap(u) = f on [0, 2 pi) x [-1, 1], u = 0 at the walls, with u = cos(3x) cos(pi z / 2) +
    sin(x) (1 - z^2) exp(z): f by hand. Error: max over the grid; the solution is entire in z, so
    N_z = 64 resolves it to round-off (tolerance 1e-10). The solve is timed REPEAT times on the
    same plan (median), the setup (compile, discretise, first solve) once."""
    import conatus as cn
    Nx, Nz = {"default": (64, 64), "large": (512, 256)}[size]
    REPEAT = 5

    def exact(x, z):
        return np.cos(3 * x) * np.cos(np.pi * z / 2) + np.sin(x) * (1 - z * z) * np.exp(z)

    def rhs(x, z):
        a = -(9 + np.pi ** 2 / 4) * np.cos(3 * x) * np.cos(np.pi * z / 2)
        # d2/dz2 [(1 - z^2) e^z] = (-1 - 4z - z^2) e^z ; d2/dx2 sin x = -sin x
        b = np.sin(x) * ((-1 - 4 * z - z * z) * np.exp(z) - (1 - z * z) * np.exp(z))
        return a + b

    t0 = time.perf_counter()
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(-1, 1)), name="Poisson")
    m.scalar("u")
    m.given(f=rhs)
    m.equations("lap(u) = f")
    m.at("walls", u=0)
    plan = m.discretize((Nx, Nz))
    sol = plan.solve()
    setup = time.perf_counter() - t0
    ts = []
    for _ in range(REPEAT):
        t1 = time.perf_counter()
        sol = plan.solve()
        ts.append(time.perf_counter() - t1)
    X, Z = np.meshgrid(sol.coordinate("x"), sol.coordinate("z"), indexing="ij")
    err = float(np.max(np.abs(sol.grid("u") - exact(X, Z))))
    acc = _acc("max |u - u_exact|", None, None, err, 1e-10, "manufactured closed form")
    acc["max_relative_residual"] = float(sol.max_residual)   # a property
    return dict(problem=dict(N=[Nx, Nz], basis="RealFourier x Chebyshev"),
                timings=dict(setup_s=setup, run_s=float(np.sum(ts)), solve_s=float(np.median(ts)),
                             repeats=REPEAT),
                accuracy=acc)


# ------------------------------------------------------------------------------------------------
# EVP: Orr-Sommerfeld, dense and shift-invert
# ------------------------------------------------------------------------------------------------

C_ORSZAG = 0.23752649 + 0.00373967j   # Orszag (1971), Re = 1e4, alpha = 1


def _os_model():
    import conatus as cn
    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y ** 2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("Orr_Sommerfeld: L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    return m


def _os_check(res, i, N, method, t_setup, t_solve) -> dict:
    c = complex(res.values[i])
    err = abs(c - C_ORSZAG)
    # Orszag prints 8 decimals: half a unit of the 8th decimal on each part, |error| <= sqrt(2) 5e-9
    acc = _acc("least stable c", _cx(c), _cx(C_ORSZAG), err, float(np.sqrt(2) * 5e-9),
               "Orszag (1971), JFM 50, 689")
    acc["backward_residual"] = float(res.residuals[i])
    acc["passed"] = bool(acc["passed"] and res.residuals[i] < 1e-8)
    return dict(problem=dict(N=N, method=method, Re=1e4, alpha=1.0, basis="Chebyshev"),
                timings=dict(setup_s=t_setup, run_s=t_solve), accuracy=acc)


def evp_dense(size: str) -> dict:
    """All eigenvalues of the Orr-Sommerfeld pencil (dense QZ) at N and at 3N/2 (``drift``: a mode
    is kept when found again at 3N/2, which sets the spurious tau modes aside, as in tutorial 1);
    least stable kept mode against Orszag. Both dense solves are in run_s."""
    N = {"default": 128, "large": 256}[size]
    t0 = time.perf_counter()
    m = _os_model()
    m.compile(N)
    t1 = time.perf_counter()
    res = m.eigenvalues(N=N, drift=3 * N // 2, drift_tol=1e-6)
    t2 = time.perf_counter()
    i = int(np.argmax(res.values.imag))
    return _os_check(res, i, N, "dense", t1 - t0, t2 - t1)


def evp_shift_invert(size: str) -> dict:
    """One eigenvalue near c = 0.24 by shift-invert (sparse LU + Arnoldi)."""
    N = {"default": 512, "large": 2048}[size]
    t0 = time.perf_counter()
    m = _os_model()
    m.compile(N)
    t1 = time.perf_counter()
    res = m.eigenvalues(N=N, method="shift-invert", target=0.24, n_modes=1)
    t2 = time.perf_counter()
    i = int(np.argmin(np.abs(res.values - 0.24)))
    return _os_check(res, i, N, "shift-invert", t1 - t0, t2 - t1)


# ------------------------------------------------------------------------------------------------
# Continuation: Bratu fold
# ------------------------------------------------------------------------------------------------

def continuation_bratu(size: str) -> dict:
    """Pseudo-arclength continuation of u'' + lam exp(u) = 0, u(+-1) = 0, from lam = 0.3 round the
    fold; the fold located by the continuation against the closed form a tanh a = 1,
    lam_c = 2 a^2 / cosh(a)^2 (tutorial 4; tolerance 1e-10 as there)."""
    from scipy.optimize import brentq

    import conatus as cn
    from conatus.expert import ContinuationSettings, NewtonSolver, NLBVPContinuation
    N = {"default": 64, "large": 256}[size]
    t0 = time.perf_counter()
    m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
    m.scalar("u")
    m.equations("bratu: dx(dx(u)) + lam*exp(u) = 0")
    m.at("walls", u=0)
    pb = m.compile(N, dealias=1)
    newton = NewtonSolver(pb.problem, tol=1e-13, damping=1.0, max_iterations=20, params={"lam": 0.3})
    sol = newton.solve({pb.fields["u"]: lambda x: 0 * x})
    t1 = time.perf_counter()
    settings = ContinuationSettings(ds=0.2, ds_min=1e-8, ds_max=0.6, adapt=1.5, fast_iterations=3,
                                    tolerance=1e-12, max_newton=15, max_points=6, dp_fd=1e-6)
    branch = NLBVPContinuation(newton, "lam", settings).run(sol.coeffs, 0.3, +1, p_bounds=(0.3, 1.0))
    (fold,) = branch.folds()
    t2 = time.perf_counter()
    a_star = brentq(lambda a: a * np.tanh(a) - 1.0, 0.5, 2.0, xtol=1e-15)
    lam_c = 2 * a_star ** 2 / np.cosh(a_star) ** 2
    acc = _acc("fold lam_c", float(fold.p), float(lam_c), abs(fold.p - lam_c), 1e-10,
               "closed form a tanh a = 1 (Bratu 1914; Gelfand 1963)")
    return dict(problem=dict(N=N, points=len(branch.p) if hasattr(branch, "p") else None,
                             basis="Chebyshev"),
                timings=dict(setup_s=t1 - t0, run_s=t2 - t1), accuracy=acc)


# ------------------------------------------------------------------------------------------------
# Floquet-Hill: diffusion with a time-periodic coefficient, exact exponents
# ------------------------------------------------------------------------------------------------

def floquet_diffusion(size: str) -> dict:
    """m (s u + a u) + u - D u'' + b cos(w t) u = 0, u(+-1) = 0: u = exp(-((m a + 1 + D k^2) t +
    b sin(w t) / w) / m) sin(k (z + 1)), k = n pi / 2, so the Floquet exponents are
    -(m a + 1 + D k^2) / m exactly (tests/exact/test_exact_floquet.py). Hill matrix of
    (2 Nh + 1) N unknowns, all exponents by dense QZ; the three leading ones are checked."""
    import conatus.expert as C
    from conatus.discretize import discretize
    from conatus.floquet.hill_general import exponents_dense, hill_from_plan
    N, Nh = {"default": (32, 12), "large": (48, 24)}[size]
    m_, a_, d_, b_, om = 2.0, 0.4, 0.3, 1.5, 3.0
    t0 = time.perf_counter()
    cs = C.CartesianCoordinates("z")
    z = cs["z"]
    bz = C.Chebyshev(z, N, (-1, 1), dealias=1)
    D = C.Domain(cs, [bz])
    u, U = C.Field("u", D, dtype=complex), C.Field("U", D, dtype=complex)
    t1_, t2_ = C.Field("t1", D.without(z), dtype=complex), C.Field("t2", D.without(z), dtype=complex)
    s = C.Parameter("s", dtype=complex)
    m, a, d = C.Parameter("m", m_), C.Parameter("a", a_), C.Parameter("D", d_)
    pb = C.EVP([u, t1_, t2_], eigenvalue=s)
    pb.add_equation((m * (s * u + a * u) + u - d * C.lap(u) + U * u
                     + C.lift(t1_, bz, -1) + C.lift(t2_, bz, -2), 0))
    pb.add_equation((C.interp(u, z, "left"), 0))
    pb.add_equation((C.interp(u, z, "right"), 0))
    harm = {1: lambda Z: b_ / 2 + 0 * Z + 0j, -1: lambda Z: b_ / 2 + 0 * Z + 0j}
    H = hill_from_plan(lambda ncc: discretize(pb, ncc={U: ncc[U]}), {U: harm}, Nh=Nh, omega=om,
                       label="diffusion, periodic coefficient")
    t1 = time.perf_counter()
    w = exponents_dense(H, finite_cap=1e4)
    t2 = time.perf_counter()
    strip = w[np.abs(w.imag) <= om / 2]
    lead = strip[np.argsort(-strip.real)][:3]
    exact = -(m_ * a_ + 1 + d_ * (np.arange(1, 4) * np.pi / 2) ** 2) / m_
    err = float(max(np.abs(lead.real - exact).max(), np.abs(lead.imag).max()))
    acc = _acc("three leading Floquet exponents", [float(v) for v in lead.real],
               [float(v) for v in exact], err, 1e-10, "closed form (separable solution)")
    return dict(problem=dict(N=N, Nh=Nh, omega=om, unknowns=int((2 * Nh + 1) * N),
                             basis="Chebyshev x Fourier in time (Hill)"),
                timings=dict(setup_s=t1 - t0, run_s=t2 - t1), accuracy=acc)


CASES: dict[str, Callable[[str], dict]] = {
    "ivp1d_burgers": ivp1d_burgers,
    "ivp2d_taylor_green": ivp2d_taylor_green,
    "ivp2d_rb": ivp2d_rb,
    "ivp3d_rb": ivp3d_rb,
    "lbvp_poisson": lbvp_poisson,
    "evp_dense": evp_dense,
    "evp_shift_invert": evp_shift_invert,
    "continuation_bratu": continuation_bratu,
    "floquet_diffusion": floquet_diffusion,
}


if __name__ == "__main__":
    # every case at its default size, with its accuracy check (the runner suite.py adds timings
    # in separate processes and the hardware and version records)
    for name, case in CASES.items():
        acc = case("default")["accuracy"]
        print(f"{name:22s} error {acc['error']:.2e} (tolerance {acc['tolerance']:.0e})")
        assert acc["passed"], (name, acc)
