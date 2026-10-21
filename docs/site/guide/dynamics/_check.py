"""Executable snippets of the user-guide part "Dynamics" (docs/site/guide/dynamics/*.md).

Every code block of those pages is a region of this file (``# [name]`` ... ``# [/name]``), included
verbatim with ``literalinclude`` (``:dedent: 4``): what the reader sees is what ran. One function
per page; the regions of one page build on each other, in reading order. The asserts at the end of
each function are the checks (not shown on the pages). Not a Sphinx source (leading underscore).

Run (CPU, a few minutes; as a job, not on the login node):
    python docs/site/guide/dynamics/_check.py [page ...]       # only some pages
"""
from __future__ import annotations

import sys
import time
import warnings


def eigenvalues() -> None:
    # [eig-model]
    import numpy as np

    import conatus as cn

    m = cn.Model(cn.Box(y=cn.walls(-1, 1)), name="Orr-Sommerfeld", Re=1e4, alpha=1.0)
    m.scalar("v")
    m.given(U=lambda y: 1 - y**2)
    m.eigenvalue("c", growth="imag")
    m.define("L2(f) = dy(dy(f)) - alpha**2*f")
    m.equations("L2(L2(v))/(1j*alpha*Re) - U*L2(v) + dy(dy(U))*v + c*L2(v) = 0")
    m.at("walls", "v = 0", "dy(v) = 0")
    # [/eig-model]

    # [eig-solve]
    res = m.eigenvalues(N=64, drift=96, drift_tol=1e-6)   # keep modes found again at N = 96
    print(res.explain(key="imag"))                        # filter log, residual of every mode
    i = int(res.values.imag.argmax())
    print(res.values[i], res.residuals[i])
    # [/eig-solve]
    assert abs(res.values[i] - (0.23752649 + 0.00373967j)) < 1e-8 and res.residuals[i] < 1e-12

    # [eig-mode]
    mode = res.mode(i)                                    # eigenfunction on the grid, no taus
    y, v = mode.coordinate("y"), mode["v"]                # Chebyshev points, complex v(y)
    print(mode.explain())                                 # declared normalisation
    # [/eig-mode]
    assert abs(abs(v).max() - 1) < 1e-14 and v.shape == y.shape

    # [eig-linearize]
    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
    rb = cn.Model(box, name="Rayleigh-Benard", Ra=1708, Pr=1)
    u, p, T = rb.velocity("u"), rb.pressure("p"), rb.scalar("T")
    rb.equations("""
        dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        div(u) = 0
        dt(T) - lap(T) = -u·grad(T)
    """)
    rb.at("bottom", u=0, T=+0.5)
    rb.at("top", u=0, T=-0.5)

    lin = rb.linearize(about={u: 0, T: lambda z: -z})     # conduction state
    modes = lin.eigenvalues(N=24, k=np.pi)                # normal modes exp(s t) of wavenumber pi
    print(lin.explain())
    print(modes.explain(n=3))
    # [/eig-linearize]
    assert abs(modes.values.real.max()) < 2e-3

    # [eig-critical]
    from conatus.expert import critical

    # L(Re, alpha) + c M(Re, alpha); tau modes rejected by the same drift test at N = 96
    P = m.parametric(N=64, over=("Re", "alpha"), drift=96, drift_tol=1e-6)
    crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-9)
    print(crit.explain())                                 # growth = Im c (phase speed), drift log
    print(f"Re_c = {crit.value:.2f}, alpha_c = {crit.over['alpha']:.5f}")
    # [/eig-critical]
    assert abs(crit.value - 5772.22) < 0.01

    # [eig-left]
    left = m.eigenvalues(N=64, left=True)                 # right and left eigenvectors
    j = int(left.values.imag.argmax())
    mode = P.least_stable(Re=1e4, alpha=1.0)
    print(f"dc/dRe = {P.sensitivity(mode, 'Re'):.3e}")    # from the left and right eigenvectors
    # [/eig-left]
    assert left.left is not None and left.residuals[j] < 1e-12


def onset() -> None:
    # [onset-ready]
    import numpy as np

    import conatus as cn
    import conatus.expert as X

    # Rayleigh-Benard between no-slip walls, linearized about the conduction state T = -z
    lin = cn.problems.native.rayleigh_benard_evp(k=3.1, Ra=1708.0, Pr=1.0)
    # Ra and the wavenumber k stay symbolic in the matrices; drift=32 rejects spurious tau modes
    P = lin.parametric(24, over=("Ra", "k"), drift=32, drift_tol=1e-6)
    crit = X.critical(P, "Ra", bracket=(1500.0, 2000.0), over={"k": (2.5, 3.8)}, xtol=1e-9)
    print(crit)                                         # one line; crit.explain() for the log
    # [/onset-ready]
    assert abs(crit.value - 1707.7618) < 1e-3 and abs(crit.over["k"] - 3.11632) < 1e-4

    # [onset-mode]
    kc = crit.over["k"]
    lin_c = cn.problems.native.rayleigh_benard_evp(k=kc, Ra=crit.value, Pr=1.0)
    res = lin_c.eigenvalues(24, k=kc, drift=32, drift_tol=1e-6)
    i = int(res.order()[0])                             # least stable first: growth = Re s
    print(res.values[i], res.residuals[i])              # s = 0 at onset, with its residual
    md = res.mode(i)
    z = np.linspace(-0.5, 0.5, 101)                     # walls included
    T, u = md.at("T", z=z)[0], md.at("u", z=z)[:, 0]    # exact evaluation; row x = 0
    x = np.linspace(0.0, 2 * np.pi / kc, 64)            # one wavelength
    T_xz = (T[None, :] * np.exp(1j * kc * x[:, None])).real   # the mode exp(i k x) T(z)
    # import matplotlib.pyplot as plt; plt.contourf(x, z, T_xz.T, 21, cmap="RdBu_r")
    # [/onset-mode]
    assert abs(res.values[i]) < 1e-8 and res.residuals[i] < 1e-12
    assert np.abs(T[[0, -1]]).max() < 1e-12 and np.abs(u[:, [0, -1]]).max() < 1e-12
    assert T_xz.shape == (64, 101)

    # [onset-primitive]
    m = cn.Model(cn.Box(z=cn.walls(-0.5, 0.5)), name="RB onset", Ra=1708.0, Pr=1.0, k=3.1)
    m.scalar("u"), m.scalar("w"), m.pressure("p"), m.scalar("T")   # amplitudes of exp(i k x + s t)
    m.eigenvalue("s", growth="real")
    m.define("L(f) = dz(dz(f)) - k**2*f")
    m.equations("s*u + 1j*k*p - Pr*L(u) = 0",
                "s*w + dz(p) - Pr*L(w) - Ra*Pr*T = 0",
                "1j*k*u + dz(w) = 0",
                "s*T - w - L(T) = 0")
    m.at("walls", u=0, w=0, T=0)                        # no condition on the pressure
    P2 = m.parametric(24, over=("Ra", "k"), drift=32, drift_tol=1e-6)
    crit2 = X.critical(P2, "Ra", bracket=(1500.0, 2000.0), over={"k": (2.5, 3.8)}, xtol=1e-9)
    print(crit2)
    # [/onset-primitive]
    assert abs(crit2.value - crit.value) < 1e-6 * crit.value
    assert "Normal-mode pressure recognized" in m.explain(24, full=True)


def nonmodal() -> None:
    # [nm-system]
    from conatus.expert import Y, max_growth, parallel_shear_flow, state_space

    # Plane Poiseuille flow, streamwise-constant mode kx = 0, kz = 2, Re = 1000
    flow = parallel_shear_flow(1 - Y**2, kx=0.0, kz=2.0, Re=1000.0, walls="no-slip")
    desc = flow.discretize(48)                     # 48 Chebyshev coefficients per variable
    ss = state_space(desc, infinite_tol=1e-10)     # energy-orthonormal generator
    print(ss.explain())
    # [/nm-system]

    # [nm-growth]
    G, t_opt = max_growth(ss, (1.0, 200.0), xtol=1e-6, n_scan=60)
    print(f"G_max = {G:.2f} at t = {t_opt:.1f}")
    # [/nm-growth]
    assert abs(G - 196.0) < 0.5            # Reddy & Henningson (1993)




def invariant() -> None:
    # [inv-model]
    import numpy as np

    import conatus as cn
    from conatus.expert import (NewtonKrylovHookstep, equilibrium_eigenvalues, find_equilibrium,
                                find_travelling_wave)

    L = 22.0
    m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=0.01)        # the flow map used by the searches

    newton = NewtonKrylovHookstep(tol=1e-12)    # accuracy required; algorithm controls defaulted
    print(newton.explain())                     # budget, GMRES, trust radii, marked "(default)"
    # [/inv-model]

    # [inv-eq]
    q = 2 * np.pi / L
    sim.initial(u=lambda x: np.sin(q * x))                 # odd seed: u(-x) = -u(x)
    E1 = find_equilibrium(sim, sim.solver.X.copy(), newton)   # the Simulation itself
    print(E1.explain())
    print(equilibrium_eigenvalues(sim.solver, E1, n_modes=4, dense_max=500, arnoldi_tol=1e-12))
    # [/inv-eq]
    assert E1.residual < 1e-12 and abs(E1.energy - 0.2609) < 6e-5

    # [inv-tw]
    sim.initial(u=lambda x: np.cos(q * x) + np.sin(2 * q * x))
    TW1 = find_travelling_wave(sim.solver, sim.solver.X.copy(), 0.7, newton)
    print(f"c = {TW1.params['c']:.5f}, residual {TW1.residual:.1e}")
    # [/inv-tw]
    assert TW1.residual < 1e-12 and abs(TW1.params["c"] - 0.73699) < 6e-6

    # [inv-threshold]
    from conatus.expert import ThresholdBisection

    def fate(a):
        """Declared verdict for amplitude a: x' = -x (x - 1)(x - 2), x(0) = a, t in [0, 40]."""
        x = a
        for _ in range(4000):
            x += 0.01 * (-x * (x - 1) * (x - 2))
        return -1 if x < 0.5 else (+1 if x > 1.5 else 0)

    tb = ThresholdBisection(fate, rtol=1e-4)
    lo, hi = tb.run(0.5, 1.6)                    # the threshold as a bracket ]lo, hi]
    print(tb.explain())
    # [/inv-threshold]
    assert lo < 1.0 <= hi and (hi - lo) / hi <= 1e-4


def continuation() -> None:
    # [cont-model]
    import numpy as np

    import conatus as cn
    from conatus.expert import ContinuationSettings, NewtonSolver, NLBVPContinuation

    m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3)
    m.scalar("u")
    m.equations("dx(dx(u)) + lam*exp(u) = 0")
    m.at("walls", u=0)
    compiled = m.compile(48, dealias=1)        # a nonlinear boundary-value problem, N = 48
    problem, u = compiled.problem, compiled.fields["u"]
    # [/cont-model]

    # [cont-newton]
    newton = NewtonSolver(problem, tol=1e-13, damping=1.0, max_iterations=20,
                          params={"lam": 0.3})
    sol = newton.solve({u: lambda x: 0 * x})                # initial guess u = 0
    print(sol.explain())                                    # residual per iteration, orders
    # [/cont-newton]
    assert sol.converged and sol.residual < 1e-12

    # [cont-branch]
    settings = ContinuationSettings(ds=0.2, ds_min=1e-8, ds_max=0.6, adapt=1.5,
                                    fast_iterations=3, tolerance=1e-12, max_newton=15,
                                    max_points=12, dp_fd=1e-6)
    branch = NLBVPContinuation(newton, "lam", settings).run(sol.coeffs, 0.3, +1,
                                                            p_bounds=(0.3, 1.0))
    (fold,) = branch.folds()
    print(branch.explain())
    print(f"fold at lam = {fold.p:.12f}")
    # [/cont-branch]
    assert abs(fold.p - 0.878457679781) < 1e-9

    # [cont-ode]
    from conatus.expert import continue_equilibria, ir_system, ode_problem

    lorenz = ode_problem(["x", "y", "z"], {"sigma": 10.0, "r": 0.5, "b": 8 / 3},
                         lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                       P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                       X["x"] * X["y"] - P["b"] * X["z"]], name="Lorenz")
    L = ir_system(lorenz, ["r"])            # G(x; r), exact Jacobian, dG/dr, multilinear forms
    cfg = ContinuationSettings(ds=0.1, ds_min=1e-9, ds_max=2.0, adapt=1.3, fast_iterations=4,
                               tolerance=1e-11, max_newton=12, max_points=60, dp_fd=None)
    origin = continue_equilibria(L, np.zeros(3), "r", cfg, marginal_tol=1e-9, bounds=(0, 3))
    (bp,) = origin.events_of("branch point")
    # [/cont-ode]
    assert abs(bp.point.z[-1] - 1) < 1e-12

    # [cont-switch]
    from conatus.expert import CollocationMesh, branch_switch, normal_form

    S = np.diag([-1.0, -1.0, 1.0])          # (x, y, z) -> (-x, -y, z), a symmetry of Lorenz
    cplus = branch_switch(origin, bp, cfg, step=0.05, symmetry=S, bounds=(0, 30))
    (hopf,) = cplus.events_of("hopf")
    hf = normal_form(L, hopf.point.z[:-1], kind="hopf", par={"r": hopf.point.z[-1]},
                     parameter="r")
    print(hf.explain())                     # first Lyapunov coefficient: sub- or supercritical

    cfo = ContinuationSettings(ds=0.2, ds_min=1e-9, ds_max=1.0, adapt=1.3, fast_iterations=4,
                               tolerance=1e-9, max_newton=12, max_points=12, dp_fd=None)
    cycles = branch_switch(cplus, hopf, cfo, amplitude=0.2,
                           mesh=CollocationMesh.uniform(40, 4), tests=("lpc",))
    print(cycles.explain())                 # periodic orbits born at the Hopf point
    # [/cont-switch]
    sig, b = 10.0, 8 / 3
    assert abs(hopf.point.z[-1] - sig * (sig + b + 3) / (sig - b - 1)) < 1e-9 and hf.l1 > 0

    # [cont-codim2]
    from conatus.expert import continue_codim2

    bt = ir_system(ode_problem(["x", "y"], {"b1": -0.5, "b2": -1.0},
                               lambda X, P: [X["y"], P["b1"] + P["b2"] * X["x"]
                                             + X["x"] * X["x"] - X["x"] * X["y"]],
                               name="Bogdanov-Takens"), ["b1", "b2"])
    cfg2 = ContinuationSettings(ds=0.05, ds_min=1e-9, ds_max=0.3, adapt=1.3, fast_iterations=3,
                                tolerance=1e-11, max_newton=12, max_points=40, dp_fd=None)
    eq = continue_equilibria(bt, np.array([(1 - np.sqrt(3)) / 2, 0.0]), "b1", cfg2,
                             marginal_tol=1e-9, bounds=(-2, 2))
    fold_pt = eq.events_of("fold")[0]
    folds = continue_codim2(bt, fold_pt.point.z[:-1], "fold", ("b1", "b2"), cfg2,
                            par={"b1": fold_pt.point.z[-1]}, bounds={"b2": (-1.5, 1.5)})
    print(folds.explain())                  # the fold curve b1 = b2^2 / 4 in the (b1, b2) plane
    # [/cont-codim2]
    assert np.abs(folds.column("b1") - folds.column("b2") ** 2 / 4).max() < 1e-10


def floquet() -> None:
    # [fl-mathieu]
    import numpy as np

    from conatus.expert import HillBlocks, exponents_dense

    a, q = 1.5, 0.5                      # Mathieu x'' + (a - 2q cos 2t) x = 0, period pi
    L0 = np.array([[0, -1], [a, 0]], complex)
    C = np.array([[0, 0], [-q, 0]], complex)   # -2q cos 2t = -q (e^{2it} + e^{-2it})
    H = HillBlocks(diag=lambda mu, n: L0 + (mu + 2j * n) * np.eye(2), couplings={1: C, -1: C},
                   Nh=24, size=2, label="Mathieu", omega=2.0, mass=np.eye(2))
    mus = exponents_dense(H, finite_cap=1e3)   # every Floquet exponent, modulo i omega
    print(H.explain())
    # [/fl-mathieu]
    assert np.abs(np.sort(np.abs(mus.real))[:2]).max() < 1e-10   # stable point: Re mu = 0



def chaos() -> None:
    # [ly-system]
    import numpy as np

    from conatus.bifurcation import ode_problem
    from conatus.chaos import lyapunov_spectrum
    from conatus.discretize import discretize
    from conatus.timestep import IVPSolver

    lorenz = ode_problem(["x", "y", "z"], {"sigma": 10.0, "r": 28.0, "b": 8 / 3},
                         lambda X, P: [P["sigma"] * (X["y"] - X["x"]),
                                       P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                                       X["x"] * X["y"] - P["b"] * X["z"]], name="Lorenz")
    s = IVPSolver(discretize(lorenz), "RK443", 0.01)      # scheme and step: your choice
    # [/ly-system]

    # [ly-spectrum]
    rng = np.random.default_rng(0)
    X0 = np.column_stack([rng.uniform(-10, 10, 16), rng.uniform(-10, 10, 16),
                          rng.uniform(10, 40, 16)])          # 16 independent trajectories
    res = lyapunov_spectrum(s, X0, n_vectors=3, transient_steps=1000,
                            tangent_transient_steps=1000, steps=5000, reortho_every=10,
                            inner="euclidean", n_blocks=5, seed=1)
    print(res.explain())                                     # exponents with standard errors
    D, dD = res.kaplan_yorke()
    # [/ly-spectrum]
    assert abs(res.exponents.sum() + 13.6667) < 1e-2 and abs(res.exponents[0] - 0.906) < 0.05

    # [ly-ensemble]
    from conatus.ensembles import Ensemble, ensemble_fate

    s20 = IVPSolver(discretize(ode_problem(
        ["x", "y", "z"], {"sigma": 10.0, "r": 20.0, "b": 8 / 3},
        lambda X, P: [P["sigma"] * (X["y"] - X["x"]), P["r"] * X["x"] - X["y"] - X["x"] * X["z"],
                      X["x"] * X["y"] - P["b"] * X["z"]], name="Lorenz r=20")), "RK443", 0.01)
    c = np.sqrt(8 / 3 * 19.0)
    Cp, Cm = np.array([c, c, 19.0]), np.array([-c, -c, 19.0])   # the two stable equilibria
    starts = np.column_stack([rng.uniform(-20, 20, 256), rng.uniform(-20, 20, 256),
                              np.full(256, 19.0)])
    ens = Ensemble(s20, len(starts), rhs="batched")             # 256 members in one batch
    ens.set_states(starts)

    def attractor(e):                                           # declared classifier
        X = e.states()
        return np.where(np.linalg.norm(X - Cp, axis=1) < 1, 1,
                        np.where(np.linalg.norm(X - Cm, axis=1) < 1, -1, 0))

    fate = ensemble_fate(ens, attractor, check_every=100, min_steps=100, max_steps=10000,
                         classifier="|X - C+-| < 1")
    print(fate.explain())                                       # fractions with intervals
    # [/ly-ensemble]
    assert (fate.labels != 0).mean() > 0.95

    # [ly-native]
    import numpy as np

    import conatus as cn

    L = 22.0
    m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky L=22")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=0.05)
    sim.initial(u=lambda x: 0.3 * np.cos(2 * np.pi * x / L) + 0.2 * np.sin(4 * np.pi * x / L))
    res = sim.lyapunov(12, transient=100.0, align=50.0, T=200.0, reortho_every=10,
                       inner="euclidean", n_blocks=4, seed=1, zero_mean="u", covariant=True,
                       clv_window=40.0, clv_backward=40.0, clv_checkpoints=8)
    print(res.explain())
    dom = res.domination(5.0)
    print(dom.explain())
    # [/ly-native]
    # one short trajectory: lambda_1 = 0.047 within its sampling error, the zero-mean subspace
    # removes the trivial mean direction, the domination clusters cover the 12 vectors
    assert 0.0 < res.exponents[0] < 0.1 and res.exponents[-1] < -1.0
    assert res.clv is not None and dom.n_physical >= 1


def manifolds() -> None:
    # [man-setup]
    import numpy as np

    import conatus as cn
    from conatus.expert import NewtonKrylovHookstep, find_equilibrium, find_travelling_wave

    L = 22.0
    q = 2 * np.pi / L
    m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=0.01)
    newton = NewtonKrylovHookstep(tol=1e-12, max_iterations=40, gmres_rtol=1e-3,
                                  gmres_max_iterations=60, delta0=1.0, delta_min=1e-10, delta_max=10.0)
    sim.initial(u=lambda x: np.sin(q * x))
    E1 = find_equilibrium(sim.solver, sim.solver.X.copy(), newton)
    sim.initial(u=lambda x: np.cos(q * x) + np.sin(2 * q * x))
    TW1 = find_travelling_wave(sim.solver, sim.solver.X.copy(), 0.7, newton)
    # [/man-setup]

    # [man-unstable]
    from conatus.expert import ecs_system, unstable_manifold

    ks = ecs_system(sim.solver, odd=True, degree=2, dense_max=400)   # active coordinates
    x0 = ks.space.active(E1.X)
    W = unstable_manifold(ks, x0, order=6, degree=2, eigvec_scale=0.1, unstable_tol=1e-8)
    print(W.explain())
    print(f"defect on the circle |s| = 0.5: {W.max_defect(0.5, 32):.1e}")
    # [/man-unstable]
    assert W.dim >= 1

    # [man-slice]
    from conatus.expert import FirstFourierModeSlice, FourierOps, slice_equilibrium_check

    slc = FirstFourierModeSlice(FourierOps(sim.solver), border_tol=1e-6)
    check = slice_equilibrium_check(slc, sim.solver, TW1)   # a dict of evidence
    print(check)
    # [/man-slice]


def adjoint() -> None:
    # [adj-flowmap]
    import numpy as np

    import conatus as cn
    from conatus.expert import FlowMap

    L = 22.0
    m = cn.Model(cn.Box(x=cn.periodic(L)), name="Kuramoto-Sivashinsky")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    sim = m.simulate(N=32, scheme="RK443", dt=0.01)
    sim.initial(u=lambda x: np.cos(2 * np.pi * x / L) + 0.3 * np.sin(4 * np.pi * x / L))

    phi = FlowMap(sim.solver, steps=200)          # X_N = Phi(X_0) over 200 steps
    X0 = sim.solver.X.copy()
    rng = np.random.default_rng(1)
    V, W = rng.normal(size=X0.size), rng.normal(size=X0.size)
    jv = phi.jvp(X0, V)                            # tangent: dPhi V
    vj = phi.vjp(X0, W)                            # adjoint of the discrete map: dPhi^T W
    print(f"<W, J V> - <J^T W, V> = {W @ jv - vj @ V:.1e}")
    print(phi.explain())
    # [/adj-flowmap]
    assert abs(W @ jv - vj @ V) < 1e-10 * abs(W @ jv)

    # [adj-continuous]
    from conatus.expert import Y, compare_adjoints, parallel_shear_flow

    flow = parallel_shear_flow(1 - Y**2, kx=1.0, kz=0.0, Re=2000.0, walls="no-slip")
    bcs = [(f, 0, side) for f in ("u", "v", "w") for side in ("left", "right")]
    cmp = compare_adjoints(flow, bcs, 64, target=-0.05 + 0.3j, infinite_tol=1e-10)
    print(cmp.explain())                           # discrete adjoint vs continuous adjoint
    # [/adj-continuous]

    # [adj-sensitivity]
    from conatus.expert import (Y, base_flow_sensitivity, ginzburg_landau_exact,
                                linear_ginzburg_landau, structural_sensitivity)

    NU, GAMMA, MU_C, MU_2 = 2 + 0.4j, 1 - 1j, 0.34, -0.01
    exact = ginzburg_landau_exact(nu=NU, gamma=GAMMA, mu_c=MU_C, mu2=MU_2, n=0)

    def gl(mu):                                    # dq/dt = -nu q' + gamma q'' + mu(x) q
        return linear_ginzburg_landau(nu=NU, gamma=GAMMA, mu=mu, bounds=(-60.0, 60.0))

    bfs = base_flow_sensitivity(gl, MU_C + MU_2 * Y**2 / 2, N=140, target=exact.eigenvalue,
                                adjoint="continuous", infinite_tol=1e-10,
                                adjoint_bcs=[("q", 0, "left"), ("q", 0, "right")])
    x_star, s_star = structural_sensitivity(bfs.pair).maximum(points=1201)   # wavemaker
    print(f"wavemaker maximum {s_star:.6f} at x = {x_star:.1e}")
    print(f"d lambda / d mu_2 = {bfs.predict(Y**2 / 2):.6f}")
    # [/adj-sensitivity]
    assert abs(s_star - abs(exact.chi) / np.sqrt(np.pi)) < 1e-9 * s_star


def datadriven() -> None:
    # [dd-setup]
    import numpy as np
    import scipy.linalg as sla

    from conatus.diagnostics import Chebyshev, Grid
    from conatus.expert import DataInnerProduct, Y, linear_ginzburg_landau, state_space

    small = linear_ginzburg_landau(nu=2 + 0.4j, gamma=1 - 1j, mu=-0.2 - 0.01 * Y**2 / 2,
                                   bounds=(-20, 20))
    ss = state_space(small.discretize(32), infinite_tol=1e-10)
    grid = Grid([Chebyshev(-20.0, 20.0, 64)])              # snapshots on 64 Gauss nodes
    E = np.polynomial.chebyshev.chebvander(grid.axes[0].nodes() / 20.0, 31) @ ss.to_state
    ip = DataInnerProduct.from_grid(grid, components=["q"], weights={"q": 1.0})
    # [/dd-setup]

    # [dd-dmd]
    from conatus.expert import dmd

    lam, V = np.linalg.eig(ss.A)
    lead = np.argsort(-lam.real)[:4]
    q0 = V[:, lead] @ np.array([1.0, 0.5j, 0.3, 0.2 - 0.1j])
    Phi = sla.expm(ss.A * 0.5)
    X = E @ np.stack([np.linalg.matrix_power(Phi, j) @ q0 for j in range(40)], axis=1)
    r = dmd(X, dt=0.5, rank=4, inner_product=ip, amplitudes="optimal")
    print(r.explain())
    # [/dd-dmd]
    for mu in lam[lead]:
        assert np.min(np.abs(r.omega - mu)) < 1e-9



PAGES = {"eigenvalues": eigenvalues, "onset": onset, "nonmodal": nonmodal, "invariant": invariant,
         "continuation": continuation, "floquet": floquet, "chaos": chaos,
         "manifolds": manifolds, "adjoint": adjoint, "datadriven": datadriven}

if __name__ == "__main__":
    warnings.simplefilter("ignore")                     # resolution warnings are not the point
    failed = []
    for name in sys.argv[1:] or list(PAGES):
        t0 = time.perf_counter()
        try:
            PAGES[name]()
            print(f"== PASS {name} ({time.perf_counter() - t0:.1f} s)", flush=True)
        except Exception as e:  # noqa: BLE001 -- report every page, then fail
            import traceback
            traceback.print_exc()
            print(f"== FAIL {name}: {type(e).__name__}: {e}", flush=True)
            failed.append(name)
    print("FAILED: " + ", ".join(failed) if failed else "all dynamics snippets passed")
    sys.exit(1 if failed else 0)
