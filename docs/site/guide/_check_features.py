"""Executable examples of the user guide for the features added in the parity and
Studio lots: sine/cosine walls and free slip, the whole and half line and compound intervals, the
new time steppers, the native tangent/adjoint/NLOP of a simulation, and the Studio's ``sim.watch``.

Every code block of these examples on the pages is a region of this file (``# [name]`` ...
``# [/name]``), included verbatim with ``literalinclude`` (``:dedent: 4``), and what the example
printed is written to ``docs/site/guide/outputs/<function>.txt`` (with ``--sync``), included on the
page as is: what the reader sees is what ran. The asserts after each example are the checks (not
shown on the pages). Not a Sphinx source (leading underscore).

Run (CPU, a few minutes; as a job, not on the login node):
    python docs/site/guide/_check_features.py [name ...] [--sync]
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import time
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
TMP = Path(tempfile.mkdtemp(prefix="cn_guide_feat_"))


def sincos_free_slip() -> None:
    """bases.md, *Sine/cosine walls*; boundary_conditions.md, *Free-slip walls*."""
    # [sincos]
    import numpy as np

    import conatus as cn

    kc = np.pi / np.sqrt(2)                     # critical wavenumber of stress-free RB
    Ra_c = 27 * np.pi**4 / 4                    # 657.511...

    def model(basis, Ra):
        box = cn.Box(x=cn.periodic(2 * np.pi / kc), z=cn.walls(0, 1, basis=basis))
        m = cn.Model(box, name="RB stress-free", Ra=Ra, Pr=1)
        u, T = m.velocity("u"), m.scalar("T")
        m.pressure("p")
        m.equations("""
            momentum:   dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*T*ez = -u·grad(u)
            continuity: div(u) = 0
            heat:       dt(T) - lap(T) - u·ez = -u·grad(T)
        """)
        m.at("walls", u=cn.free_slip, T=0)      # stress-free, fixed temperature (perturbation)
        return m, u, T

    for Ra in (0.99 * Ra_c, 1.01 * Ra_c):
        m, u, T = model("sincos", Ra)
        s = m.linearize(about={u: 0, T: 0}).eigenvalues(8, k=kc, n=1)
        print(f"sincos     Ra/Ra_c = {Ra / Ra_c:.2f}: s = {s.values.real.max():+.15f}")
        m, u, T = model("chebyshev", Ra)
        s = m.linearize(about={u: 0, T: 0}).eigenvalues(16, k=kc)
        print(f"chebyshev  Ra/Ra_c = {Ra / Ra_c:.2f}: s = {s.values.real.max():+.15f}")
    # [/sincos]
    # closed form (s/Pr + K^2)(s + K^2) K^2 = Ra k^2, K^2 = k^2 + pi^2 (Pr = 1)
    K2 = kc**2 + np.pi**2
    for Ra in (0.99 * Ra_c, 1.01 * Ra_c):
        exact = -K2 + np.sqrt(Ra * kc**2 / K2)
        m, u, T = model("sincos", Ra)
        s = m.linearize(about={u: 0, T: 0}).eigenvalues(8, k=kc, n=1).values.real.max()
        assert abs(s - exact) < 1e-12, (s, exact)
    # [sincos-explain]
    m, u, T = model("sincos", Ra_c)
    print("\n".join(l.strip() for l in m.explain().splitlines() if l.lstrip().startswith(("z:", "u:", "p:", "T:"))))
    m, u, T = model("chebyshev", Ra_c)
    print("\n".join(l.strip() for l in m.explain().splitlines() if ("free-slip" in l and "tau" in l) or l.strip().startswith("divergence tau")))
    # [/sincos-explain]
    assert "no tau" in model("sincos", Ra_c)[0].explain()


def unbounded_compound() -> None:
    """bases.md, *Whole line, half line, compound intervals*."""
    # [unbounded]
    import numpy as np

    import conatus as cn

    # whole line: quantum harmonic oscillator, E_n = 2n + 1
    m = cn.Model(cn.Box(x=cn.line(scale=1.0)), name="harmonic oscillator")
    m.scalar("u")
    m.eigenvalue("E", growth="real")
    m.given(V=lambda x: x**2)                     # polynomial coefficient: banded, exact
    m.equations("E*u + lap(u) - V*u = 0")
    E = np.sort(m.eigenvalues(N=40).values.real)
    print("E_0..4 =", np.round(E[:5], 12), " max error (n < 20):",
          f"{np.max(abs(E[:20] - (2*np.arange(20) + 1))):.1e}")

    # half line: heat equation with u(0) = 0, CNAB2
    m = cn.Model(cn.Box(y=cn.half_line(scale=0.5)), name="heat on y >= 0")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = 0")
    m.at("wall", u=0)
    exact = lambda y, t: y / (1 + t)**1.5 * np.exp(-y**2 / (4*(1 + t)))   # noqa: E731
    sim = m.simulate(N=40, scheme="CNAB2", dt=1e-3)
    sim.initial(u=lambda y: exact(y, 0))
    sim.run(until=0.5)
    err = np.max(abs(sim.grid("u") - exact(sim.coordinate("y"), sim.t)))
    print(f"half line, t = 0.5: max error {err:.1e}")

    # compound interval: an interior layer tanh(20 x), same total N = 192
    k = 20.0
    for label, axis, N in (("walls(-1, 1)", cn.walls(-1, 1), 192),
                           ("walls(-1, -0.15, 0.15, 1)", cn.walls(-1, -0.15, 0.15, 1), (64, 64, 64))):
        m = cn.Model(cn.Box(x=axis), name="layer")
        m.scalar("u")
        m.given(f=lambda x: -2*k*k*np.tanh(k*x) / np.cosh(k*x)**2)
        m.equations("lap(u) = f")
        m.at("bottom", u=np.tanh(-k))
        m.at("top", u=np.tanh(k))
        sol = m.solve(N=N)                        # linear boundary-value model: direct solve
        print(f"{label:>26}: max error {np.max(abs(sol.grid('u') - np.tanh(k*sol.coordinate('x')))):.1e}")
    # [/unbounded]
    assert np.max(abs(E[:20] - (2 * np.arange(20) + 1))) < 1e-12
    assert err < 1e-6
    # [compound-explain]
    print("\n".join(l.strip() for l in m.explain(N=(64, 64, 64)).splitlines()
                    if l.strip().startswith(("x:", "u = ", "u continuous", "dx(u) continuous"))))
    # [/compound-explain]


def schemes() -> None:
    """timesteppers.md, *Schemes*: observed orders of the CN family and of RKSMR / RKGFY."""
    # [schemes]
    import numpy as np

    import conatus as cn

    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat with a nonlinear source")
    m.scalar("u")
    m.equations("dt(u) - lap(u) = -u*u")          # implicit lap(u), explicit -u^2

    def final(scheme, dt):
        sim = m.simulate(N=16, scheme=scheme, dt=dt)
        sim.initial(u=lambda x: 0.5 + 0.1 * np.cos(x))
        sim.run(until=1.0)
        return sim.grid("u")

    ref = final("RK443", 1e-4)                    # reference: third order, 10x finer
    for scheme in ("CNAB1", "CNAB2", "MCNAB2", "CNLF2", "RKSMR", "RKGFY"):
        e1, e2 = (np.max(abs(final(scheme, dt) - ref)) for dt in (2e-2, 1e-2))
        print(f"{scheme:>6}: error {e2:.2e} at dt = 0.01, observed order {np.log2(e1 / e2):.2f}")
    # [/schemes]
    for scheme, q in (("CNAB1", 1), ("CNAB2", 2), ("MCNAB2", 2), ("CNLF2", 2), ("RKSMR", 2), ("RKGFY", 2)):
        e1, e2 = (np.max(abs(final(scheme, dt) - ref)) for dt in (2e-2, 1e-2))
        assert abs(np.log2(e1 / e2) - q) < 0.25, (scheme, np.log2(e1 / e2))


def flowmap_nlop() -> None:
    """dynamics/optimal_perturbations.md, *In the native layer* (spherical shell)."""
    # [nlop-shell]
    import conatus as cn

    m = cn.Model(cn.Shell(0.5, 1.5), name="shell", nu=1.0)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)\ndiv(u) = 0")
    m.at("inner", u=0)
    m.at("outer", u=0)

    sim = m.simulate((4, 5, 16), scheme="RK443", dt=1e-3)   # (Nphi, Lmax, Nr)
    fm = sim.flowmap(0.1)                     # T = 0.1 = 100 steps (never rounded)
    nl = sim.nlop(T=0.1, E0=1e-10)            # energy 1/2 int |u|^2 dV of the velocity
    G, seeds = nl.linear_gain(n_modes=2, tol=1e-12)  # linear optimal gains (E0 -> 0 limit)
    st = nl.iterate(nl.start(seeds[0], alpha=0.5), max_iterations=20, rtol=1e-6)
    print(f"linear optimal gains G = {G[0]:.10f}, {G[1]:.10f}")
    print(st.explain())
    # [/nlop-shell]
    assert fm is not None and G[0] >= G[1] > 0
    gain = st.history[-1]["gain"]                 # E0 -> 0: the NLOP gain is the linear one
    assert abs(gain - G[0]) < 1e-6 * G[0], (gain, G[0])


def shell_native() -> None:
    """shells.md: a spherical shell in the native layer, at a radial resolution (N_r = 48) and a
    viscosity where the textbook discretisation has a growing artefact; the native
    recipe has none, and the check before the first step says so."""
    # [shell-native]
    import numpy as np

    import conatus as cn

    shell = cn.Shell(0.35, 1.0)                  # inner and outer radius
    m = cn.Model(shell, name="viscous shell", nu=1e-4)
    m.velocity("u")
    m.pressure("p")
    m.equations("dt(u) + grad(p) - nu*lap(u) = -u@grad(u)\ndiv(u) = 0")
    m.at("walls", u=0)                           # no slip on both spheres

    sim = m.simulate((8, 5, 48), scheme="RK443", dt=1e-3)   # (N_phi, L_max, N_r)
    rng = np.random.default_rng(0)
    sim.initial(u=lambda phi, theta, r: 1e-3 * rng.standard_normal((3, *np.broadcast(phi, theta, r).shape))
                * (r - 0.35) * (1.0 - r))
    sim.run(steps=200)
    for line in sim.explain().splitlines():
        if line.startswith("Simulation") or line.strip().startswith("spurious"):
            print(line.strip())
    print(f"t = {sim.t:.3f}, max |u| = {np.max(np.abs(sim.grid('u'))):.3e}")
    # [/shell-native]
    assert "spurious tau modes: none" in sim.explain()
    assert np.isfinite(sim.grid("u")).all() and np.max(np.abs(sim.grid("u"))) < 1e-2


def studio_watch() -> None:
    """studio.md: declare what the Studio may show, run, read back what was sent."""
    spool_dir = TMP / "spool"
    # [watch]
    import numpy as np

    import conatus as cn

    box = cn.Box(x=cn.periodic(2.0), z=cn.walls(-0.5, 0.5))
    m = cn.Model(box, name="Rayleigh-Benard 2D", Ra=2e4, Pr=1)
    m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        dt(u) + grad(p) - Pr*lap(u) = Pr*Ra*T*ez - u·grad(u)
        div(u) = 0
        dt(T) - lap(T) = -u·grad(T)
    """)
    m.at("bottom", u=0, T=+0.5)
    m.at("top", u=0, T=-0.5)
    sim = m.simulate(N=(32, 16), scheme="RK443", dt=1e-3)
    sim.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))

    def kinetic(sim):
        """Mean kinetic energy: one number per call, a time series in the Studio."""
        return 0.5 * float(np.mean(sim.grid("u") ** 2))

    w = sim.watch("rb_demo",
                  film={"T": cn.view("T", scale="signed")},          # the first view is the default
                  stats={"kinetic": kinetic,
                         "T(z)": cn.profile("T", along="z")},        # a declared curve
                  every_s=0.5,                                       # wall seconds between frames
                  spool=spool_dir)                                   # where the Studio reads
    print(w.explain().splitlines()[0])
    sim.run(steps=50)
    sim.unwatch()
    # [/watch]
    plain = m.simulate(N=(32, 16), scheme="RK443", dt=1e-3)
    plain.initial(T=lambda x, z: -z + 1e-3 * np.cos(np.pi * x) * np.cos(np.pi * z))
    plain.run(steps=50)
    assert np.array_equal(plain.solver.X, sim.solver.X)      # watching never changes the numbers
    assert w.tel.dropped_records == 0
    print("watched run identical to the unwatched one: True")


def io_hdf5() -> None:
    """io.md: analysis files and checkpoints in HDF5, a run cut in two jobs equal to one run, files
    read back with cn.io and with h5py alone."""
    work = TMP / "io"
    work.mkdir(parents=True, exist_ok=True)
    # [io]
    import h5py
    import numpy as np

    import conatus as cn

    def burgers():
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="viscous Burgers", nu=0.05)
        m.scalar("u")
        m.equations("dt(u) - nu*dx(dx(u)) = -u*dx(u)")
        return m.simulate(N=64, scheme="SBDF3", dt=2e-3)

    def job(steps):
        """The same script for every job of the chain."""
        sim = burgers()
        # the latest checkpoint of chk/ if there is one, else this initial state (t = 0)
        sim.resume_or_start(work / "chk", u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))
        sim.checkpoints(work / "chk", every=100, keep=2)  # one every 100 steps, two kept
        # no mode=: created on a fresh start, continued after a restart (said by explain())
        sim.output(work / "snapshots.h5", tasks=["u", "dx(u)"], every=50)
        sim.run(steps=steps)
        return sim

    sim = job(300)                                       # first job, stopped at its time limit
    sim = job(200)                                       # next job: from the checkpoint of step 300

    d = cn.io.read(work / "snapshots.h5")
    print(f"{len(d.sim_time)} writes, t = {d.sim_time[0]:.1f} ... {d.sim_time[-1]:.1f}, "
          f"u: {d.tasks['u'].shape}")
    with h5py.File(work / "snapshots.h5", "r") as f:     # without Conatus
        print("tasks:", sorted(f["tasks"]), " format:", f.attrs["format"])
    # [/io]
    one = burgers()
    one.initial(u=lambda x: np.sin(x) + 0.3 * np.cos(2 * x))
    one.run(steps=500)
    same = bool(np.array_equal(one.solver.X, sim.solver.X))
    print(f"two jobs equal to one uninterrupted run, bit for bit: {same}")
    assert same and len(d.sim_time) == 11       # steps 0, 50, ..., 500 (the initial state first)


def uncertainty_report() -> None:
    """uncertainty.md: value and error estimate of a computed number from declared N and dt
    studies (heat equation with an exact Bessel-series value, the case of tests/verify)."""
    # [uncertainty]
    import numpy as np

    import conatus as cn

    def u_at_origin(N, dt):
        """u(0, T) of the heat equation u_t = nu u_xx, u(x, 0) = exp(cos x)."""
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi)), name="heat", nu=0.5)
        m.scalar("u")
        m.equations("dt(u) - nu*dx(dx(u)) = 0")
        sim = m.simulate(N=N, scheme="RK222", dt=dt)
        sim.initial(u=lambda x: np.exp(np.cos(x)))
        sim.run(until=0.4)
        return float(sim.grid("u")[0])                   # x = 0 is the first grid point

    rep = cn.uncertainty(u_at_origin, N=[8, 12, 16], dt=[0.04, 0.02, 0.01], scheme_order=2,
                         name="u(0, T)")
    lo, hi = rep.interval
    print(f"u(0, T) = {rep.value:.8f} +- {rep.error:.1e}   trusted: {rep.trusted}")
    print(f"observed order in time: {rep.temporal.order:.2f} (scheme order 2)")
    # [/uncertainty]
    from scipy.special import iv
    k = np.arange(1, 40)
    exact = float(iv(0, 1) + 2 * np.sum(iv(k, 1) * np.exp(-0.5 * k ** 2 * 0.4)))
    print(f"exact value {exact:.8f}: inside the interval: {lo <= exact <= hi}")
    assert lo <= exact <= hi and rep.trusted


def initial_time() -> None:
    """io.md, *A run that starts at a time other than zero*: moving walls read on the
    clock of initial(t=), the start time in outputs, checkpoints and explain()."""
    # [initial-time]
    import tempfile
    from pathlib import Path

    import numpy as np

    import conatus as cn

    work = Path(tempfile.mkdtemp())                     # outputs and checkpoints of this example

    m = cn.Model(cn.Box(y=cn.walls(0, 5)), name="moving walls")
    m.scalar("u")
    m.equations("dt(u) - dy(dy(u))/2 = 0")             # exact: exp(-y) cos(t - y)
    m.at("bottom", u="cos(t)")
    m.at("top", "u = exp(-5)*cos(t - 5)")

    sim = m.simulate(N=32, scheme="RK443", dt=0.01)
    sim.initial(u=lambda y: np.exp(-y) * np.cos(0.7 - y), t=0.7)   # the state at t = 0.7
    sim.output(work / "run.h5", tasks=["u"], every=50)
    sim.checkpoints(work / "chk", every=100, keep=1)
    sim.run(until=1.7)
    y = sim.coordinate("y")
    err = np.abs(sim.grid("u") - np.exp(-y) * np.cos(sim.t - y)).max()
    print(f"t = {sim.t:.2f}, error against the exact solution {err:.1e}")
    print("output times:", cn.io.read(work / "run.h5").sim_time)
    print([ln for ln in sim.explain().splitlines() if "Initial state" in ln][0])
    # [/initial-time]
    assert abs(sim.t - 1.7) < 1e-12 and err < 1e-6
    assert np.allclose(cn.io.read(work / "run.h5").sim_time, [0.7, 1.2, 1.7])


def implicit_coriolis() -> None:
    """operators.md, *cross products in a box*: an implicit Coriolis term
    in a rotating periodic box, its explain() line and the warning when 2 Om dt > 1."""
    # [coriolis]
    import warnings

    import numpy as np

    import conatus as cn

    def rotating_box(Om, dt):
        box = cn.Box(x=cn.periodic(2 * np.pi), y=cn.periodic(2 * np.pi), z=cn.periodic(2 * np.pi))
        m = cn.Model(box, name="rotating box", Om=Om)
        m.vector("u")
        m.pressure("p")
        m.equations("dt(u) + 2*Om*cross(ez, u) + grad(p) = 0", "div(u) = 0")   # implicit
        return m.simulate(N=(8, 8, 8), scheme="RK443", dt=dt)

    # one inertial wave u = a(t) sin(x + z): it turns at the frequency 2 Om k_z/|k| = sqrt(2) Om
    sim = rotating_box(Om=1.0, dt=0.01)
    sim.initial(u=lambda x, y, z: np.array([1.0, 1.0, -1.0])[:, None, None, None]
                * np.sin(x + z)[None])
    sim.run(until=1.0)
    print([ln.strip() for ln in sim.explain().splitlines() if "implicit rotation" in ln][0])

    with warnings.catch_warnings(record=True) as caught:      # 2 Om dt = 5: stable, but damped
        warnings.simplefilter("always")
        rotating_box(Om=50.0, dt=0.05)
    print(type(caught[0].message).__name__ + ":", str(caught[0].message)[:72], "...")
    # [/coriolis]
    # exact amplitude: a' = -P(2 Om ez x a), P the projection perpendicular to k = (1, 0, 1)
    from scipy.linalg import expm
    k = np.array([1.0, 0.0, 1.0])
    P = np.eye(3) - np.outer(k, k) / 2
    Ez = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)   # ez x a
    a = expm(-2.0 * P @ Ez) @ np.array([1.0, 1.0, -1.0])
    g = [np.asarray(sim.coordinate(c)) for c in "xyz"]
    X, _, Z = np.meshgrid(*g, indexing="ij")
    got = np.array([2 * np.mean(sim.grid("u")[i] * np.sin(X + Z)) for i in range(3)])
    assert np.abs(got - a).max() < 1e-6, (got, a)
    assert isinstance(caught[0].message, cn.errors.UnderResolvedWarning)


CHECKS = {"sincos_free_slip": sincos_free_slip, "unbounded_compound": unbounded_compound,
          "schemes": schemes, "flowmap_nlop": flowmap_nlop, "shell_native": shell_native,
          "studio_watch": studio_watch, "io_hdf5": io_hdf5,
          "uncertainty_report": uncertainty_report, "initial_time": initial_time,
          "implicit_coriolis": implicit_coriolis}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sync = "--sync" in sys.argv
    names = args or list(CHECKS)
    bad = 0
    for name in names:
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf):
                CHECKS[name]()
            status = "PASS"
        except Exception:  # noqa: BLE001 -- every failure is reported, the run goes on
            status, bad = "FAIL", bad + 1
            traceback.print_exc()
        print(f"{status} {name} ({time.perf_counter() - t0:.1f} s)")
        print("    " + buf.getvalue().replace("\n", "\n    ").rstrip())
        if sync and status == "PASS":
            OUT.mkdir(exist_ok=True)
            (OUT / f"{name}.txt").write_text(buf.getvalue())
    print(f"{len(names) - bad} pass, {bad} fail")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
