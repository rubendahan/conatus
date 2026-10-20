"""Metamorphic relations of the dynamical-systems layer.

  Lyapunov       the exponents are averages of log R_ii over the QR products: the same with a
                 reorthonormalisation every interval or every other interval, with 2 or 4 blocks,
                 with a scaled diagonal inner product; a trajectory's exponents do not depend on
                 its place in the batch of starts;
  Newton-Krylov  a translated (reflected) seed gives the translated (reflected) equilibrium, the
                 same energy; a travelling wave's speed is unchanged by a translation of the seed
                 and changes sign under the reflection u(x) -> -u(-x);
  ensembles      a permutation of the members permutes the results; run(a) + run(b) = run(a + b);
  DMD / POD      rows (space) permuted with their weights, data scaled, snapshot order (POD),
                 time unit (DMD): the same spectrum, scaled as the unit;
  Floquet        a time shift of the periodic base flow leaves the exponents unchanged; Hill and
                 monodromy give the same leading exponent;
  growth         the optimal gain is the same one period later, and with the energy weights
                 multiplied by a constant;
  continuation   an equation multiplied by a constant has the same fold.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import conatus as cn
from conatus._tolerances_internal import INTERNAL

RTOL = INTERNAL["INVARIANCE_ROUNDING"].value
ETOL = INTERNAL["INVARIANCE_EIGENVALUE"].value
L22 = 22.0


def _ks(N=24, dt=0.05):
    m = cn.Model(cn.Box(x=cn.periodic(L22)), name="KS")
    m.scalar("u")
    m.equations("dt(u) + lap(u) + lap(lap(u)) = -u*dx(u)")
    return m.simulate(N=N, scheme="RK443", dt=dt)


def _seed(c):
    return lambda x: 0.3 * sum(c[j] * np.cos(2 * np.pi * (j + 1) * x / L22 + j) for j in range(3))


# ------------------------------------------------------------------------------- Lyapunov
LY = dict(transient=2.0, align=2.0, T=4.0, inner="euclidean", seed=0, zero_mean="u")


def _lyap(sim, starts, **kw):
    opts = {**LY, "reortho_every": 4, "n_blocks": 2, **kw}
    return sim.lyapunov(3, starts=starts, **opts)


def test_lyapunov_reorthonormalisation_and_blocks():
    sim = _ks()
    starts = [{"u": _seed([0.5, -1.0, 0.7])}]
    a = _lyap(sim, starts)
    b = _lyap(sim, starts, reortho_every=2)
    c = _lyap(sim, starts, n_blocks=4)
    assert np.allclose(a.exponents, b.exponents, rtol=1e-9, atol=1e-12), (a.exponents, b.exponents)
    assert np.array_equal(a.exponents, c.exponents) or np.allclose(
        a.exponents, c.exponents, rtol=RTOL, atol=RTOL), (a.exponents, c.exponents)


def test_lyapunov_inner_product_scale():
    sim = _ks()
    starts = [{"u": _seed([0.5, -1.0, 0.7])}]
    a = _lyap(sim, starts)
    n = sim.solver.system.n
    b = _lyap(sim, starts, inner=np.full(n, 7.0))
    assert np.allclose(a.exponents, b.exponents, rtol=1e-9, atol=1e-12), (a.exponents, b.exponents)


def test_lyapunov_trajectory_independent_of_batch_place():
    sim = _ks()
    s1, s2 = {"u": _seed([0.5, -1.0, 0.7])}, {"u": _seed([-0.2, 0.4, 1.1])}
    both = _lyap(sim, [s1, s2])
    swapped = _lyap(sim, [s2, s1])
    alone = _lyap(sim, [s2])
    assert np.allclose(both.per_trajectory[::-1], swapped.per_trajectory, rtol=RTOL, atol=RTOL), (
        both.per_trajectory, swapped.per_trajectory)
    assert np.allclose(both.per_trajectory[1], alone.per_trajectory[0], rtol=RTOL, atol=RTOL), (
        both.per_trajectory[1], alone.per_trajectory[0])


# ------------------------------------------------------------------------- Newton-Krylov
def _newton():
    from conatus.expert import NewtonKrylovHookstep
    return NewtonKrylovHookstep(tol=1e-12, max_iterations=40)


@pytest.mark.slow
def test_equilibrium_translation_and_reflection():
    from conatus.expert import find_equilibrium
    sim = _ks(N=32, dt=0.01)
    ks = sim.solver

    def state(f):
        sim.initial(u=f)
        return ks.X.copy()
    base = lambda x: np.sin(2 * np.pi * x / L22)                 # noqa: E731
    shift = L22 / 2                     # the searches work in the subspace -u(-x) = u(x): a shift
    #                                     by half the period keeps it, a general shift leaves it
    E = find_equilibrium(ks, state(base), _newton())
    Es = find_equilibrium(ks, state(lambda x: base(x - shift)), _newton())
    Er = find_equilibrium(ks, state(lambda x: -base(-x)), _newton())
    for other in (Es, Er):
        assert abs(other.energy - E.energy) < 1e-9 * E.energy, (E.energy, other.energy)


@pytest.mark.slow
def test_travelling_wave_symmetries():
    from conatus.expert import find_travelling_wave
    sim = _ks(N=32, dt=0.01)
    ks = sim.solver
    a = [0.34389095585778406, 0.5568341791702948, -0.9472943734057513, -0.529362057631228,
         -0.349148197714577, -0.9358415262182913, 1.3914943017041073, -0.3967285827537215]
    k = 2 * np.pi * np.arange(1, 5) / L22
    f = lambda x: sum(a[2 * i] * np.cos(k[i] * x) - a[2 * i + 1] * np.sin(k[i] * x)  # noqa: E731
                      for i in range(4))

    def state(g):
        sim.initial(u=g)
        return ks.X.copy()
    tw = find_travelling_wave(ks, state(f), 0.737, _newton())
    tws = find_travelling_wave(ks, state(lambda x: f(x - 2.0)), 0.737, _newton())
    twr = find_travelling_wave(ks, state(lambda x: -f(-x)), -0.737, _newton())
    c = tw.params["c"]
    assert abs(tws.params["c"] - c) < 1e-8 * abs(c), (c, tws.params["c"])
    assert abs(twr.params["c"] + c) < 1e-8 * abs(c), (c, twr.params["c"])


# ------------------------------------------------------------------------------ ensembles
def test_ensemble_permutation_and_segments(rng):
    from conatus.ensembles import Ensemble
    sim = _ks(N=32, dt=0.02)
    s = sim.solver
    X0 = 0.3 * rng.standard_normal((4, s.system.n))
    perm = [2, 0, 3, 1]
    for rhs in ("loop", "batched"):
        a = Ensemble(s, 4, rhs=rhs)
        a.set_states(X0)
        a.run(15)
        b = Ensemble(s, 4, rhs=rhs)
        b.set_states(X0[perm])
        b.run(6)
        b.run(9)
        assert np.array_equal(a.X[perm], b.X), (rhs, np.abs(a.X[perm] - b.X).max())


# ------------------------------------------------------------------------------- DMD / POD
def _snapshots(rng, n=40, m=24):
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    t = 0.1 * np.arange(m)
    # three travelling / growing waves: rank 2 + 2 + 1 = 5, five DMD eigenvalues
    X = (np.exp(-0.3 * t) * np.sin(x[:, None] - 2.0 * t)
         + 0.5 * np.exp(-0.1 * t) * np.cos(2 * x[:, None] + 3.0 * t)
         + 0.2 * np.outer(np.sin(3 * x), np.exp(0.05 * t)))
    return X, 0.1


def test_dmd_relations(rng):
    from conatus.datadriven import DataInnerProduct, dmd
    X, dt = _snapshots(rng)
    n = X.shape[0]
    w = 1.0 + rng.random(n)
    ip = DataInnerProduct.quadrature(w)
    ref = dmd(X, dt=dt, rank=5, inner_product=ip)
    perm = rng.permutation(n)
    alt = dmd(1e-7 * X[perm], dt=dt, rank=5, inner_product=DataInnerProduct.quadrature(w[perm]))
    slow = dmd(X, dt=60 * dt, rank=5, inner_product=ip)            # time in minutes, not seconds
    o = np.sort_complex(np.asarray(ref.omega))
    assert np.allclose(np.sort_complex(np.asarray(alt.omega)), o, rtol=ETOL, atol=ETOL * np.abs(o).max())
    assert np.allclose(np.sort_complex(np.asarray(slow.omega)) * 60, o, rtol=ETOL,
                       atol=ETOL * np.abs(o).max())


def test_pod_relations(rng):
    from conatus.datadriven import DataInnerProduct, pod
    X, _ = _snapshots(rng)
    n, m = X.shape
    w = 1.0 + rng.random(n)
    ref = pod(X, inner_product=DataInnerProduct.quadrature(w), rank=3, method="svd", mean="subtract")
    cols = rng.permutation(m)
    rows = rng.permutation(n)
    alt = pod(3.0 * X[rows][:, cols], inner_product=DataInnerProduct.quadrature(w[rows]), rank=3,
              method="snapshots", mean="subtract")
    assert np.allclose(alt.energies / 9.0, ref.energies, rtol=RTOL * 100, atol=0)
    assert abs(alt.energy_fraction - ref.energy_fraction) < RTOL * 100


# ---------------------------------------------------------------------------------- Floquet
def _periodic_lin(phase=0.0):
    m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi, complex=True), y=cn.walls(-1, 1)), nu=0.3)
    u = m.scalar("u")
    m.equations("dt(u) - nu*lap(u) = -u*dx(u)")
    m.at("walls", u=0)
    return m.linearize(about={u: lambda y, t: (1 + np.cos(2 * np.pi * (t + phase))) * (1 - y**2)},
                       period=1.0)


@pytest.mark.slow
def test_floquet_phase_and_paths():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h0 = _periodic_lin().floquet(12, k=1.0, method="hill", Nh=12, targets=[0.0], tol=1e-12,
                                     fields=("u",))
        h1 = _periodic_lin(0.37).floquet(12, k=1.0, method="hill", Nh=12, targets=[0.0],
                                         tol=1e-12, fields=("u",))
        mo = _periodic_lin().floquet(12, k=1.0, method="monodromy", steps=400, scheme="RK443",
                                     fields=("u",))
    lead = lambda r: max(np.asarray(r.mu), key=lambda z: z.real)  # noqa: E731
    assert abs(lead(h0) - lead(h1)) < ETOL * max(1.0, abs(lead(h0))), (lead(h0), lead(h1))
    assert abs(lead(h0).real - lead(mo).real) < 1e-6, (lead(h0), lead(mo))


@pytest.mark.slow
def test_optimal_gain_period_and_weight():
    lin = _periodic_lin()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g0 = lin.optimal_gain(12, k=1.0, t0=0.1, T=0.3, steps=60, scheme="RK443", fields=("u",)).G
        g1 = lin.optimal_gain(12, k=1.0, t0=1.1, T=0.3, steps=60, scheme="RK443", fields=("u",)).G
        g2 = lin.optimal_gain(12, k=1.0, t0=0.1, T=0.3, steps=60, scheme="RK443",
                              fields={"u": 5.0}).G
    assert abs(g1 / g0 - 1) < 1e-9 and abs(g2 / g0 - 1) < 1e-9, (g0, g1, g2)


# ----------------------------------------------------------------------------- continuation
def _fold(c, unit=1.0):
    from conatus.expert import ContinuationSettings, NewtonSolver, NLBVPContinuation
    m = cn.Model(cn.Box(x=cn.walls(-1, 1)), name="Bratu", lam=0.3 / unit)
    m.scalar("u")
    m.equations(f"bratu: {c!r}*dx(dx(u)) + {c!r}*{unit!r}*lam*exp(u) = 0")
    m.at("walls", u=0)
    pb = m.compile(48, dealias=1)
    newton = NewtonSolver(pb.problem, tol=1e-13, max_iterations=20, params={"lam": 0.3 / unit})
    sol = newton.solve({pb.fields["u"]: lambda x: 0 * x})
    st = ContinuationSettings(ds=0.2, ds_min=1e-8, ds_max=0.6, adapt=1.5, fast_iterations=3,
                              tolerance=1e-12, max_newton=15, max_points=6, dp_fd=1e-6)
    br = NLBVPContinuation(newton, "lam", st).run(sol.coeffs, 0.3 / unit, +1,
                                                    p_bounds=(0.3 / unit, 1.0 / unit))
    (fold,) = br.folds()
    return fold.p * unit


@pytest.mark.slow
@pytest.mark.parametrize("c", [1e-6, 1e6])
def test_fold_equation_scaling(c):
    """The fold of Bratu's branch with the equation multiplied by c. (A change of unit of the
    parameter is not a relation of pseudo-arclength continuation: ds measures (u, p) together,
    so the settings are in the units of both; not tested.)"""
    ref = _fold(1.0)
    assert abs(_fold(c) - ref) < 1e-9, (c, _fold(c), ref)
