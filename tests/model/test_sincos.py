"""Sine/cosine wall axis of the native layer: ``cn.walls(a, b, basis="sincos")``.

The axis is the reflection subspace (about z = a) of RealFourier(2N) on the doubled period
[a, 2b - a); each unknown's parity is read off its wall conditions (model/sincos.py). Checks:

* bit identity with the hand-written IR problem (RealFourier(2N) + SymmetrySubspace): L, M of
  every group and F on a random subspace state, ``np.array_equal``;
* Rayleigh-Benard between stress-free walls: onset sign change and the exact growth rate of the
  mode n = 1 to round-off (one sine/cosine wavenumber per group), and the same model text with
  basis="chebyshev" + cn.free_slip (tests/model/test_free_slip.py);
* heat equation with T = 0 walls: decaying sine modes exact in space (only the time scheme's
  amplification factor differs from exp(-lambda t));
* nonlinear free-slip RB, sincos vs Chebyshev at equal resolution (slow);
* every refusal says what to write instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import conatus as cn
from conatus.errors import (BoundaryConditionError, EquationError, InputTypeError, OptionError)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_free_slip import K_C, RA_C, exact_growth, rb_free_slip  # noqa: E402


def _top(res) -> complex:
    return res.values[np.argmax(res.values.real)]


# ------------------------------------------------------------------------------------------------
# geometry, explain
# ------------------------------------------------------------------------------------------------

def test_walls_basis_declared():
    w = cn.walls(0, 1, basis="sincos")
    assert w.basis == "sincos" and cn.walls(0, 1).basis == "chebyshev"
    box = cn.Box(x=cn.periodic(2.0), z=w)
    assert box.sincos_coord == "z" and "basis='sincos'" in repr(box)
    _, bases, _ = box.build({"x": 8, "z": 12}, {"x": 1.5, "z": 1.5})
    bz = bases["z"]
    assert type(bz).__name__ == "RealFourier" and bz.size == 24 and bz.bounds == (0.0, 2.0)
    assert bz.dealias == 1.5                                       # on the doubled grid
    assert ("z: walls at 0, 1 -> sine/cosine bases (N_z wavenumbers n pi/L, n < N_z, L = 1), "
            "realised as the reflection subspace of RealFourier(2 N_z) on [0, 2); no tau"
            ) in box.explain()
    with pytest.raises(InputTypeError, match="sincos"):
        cn.walls(0, 1, basis="sine")


def test_explain_parities():
    m, *_ = rb_free_slip("sincos")
    text = m.explain()
    assert "u: cosine/sine by component (x: cosine, z: sine) (free-slip (u·n = 0, dz(u_t) = 0) at walls)" in text
    assert "T: sine series (T = 0 at walls)" in text
    assert "p: cosine series (deduced from the equivariance of momentum)" in text
    assert "no tau" in text and "tau_u" not in text
    comp = m.compile((8, 8))
    assert comp.taus == [] and [v.name for v in comp.problem.variables] == ["u", "p", "T", "tau_p"]
    assert {k: int(v[comp.symmetry.generators[0]]) for k, v in comp.symmetry.parities.items()} == \
        {"u": 1, "p": 1, "T": -1, "tau_p": 1}


def test_same_text_dz_condition_gives_cosine():
    """dz(T) = 0 (insulating walls) -> cosine; dz^2 of a sine -> still a sine."""
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(-1, 1, basis="sincos")), name="heat")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    m.at("walls", "dz(T) = 0")
    assert "T: cosine series (dz(T) = 0 at walls)" in m.explain()
    m2 = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="beam")
    m2.scalar("w")
    m2.equations("dt(w) + dz(dz(dz(dz(w)))) = 0")
    m2.at("walls", "w = 0", "dz(dz(w)) = 0")                      # hinged: sine
    assert "w: sine series (w = 0; dz(dz(w)) = 0 at walls)" in m2.explain()


# ------------------------------------------------------------------------------------------------
# bit identity with the hand-written IR problem
# ------------------------------------------------------------------------------------------------

def _hand_rb(Nx, Nz, Ra, Pr, k):
    """RealFourier(2 Nz) on [0, 2) with SymmetrySubspace(Reflection(z, about=0)), written by hand
    in the IR in the native declaration order (identity permutation)."""
    from conatus.expert import (EVEN, IVP, ODD, CartesianCoordinates, Domain, Field, Parameter,
                                RealFourier, Reflection, SymmetrySubspace, VectorField, discretize,
                                div, dot, dt, grad, integ, lap, unit_vector)
    cs = CartesianCoordinates("x", "z")
    x, z = cs.coords
    bx = RealFourier(x, Nx, (0, 2 * np.pi / k), dealias=1.5)
    bz = RealFourier(z, 2 * Nz, (0, 2.0), dealias=1.5)
    D = Domain(cs, [bx, bz])
    u, p, T = VectorField("u", D), Field("p", D), Field("T", D)
    tp = Field("tau_p", Domain(cs, []))
    Ra_, Pr_ = Parameter("Ra", Ra), Parameter("Pr", Pr)
    ez = unit_vector(z)
    G = SymmetrySubspace(Reflection(z, about=0.0), u=EVEN, p=EVEN, T=ODD, tau_p=EVEN)
    pb = IVP([u, p, T, tp], name="RB free-slip", symmetry=G)
    pb.add_equation((dt(u) + grad(p) - Pr_ * lap(u) - Pr_ * Ra_ * T * ez, -dot(u, grad(u))))
    pb.add_equation((div(u) + tp, 0))
    pb.add_equation((dt(T) - lap(T) - dot(u, ez), -dot(u, grad(T))))
    pb.add_equation((integ(p), 0))
    return discretize(pb)


def test_bit_identical_to_hand_written_subspace():
    from conatus.timestep import IVPSolver
    m, *_ = rb_free_slip("sincos", Ra=2000.0)
    pn = m.discretize((4, 4))
    ph = _hand_rb(4, 4, 2000.0, 1.0, K_C)
    assert len(pn.groups) == len(ph.groups) == 2 * 4
    nnz = 0
    for gn, gh in zip(pn.groups, ph.groups):
        a, b = pn.matrices(gn), ph.matrices(gh)                 # reduced (declared subspace)
        assert a.reduced and b.reduced and a.rows == b.rows and a.cols == b.cols
        for X, Y in ((a.L, b.L), (a.M, b.M)):
            assert np.array_equal(X.toarray(), Y.toarray())
            nnz += Y.nnz
    assert nnz > 100
    sn, sh = IVPSolver(pn, "RK443", 1e-3), IVPSolver(ph, "RK443", 1e-3)
    assert sn.system.reduced and sn.system.n == sh.system.n
    X = np.random.default_rng(1).standard_normal(sn.system.n)
    Fn, Fh = sn.F(X, 0.0), sh.F(X, 0.0)
    assert np.array_equal(Fn, Fh) and np.abs(Fh).max() > 1


# ------------------------------------------------------------------------------------------------
# Rayleigh-Benard onset between stress-free walls
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("Pr", [1.0, 0.7])
def test_rb_onset_exact_dispersion_relation(Pr):
    errs = []
    for Ra in (0.99 * RA_C, 1.01 * RA_C, 2000.0):
        m, u, p, T = rb_free_slip("sincos", Ra=Ra, Pr=Pr)
        lin = m.linearize(about={u: 0, T: 0})
        s = _top(lin.eigenvalues(8, k=K_C, n=1))
        errs.append(abs(s - exact_growth(K_C, Ra, Pr)))
        assert (s.real > 0) == (Ra > RA_C)
    assert max(errs) < 2e-13, errs                                 # round-off: one mode per group
    assert any("sine/cosine parity subspace" in n for n in lin.notes)


def test_rb_onset_all_groups_and_same_text_chebyshev():
    """All vertical wavenumbers gathered; the same model text with Chebyshev + free_slip gives
    the same leading growth rate (spectrally)."""
    m, u, p, T = rb_free_slip("sincos", Ra=2000.0)
    res = m.linearize(about={u: 0, T: 0}).eigenvalues(6, k=K_C)
    assert "groups n = 0..5" in res.log[0]
    s_sc = _top(res)
    mc, uc, pc, Tc = rb_free_slip("chebyshev", Ra=2000.0)
    s_ch = _top(mc.linearize(about={uc: 0, Tc: 0}).eigenvalues(16, k=K_C))
    ex = exact_growth(K_C, 2000.0, 1.0)
    assert abs(s_sc - ex) < 1e-13 and abs(s_ch - ex) < 1e-11
    # left eigenvectors cannot be gathered across groups: refused, not ignored
    with pytest.raises(OptionError, match="left=True"):
        m.linearize(about={u: 0, T: 0}).eigenvalues(6, k=K_C, left=True)
    # n = 2 is the second sine mode: exact as well
    s2 = _top(m.linearize(about={u: 0, T: 0}).eigenvalues(6, k=K_C, n=2))
    assert abs(s2 - exact_growth(K_C, 2000.0, 1.0, n=2)) < 1e-12


# ------------------------------------------------------------------------------------------------
# IVP: heat equation, exact in space
# ------------------------------------------------------------------------------------------------

def test_heat_sine_modes_exact_in_space():
    """dt(T) = lap(T), T = 0 at z = 0, 1: T = sin(pi z) cos(pi x) + 0.3 sin(3 pi z) decays as
    exp(-lambda t). In space the representation is exact: the state stays in the two initial
    modes to round-off; the remaining error is RK443's amplification factor (third order)."""
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="heat")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    m.at("walls", T=0)
    l1, l3 = 2 * np.pi ** 2, 9 * np.pi ** 2

    def exact(x, z, t):
        return (np.exp(-l1 * t) * np.sin(np.pi * z) * np.cos(np.pi * x)
                + 0.3 * np.exp(-l3 * t) * np.sin(3 * np.pi * z))
    errs = []
    for dt in (2e-3, 1e-3):
        sim = m.simulate(N=(8, 8), scheme="RK443", dt=dt)
        sim.initial(T=lambda x, z: exact(x, z, 0.0))
        sim.run(until=0.1)
        x, z = sim.coordinate("x")[:, None], sim.coordinate("z")[None, :]
        assert z.size == 9 and z[0, 0] == 0.0 and z[0, -1] == 1.0     # [a, b], both walls
        errs.append(np.abs(sim.grid("T") - exact(x, z, 0.1)).max())
        c = np.abs(sim.coeffs("T"))                                   # doubled-period layout
        c[2, 3] = c[0, 7] = 0.0          # (cos pi x, -sin pi z) and (const, -sin 3 pi z) kept
        assert c.max() < 1e-15, c.max()
    assert errs[1] < 1e-7 and 6 < errs[0] / errs[1] < 10, errs      # third order in time


def test_initial_data_extended_by_parity_and_checked():
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="heat")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    m.at("walls", T=0)
    sim = m.simulate(N=(4, 16), scheme="RK443", dt=1e-3)
    seen = []
    sim.initial(T=lambda x, z: seen.append(np.asarray(z).copy()) or z * (1 - z) + 0 * x)
    assert all(v.min() >= 0 and v.max() <= 1 for v in seen)       # evaluated on [a, b] only
    full = sim.solver.grid("T")[0]                                 # doubled period (expert)
    zf = np.arange(32) / 16
    assert np.allclose(full, np.where(zf <= 1, zf * (1 - zf), -(2 - zf) * (zf - 1)))
    with pytest.raises(InputTypeError, match="sine series .* at a wall") as e:
        sim.initial(T=lambda z: 1 - z)
    assert "sin(pi*(z-0)/1)" in str(e.value)
    with pytest.raises(OptionError, match="reduced=False"):
        m.simulate(N=(4, 8), scheme="RK443", dt=1e-3, reduced=False)


# ------------------------------------------------------------------------------------------------
# refusals
# ------------------------------------------------------------------------------------------------

def _heat():
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="heat")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 0")
    return m


def test_refuse_inhomogeneous():
    m = _heat()
    m.at("bottom", T=1)
    m.at("top", T=0)
    with pytest.raises(BoundaryConditionError, match="non-homogeneous") as e:
        m.compile((4, 8))
    assert "perturbation" in str(e.value) and "walls(0, 1)" in str(e.value)
    m = _heat()
    m.at("walls", "dz(T) = 1")
    with pytest.raises(BoundaryConditionError, match="non-homogeneous"):
        m.compile((4, 8))


def test_refuse_robin_and_coupled():
    m = _heat()
    m.at("walls", "dz(T) + T = 0")
    with pytest.raises(BoundaryConditionError, match="Robin") as e:
        m.compile((4, 8))
    assert "Chebyshev" in str(e.value)
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="two")
    m.scalar("T")
    m.scalar("S")
    m.equations("dt(T) - lap(T) = 0", "dt(S) - lap(S) = 0")
    m.at("walls", "T - S = 0", S=0)
    with pytest.raises(BoundaryConditionError, match="couples 2 unknowns"):
        m.compile((4, 8))


def test_refuse_conflicting_parities_and_counts():
    m = _heat()
    m.at("bottom", T=0)
    m.at("top", "dz(T) = 0")
    with pytest.raises(BoundaryConditionError, match="both a sine and a cosine"):
        m.compile((4, 8))
    m = _heat()
    m.at("bottom", T=0)
    with pytest.raises(BoundaryConditionError, match="1 condition.* at z=0 and 0 at z=1"):
        m.compile((4, 8))
    m = _heat()
    m.at("bottom", T=0, into=None)
    m.at("bottom", "T = 0")
    with pytest.raises(BoundaryConditionError, match="'walls'"):
        m.compile((4, 8))


def test_refuse_no_slip_velocity():
    m, *_ = rb_free_slip("sincos")
    m.conditions.clear()
    m.at("walls", u=0, T=0)
    with pytest.raises(BoundaryConditionError, match="not representable") as e:
        m.compile((4, 8))
    assert "cn.free_slip" in str(e.value) and "Chebyshev" in str(e.value)


def test_refuse_unconditioned_parity_none_or_several():
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="q")
    m.scalar("T")
    m.scalar("q")
    m.equations("dt(T) - lap(T) = 0", "dt(q) - dx(dx(q)) = 0")
    m.at("walls", T=0)
    with pytest.raises(BoundaryConditionError, match="several parities of q") as e:
        m.compile((4, 8))
    assert "q cosine" in str(e.value) and "q sine" in str(e.value)
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="q")
    m.scalar("T")
    m.scalar("q")
    m.equations("dt(T) - lap(T) = 0", "dt(q) - dx(dx(q)) = T + T*T")
    m.at("walls", T=0)
    with pytest.raises(BoundaryConditionError, match="no sine/cosine parity of q"):
        m.compile((4, 8))


def test_refuse_non_equivariant_source_and_given():
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="forced")
    m.scalar("T")
    m.equations("dt(T) - lap(T) = 1")
    m.at("walls", T=0)
    with pytest.raises(EquationError, match="not equivariant") as e:
        m.compile((4, 8))
    assert "sin(pi*(z-0)/L)" in str(e.value)
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="ncc")
    m.scalar("T")
    m.given(U=lambda z: 1 - z)
    with pytest.raises(InputTypeError, match="no parity") as e:      # at the declaring line
        m.equations("dt(T) - lap(T) + U*dx(T) = 0")
    assert "1 - z" in str(e.value) and "walls(0, 1)" in str(e.value)
    # an even coefficient is accepted and declared on the field
    m = cn.Model(cn.Box(x=cn.periodic(2.0), z=cn.walls(0, 1, basis="sincos")), name="ncc")
    m.scalar("T")
    m.given(U=lambda z: np.cos(np.pi * z))
    m.equations("dt(T) - lap(T) + U*dx(T) = 0")
    m.at("walls", T=0)
    comp = m.compile((4, 8))
    assert int(dict(comp.fields["U"].symmetry)[comp.symmetry.generators[0]]) == 1


def test_refuse_drift_on_sincos():
    m, u, p, T = rb_free_slip("sincos")
    with pytest.raises(OptionError, match="drift"):
        m.linearize(about={u: 0, T: 0}).eigenvalues(6, k=K_C, drift=8)


# ------------------------------------------------------------------------------------------------
# nonlinear: sincos vs Chebyshev + free_slip, same model text
# ------------------------------------------------------------------------------------------------

def _sincos_at(sim, name: str, zq: np.ndarray) -> np.ndarray:
    """Trigonometric interpolation of a sine/cosine field (doubled period) at the points zq."""
    g = sim.solver.grid(name)
    n2 = g.shape[-1]
    c = np.fft.rfft(g, axis=-1) / n2
    k = np.arange(c.shape[-1])
    w = np.where((k == 0) | (k == n2 // 2), 1, 2)
    a, b = 0.0, 1.0
    E = np.exp(1j * np.outer(k, 2 * np.pi * (zq - a) / (2 * (b - a))))
    return np.real((c * w) @ E)


@pytest.mark.slow
def test_nonlinear_rb_sincos_vs_chebyshev():
    """2D free-slip RB at Ra = 2000 (supercritical), t = 0.4: sincos and Chebyshev + free_slip
    (same model text, same N) agree to the spectral error of the Chebyshev run."""
    def ic(x, z):
        return 0.05 * np.sin(np.pi * z) * np.cos(K_C * x) + 0.02 * np.sin(2 * np.pi * z) * np.cos(2 * K_C * x)
    diffs = []
    for N in (16, 24):
        runs = {}
        for basis in ("sincos", "chebyshev"):
            m, *_ = rb_free_slip(basis, Ra=2000.0)
            sim = m.simulate(N=(N, N), scheme="RK443", dt=2e-3)
            sim.initial(T=ic)
            sim.run(until=0.4)
            runs[basis] = sim
        zc = runs["chebyshev"].coordinate("z")
        d = np.abs(runs["chebyshev"].grid("T") - _sincos_at(runs["sincos"], "T", zc)).max()
        diffs.append(d)
        assert np.abs(runs["chebyshev"].grid("T")).max() > 0.3          # nonlinear amplitude
    print("max |T_cheb - T_sincos| at N = 16, 24:", diffs)
    assert diffs[0] < 1e-4 and diffs[1] < 1e-7 and diffs[1] < diffs[0] / 100


# ------------------------------------------------------------------------------------------------
# GPU: the reflection-reduced IVPSolver on CUDA (evaluator F and parity-reduced fused F)
# ------------------------------------------------------------------------------------------------

def _gpu() -> bool:
    from conatus.runtime import cuda
    return cuda.available()


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu(), reason="no CUDA device")
@pytest.mark.parametrize("f_backend", ["evaluator", "fused"])
def test_sincos_rb_gpu_matches_cpu(f_backend):
    """Free-slip RB on the sine/cosine axis, backend='cuda' vs 'cpu', 20 RK443 steps."""
    def ic(x, z):
        return 0.05 * np.sin(np.pi * z) * np.cos(K_C * x) + 0.02 * np.sin(2 * np.pi * z)
    out = {}
    for backend in ("cpu", "cuda"):
        m, *_ = rb_free_slip("sincos", Ra=2000.0)
        sim = m.simulate(N=(16, 16), scheme="RK443", dt=2e-3, backend=backend, f_backend=f_backend)
        sim.initial(T=ic)
        sim.run(steps=20)
        out[backend] = (sim.grid("T"), sim.explain())
    d = np.abs(out["cuda"][0] - out["cpu"][0]).max()
    print(f"f_backend={f_backend}: max |T_gpu - T_cpu| = {d:.2e}")
    assert d < 1e-12 * np.abs(out["cpu"][0]).max()
    assert "symmetry subspace" in out["cuda"][1]
    if f_backend == "fused":
        assert "parity reduction not used" not in out["cuda"][1], out["cuda"][1]
