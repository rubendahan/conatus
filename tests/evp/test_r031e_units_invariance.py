"""Eigenvalues do not depend on the units of the problem (hunt of 0.3.0 and L2).

Oracles (closed forms, independent of Conatus):
- s T = kappa T'' on [0, H], T = 0 at both walls: s_n = -kappa (n pi / H)^2, n = 1, 2, ...
- s u = u'' on [0, pi], u' = 0 at both walls: s_n = -n^2, n = 0, 1, 2, ...
- s u = c lap(u) on the unit disk, u = 0 on the wall, m = 0: s = -c j_{0,1}^2.

Before the fix, dense QZ ran on the eigenvalue in its own unit: kappa = 1.4e-7, H = 4e3 lost three
digits, kappa = 1e-6, H = 2.9e6 was off by a factor 6.6, and s u = c u'' with c = 1e16 returned no
finite mode; shift-invert with the target on an eigenvalue raised a bare RuntimeError or dropped
the other requested modes as infinite.
"""
import warnings

import numpy as np
import pytest

import conatus as cn
from conatus.errors import CorrectnessWarning
from conatus.solvers.evp import balance_pencil, eigen_scale, solve_dense

J01_SQ = 5.783185962946784          # j_{0,1}^2, first zero of squared


def diffusion(kappa, H, *, bc="T", N=32, factor=None):
    m = cn.Model(cn.Box(z=cn.walls(0, H)), name="units", kappa=kappa)
    m.eigenvalue("s", growth="real")
    m.scalar("T")
    eq = "s*T - kappa*dz(dz(T)) = 0" if factor is None else f"{factor}*s*T - {factor}*kappa*dz(dz(T)) = 0"
    m.equations(eq)
    m.at("walls", f"{bc} = 0")
    return m


def top(r, k):
    return np.sort(np.real(r.values))[::-1][:k]


CASES = [(1.0, 1.0), (1.4e-7, 4e3), (1e3, 2e8), (1e-6, 2.9e6), (1e-9, 1e-5), (31.5576, 2.9e6),
         (1.0, np.pi * 1e-8)]


@pytest.mark.parametrize("kappa,H", CASES)
def test_dimensional_diffusion_matches_closed_form(kappa, H):
    """SI-unit problems (ocean, solar, mantle, microfluidic) give the closed form to 1e-11 and
    keep as many finite modes as the nondimensional problem."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # no resolution warning either
        r = diffusion(kappa, H).eigenvalues(N=32)
    ref = diffusion(1.0, 1.0).eigenvalues(N=32)
    oracle = -kappa * (np.arange(1, 6) * np.pi / H) ** 2
    assert np.allclose(top(r, 5), oracle, rtol=1e-11, atol=0), top(r, 5) / oracle
    assert len(r) == len(ref) and r.n_infinite == ref.n_infinite


@pytest.mark.parametrize("c", [1e-16, 1e-12, 1e-8, 1e-4, 1e4, 1e8, 1e12, 1e16])
def test_eigenvalue_rescaling_scales_the_whole_finite_spectrum(c):
    """s u = c u'' on [0, pi]: the finite spectrum is c times that of c = 1, mode for mode (to
    rounding on the resolved modes), with the same finite and infinite counts."""
    r = diffusion(c, np.pi).eigenvalues(N=24)
    ref = diffusion(1.0, np.pi).eigenvalues(N=24)
    assert len(r) == len(ref) and r.n_infinite == ref.n_infinite
    assert np.allclose(top(r, 5) / c, -np.arange(1, 6) ** 2, rtol=1e-11, atol=0)
    w, w0 = np.sort(r.values.real / c), np.sort(ref.values.real)
    assert np.allclose(w, w0, rtol=1e-8, atol=0)
    assert r.infinite_ambiguous.size == ref.infinite_ambiguous.size


@pytest.mark.parametrize("factor", ["1e-12", "1e12"])
@pytest.mark.parametrize("kappa,H", [(1.4e-7, 4e3), (1e3, 2e8)])
def test_equation_multiplied_by_a_constant_changes_nothing(factor, kappa, H):
    r = diffusion(kappa, H, factor=factor).eigenvalues(N=32)
    oracle = -kappa * (np.arange(1, 6) * np.pi / H) ** 2
    assert np.allclose(top(r, 5), oracle, rtol=1e-11, atol=0)


def test_complex_spectrum_in_any_unit():
    """s u = (1 + i) c u'' (a complex pencil): s_n = -(1 + i) c n^2 at c = 1e-12 and 1e12."""
    for c in (1e-12, 1.0, 1e12):
        m = cn.Model(cn.Box(z=cn.walls(0, np.pi)), name="cx", c=c)
        m.eigenvalue("s", growth="real")
        m.scalar("u")
        m.equations("s*u - (1 + 1j)*c*dz(dz(u)) = 0")
        m.at("walls", u=0)
        r = m.eigenvalues(N=24)
        w = r.values[np.argsort(np.abs(r.values))][:4]
        assert np.allclose(w, -(1 + 1j) * c * np.arange(1, 5) ** 2, rtol=1e-11, atol=0), w / c


@pytest.mark.parametrize("c", [1e-12, 1.0, 1e14])
def test_disk_keeps_its_modes_in_any_unit(c):
    d = cn.Model(cn.Disk(1.0), name="dk", c=c)
    d.eigenvalue("s", growth="real")
    d.scalar("u")
    d.equations("s*u - c*lap(u) = 0")
    d.at("wall", u=0)
    r = d.eigenvalues(N=(8, 24), m=0)
    ref_count = 23                                 # c = 1 at this resolution (0.3.0)
    assert len(r) == ref_count, (len(r), r.n_infinite)
    assert np.isclose(top(r, 1)[0], -c * J01_SQ, rtol=1e-11, atol=0)


@pytest.mark.parametrize("c", [1e-14, 1.0, 1e14])
def test_parametric_path_and_least_stable_are_unit_invariant(c):
    """ParametricEVP.solve (used by critical and least_stable) and the Newton tracking."""
    m = diffusion(c, np.pi)
    P = m.parametric(N=24, over=("kappa",))
    r = P.solve(kappa=c)
    assert np.allclose(top(r, 3) / c, [-1.0, -4.0, -9.0], rtol=1e-11, atol=0)
    mode = P.least_stable(kappa=c)
    assert np.isclose(complex(mode.sigma).real / c, -1.0, rtol=1e-11, atol=0)


@pytest.mark.parametrize("c", [1e-10, 1.0, 1e10])
def test_shift_invert_is_unit_invariant(c):
    r = diffusion(c, np.pi).eigenvalues(N=24, method="shift-invert", target=-2.2 * c, n_modes=3)
    assert len(r) == 3
    assert np.allclose(np.sort(r.values.real / c)[::-1], [-1.0, -4.0, -9.0], rtol=1e-10, atol=0)


def neumann(target, n_modes=3, k=None):
    if k is None:
        m = cn.Model(cn.Box(z=cn.walls(0, np.pi)), name="neu")
    else:
        m = cn.Model(cn.Box(x=cn.periodic(2 * np.pi), z=cn.walls(0, np.pi)), name="neu")
    m.eigenvalue("s", growth="real")
    m.scalar("u")
    m.equations("s*u - lap(u) = 0" if k is not None else "s*u - dz(dz(u)) = 0")
    m.at("walls", "dz(u) = 0")
    kw = {} if k is None else {"k": k}
    return m.eigenvalues(N=24 if k is None else (8, 24), method="shift-invert", target=target,
                         n_modes=n_modes, **kw)


@pytest.mark.parametrize("target,expected", [(0.0, [0.0, -1.0, -4.0]), (-1.0, [-1.0, 0.0, -4.0]),
                                             (-4.0, [-4.0, -1.0, 0.0]), (0.1, [0.0, -1.0, -4.0])])
def test_shift_invert_target_on_an_eigenvalue_returns_the_requested_modes(target, expected):
    """Target exactly on 0 (neutral mode), -1 or -4: the three nearest eigenvalues, no error, and
    the shift move is in the log."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", CorrectnessWarning)
        r = neumann(target)
    assert len(r) == 3, r.log
    assert np.allclose(np.sort(r.values.real), np.sort(expected), atol=1e-9, rtol=1e-10)
    if target in (0.0, -1.0, -4.0):
        assert any("Arnoldi ran at the shift" in str(x) for x in r.log)


def test_shift_invert_periodic_mean_mode_target_zero():
    r = neumann(0.0, k=0.0)
    assert len(r) == 3
    assert np.allclose(np.sort(r.values.real), [-4.0, -1.0, 0.0], atol=1e-10)


def test_shift_invert_direct_sum_target_on_double_eigenvalue():
    """s u = u'', s v = v''/4 on [0, pi]: -1 is double; target -1 and -0.25 return 4 modes."""
    m = cn.Model(cn.Box(z=cn.walls(0, np.pi)), name="ds")
    m.eigenvalue("s", growth="real")
    m.scalar("u")
    m.scalar("v")
    m.equations("s*u - dz(dz(u)) = 0", "s*v - 0.25*dz(dz(v)) = 0")
    m.at("walls", u=0, v=0)
    for t in (-1.0, -0.25, -4.0):
        r = m.eigenvalues(N=32, method="shift-invert", target=t, n_modes=4)
        oracle = np.array([-0.25, -1.0, -1.0, -2.25, -4.0, -4.0, -6.25, -9.0, -9.0])
        near = np.sort(oracle[np.argsort(np.abs(oracle - t), kind="stable")][:4])
        assert len(r) == 4, r.log
        assert np.allclose(np.sort(r.values.real), near, atol=1e-10)


def test_eigen_scale_is_covariant():
    """eigen_scale scales with the eigenvalue and ignores the scale of an equation; the eigenvalues
    of the rescaled pencil are c times the others (random matrices with a boundary row and a tau
    column)."""
    rng = np.random.default_rng(3)
    n = 7
    L = rng.standard_normal((n, n))
    M = rng.standard_normal((n, n))
    M[-1] = 0.0                     # boundary row
    M[:, -1] = 0.0                  # tau column
    L[-1, -1] = 0.0                 # the tau unknown enters the physical rows only
    mu = eigen_scale(L, M)
    for c in (1e-13, 1e9):
        Lc = L.copy()
        Lc[:-1, :-1] *= c           # eigenvalue unit change on the physical block
        assert np.isclose(eigen_scale(Lc, M), c * mu, rtol=1e-14)
        mu2 = balance_pencil(Lc, M)[2]                  # the unit applied beyond BALANCE_FREE
        assert 1e-4 * (1 - 1e-12) <= c * mu / mu2 <= 1e4 * (1 + 1e-12)
        w0 = solve_dense(L, M).values
        w1 = solve_dense(Lc, M).values / c
        assert w0.size == w1.size
        gap = np.abs(w1[:, None] - w0[None, :]).min(axis=1)       # conjugate pairs may swap order
        assert np.all(gap <= 1e-10 * np.abs(w1)), gap
    D = np.diag(10.0 ** rng.integers(-12, 12, n))
    assert np.isclose(eigen_scale(D @ L, D @ M), mu, rtol=1e-14)


def test_noise_row_of_m_keeps_the_d373_answer():
    """L = -I, M = diag(1, 1e-17) and its rescalings by c: root 1 kept (times c), the other
    reported in infinite_ambiguous (unchanged by the eigenvalue unit)."""
    for c in (1e-10, 1.0, 1e10):
        r = solve_dense(-c * np.eye(2), np.diag([1.0, 1e-17]))
        assert r.values.size == 1 and np.isclose(r.values[0].real, c, rtol=1e-14)
        assert r.infinite_ambiguous.size == 1
