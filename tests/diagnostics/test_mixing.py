"""Mixing diagnostics against closed forms ( §Mixing)."""
from __future__ import annotations

import numpy as np
import pytest

from conatus.diagnostics import Chebyshev, Grid, Periodic
from conatus.diagnostics import mixing as M
from conatus.errors import OptionError, UnsupportedError

TWO_PI = 2 * np.pi


def test_variance_and_mix_norms_single_mode():
    """c = 2 + cos 3x: variance 1/2; H^{-1/2}: sqrt(1/(2*3)); H^{-1}: sqrt(1/(2*9));
    MMP weight: sqrt((1 + 9)^{-1/2} / 2). The mean never enters."""
    g = Grid([Periodic(TWO_PI, 32), Periodic(TWO_PI, 16)])
    X, Y = g.mesh()
    c = 2.0 + np.cos(3 * X)
    assert abs(float(M.variance(g, c)) - 0.5) < 1e-15
    assert abs(float(M.mix_norm(g, c)) - np.sqrt(1 / 6)) < 1e-15
    assert abs(float(M.mix_norm(g, c, s=1.0)) - np.sqrt(1 / 18)) < 1e-15
    assert abs(float(M.mix_norm(g, c, weight="mmp")) - np.sqrt(0.5 / np.sqrt(10))) < 1e-15


def test_mix_norm_decays_exactly_under_diffusion():
    """Two modes diffusing: ||c||_{H^{-1/2}}^2(t) = sum a_k^2/2 e^{-2 kappa k^2 t} / |k|."""
    g = Grid([Periodic(TWO_PI, 32), Periodic(TWO_PI, 32)])
    X, Y = g.mesh()
    kap, t = 0.05, 1.3
    c = np.exp(-kap * 4 * t) * np.cos(2 * X) + 0.5 * np.exp(-kap * 25 * t) * np.sin(3 * X + 4 * Y)
    ref = np.sqrt(0.5 * np.exp(-8 * kap * t) / 2 + 0.125 * np.exp(-50 * kap * t) / 5)
    assert abs(float(M.mix_norm(g, c)) - ref) < 1e-15


def test_ape_sorted_is_zero_and_inverted_converges():
    """Stable b = z: APE = 0 to round-off (same cell geometry). Inverted b = -z on [0, 1]:
    APE -> A/6 with the declared O(h^2) cell-centre error (observed order 2)."""
    errs = []
    for n in (16, 32, 64):
        g = Grid([Periodic(TWO_PI, 8), Chebyshev(0.0, 1.0, n)])
        X, Z = g.mesh()
        assert abs(M.potential_energies(g, Z, vertical_axis=1)["APE"]) < 1e-14
        e = M.potential_energies(g, -Z, vertical_axis=1)
        assert e["APE"] >= 0
        errs.append(abs(e["APE"] / TWO_PI - 1 / 6))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.95), orders
    assert errs[-1] < 4e-5


def test_ape_nonnegative_random():
    """APE >= 0 exactly for any field (rearrangement inequality with the same geometry)."""
    rng = np.random.default_rng(4)
    g = Grid([Periodic(1.0, 6), Periodic(2.0, 5), Chebyshev(-1.0, 1.0, 9)])
    for _ in range(20):
        e = M.potential_energies(g, rng.standard_normal(g.shape), vertical_axis=2)
        assert e["APE"] >= -1e-14


def test_mixing_rate_diffusing_profile():
    """b = z + a e^{-kappa pi^2 t} cos(pi z) on [0, 1], fixed flux dz b = 1 at both walls (exact
    diffusion solution, monotone for a pi < 1). Winters et al.: Phi_d = kappa (b(1) - b(0)) =
    kappa (1 - 2 a e^{-kappa pi^2 t}) and Phi_i = -kappa [z dz b]_0^1 = -kappa."""
    kap, a = 0.1, 0.2
    g = Grid([Periodic(TWO_PI, 4), Chebyshev(0.0, 1.0, 64)])
    X, Z = g.mesh()
    t = np.linspace(0.0, 1.0, 41)
    Eb = [M.potential_energies(g, Z + a * np.exp(-kap * np.pi**2 * s) * np.cos(np.pi * Z),
                               vertical_axis=1)["E_b"] / TWO_PI for s in t]
    pd = M.mixing_rate(t, Eb, phi_i=-kap)
    ref = kap * (1 - 2 * a * np.exp(-kap * np.pi**2 * t))
    assert np.abs(pd - ref).max() < 3e-5          # O(dt^2) + O(h^2), declared
    eff = M.mixing_efficiency(pd, 3 * pd, t=t)
    assert np.allclose(eff["eta"], 0.25) and abs(eff["eta_cumulative"] - 0.25) < 1e-14


def test_refusals():
    """Mix-norm only on a torus; s > 0; declared weights only."""
    g = Grid([Periodic(TWO_PI, 8), Chebyshev(0.0, 1.0, 8)])
    with pytest.raises(UnsupportedError):
        M.mix_norm(g, np.zeros(g.shape))
    gp = Grid([Periodic(TWO_PI, 8)] * 2)
    with pytest.raises(OptionError):
        M.mix_norm(gp, np.zeros(gp.shape), s=0.0)
    with pytest.raises(OptionError):
        M.mix_norm(gp, np.zeros(gp.shape), weight="sobolev")
