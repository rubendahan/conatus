"""Numerical dye (passive scalar, declared diffusivity) against exact solutions."""
from __future__ import annotations

import numpy as np
import pytest

from conatus.diagnostics.dye import Dye
from conatus.diagnostics.grid import Chebyshev, Grid, Periodic
from conatus.errors import OptionError, UnsupportedError

TWO_PI = 2 * np.pi


def _translated_mode(g, U, kappa, T, nsteps):
    """c0 = cos(3x + 2y) in uniform flow U with diffusivity kappa; error at T vs exact."""
    X, Y = g.mesh()
    d = Dye(g, np.cos(3 * X + 2 * Y), kappa=kappa, dealias="2/3")
    u = [U[0] * np.ones(g.shape), U[1] * np.ones(g.shape)]
    d.step(T / nsteps, u, nsteps)
    ex = np.cos(3 * (X - U[0] * T) + 2 * (Y - U[1] * T)) * np.exp(-kappa * 13 * T)
    return np.max(np.abs(d.c - ex))


def test_translated_diffusing_mode_fourth_order():
    g = Grid([Periodic(TWO_PI, 32), Periodic(TWO_PI, 32)])
    errs = [_translated_mode(g, (1.0, 0.5), 0.05, 1.0, n) for n in (20, 40)]
    order = np.log2(errs[0] / errs[1])
    assert 3.8 < order < 4.3, (errs, order)
    assert _translated_mode(g, (1.0, 0.5), 0.05, 1.0, 400) < 1e-9   # measured 1.7e-10
    # pure diffusion is exact (integrating factor): only round-off
    assert _translated_mode(g, (0.0, 0.0), 0.1, 2.0, 3) < 1e-14


def test_mean_conserved_in_cellular_flow_and_variance_decays():
    g = Grid([Periodic(TWO_PI, 48), Periodic(TWO_PI, 48)])
    X, Y = g.mesh()
    c0 = np.exp(-4 * ((X - 2.0) ** 2 + (Y - 3.0) ** 2)) + 0.2
    d = Dye(g, c0, kappa=1e-2, dealias="2/3")
    m0, v0 = d.mean(), d.variance()
    u = [np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)]           # divergence-free
    d.step(0.02, u, 100)
    assert abs(d.mean() - m0) < 1e-13 * max(1.0, abs(m0))
    assert d.variance() < v0 and "IF-RK4" in d.explain()


def test_declarations_and_refusals():
    g = Grid([Periodic(TWO_PI, 8), Chebyshev(-1, 1, 8)])
    with pytest.raises(UnsupportedError):
        Dye(g, np.zeros(g.shape), kappa=0.1, dealias="2/3")
    gp = Grid([Periodic(TWO_PI, 8)] * 2)
    with pytest.raises(OptionError):
        Dye(gp, np.zeros(gp.shape), kappa=-1.0, dealias="2/3")
    with pytest.raises(OptionError):
        Dye(gp, np.zeros(gp.shape), kappa=0.0, dealias="yes")
    assert "pure advection" in Dye(gp, np.zeros(gp.shape), kappa=0.0, dealias=None).explain()


@pytest.mark.gpu
def test_dye_cpu_gpu_agree():
    cp = pytest.importorskip("cupy")
    g = Grid([Periodic(TWO_PI, 32), Periodic(TWO_PI, 24), Periodic(3.0, 16)])
    X, Y, Z = g.mesh()
    c0 = np.cos(X) * np.sin(2 * Y) + np.cos(2 * np.pi * Z / 3)
    u = [np.sin(Y), np.cos(Z * 2 * np.pi / 3), np.sin(X)]
    a = Dye(g, c0, kappa=0.01, dealias="2/3")
    a.step(0.01, u, 10)
    b = Dye(g.on(cp), cp.asarray(c0), kappa=0.01, dealias="2/3")
    b.step(0.01, [cp.asarray(v) for v in u], 10)
    assert np.max(np.abs(a.c - cp.asnumpy(b.c))) < 1e-12
