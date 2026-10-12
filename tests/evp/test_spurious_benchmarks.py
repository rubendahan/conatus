"""Spectral-tail spurious flag (X1) on the published eigenvalue benchmarks.

Contract checked here: on every benchmark of the paper and tutorials that goes through the
flag (Cartesian plans with a wall-normal Chebyshev, Laguerre or compound axis), (1) no mode that
is resolved (found again at a finer resolution to 1e-8 with a backward residual below 1e-10) is
flagged, so a physical mode is never ranked last, and (2) the least stable mode of the default
ranking is the published one. The critical points without a drift pencil (spectral-tail test
only) are the published ones and their log does not report a set-aside mode of larger growth.
Polar and spherical plans are outside the test: their result says so (spurious_note)."""
import numpy as np
import pytest

import conatus as cn
from conatus.expert import critical
from conatus.problems import native as NS

pytestmark = pytest.mark.slow
RESOLVED_DRIFT = 1e-8
RESOLVED_RESIDUAL = 1e-10


def _check_resolved_not_flagged(res, fine):
    """(1) and the least stable mode of the ranking is resolved; returns that mode's value."""
    assert res.spurious is not None, res.spurious_note
    d = np.array([np.min(np.abs(fine.values - s)) / max(1.0, abs(s)) for s in res.values])
    resolved = (d <= RESOLVED_DRIFT) & (res.residuals <= RESOLVED_RESIDUAL)
    assert resolved.sum() >= 3, int(resolved.sum())
    wrong = np.nonzero(resolved & res.spurious)[0]
    assert wrong.size == 0, [(res.values[i], res.tails[i]) for i in wrong]
    i = int(res.order()[0])
    assert resolved[i] and not res.spurious[i], (res.values[i], res.tails[i])
    return res.values[i]


def _os(Re, alpha, flow):
    # no warning to silence (review of the former DeprecationWarning filter here
    # hid nothing, and would have hidden a correctness warning of the old category)
    return NS.orr_sommerfeld(Re=Re, alpha=alpha, flow=flow)


@pytest.mark.parametrize("Re,N,N2", [(1e4, 64, 96), (1e6, 200, 300)])
def test_orr_sommerfeld_poiseuille(Re, N, N2):
    """Orszag (1971) at Re = 1e4, and Re = 1e6 (critical and wall layers of width ~ 0.01)."""
    m = _os(Re, 1.0, "poiseuille")
    c = _check_resolved_not_flagged(m.eigenvalues(N=N), m.eigenvalues(N=N2))
    if Re == 1e4:
        assert abs(c - (0.23752649 + 0.00373967j)) < 1e-7


def test_orr_sommerfeld_couette():
    """Plane Couette (stable, wall modes in pairs): Re = 1e4, alpha = 1."""
    m = _os(1e4, 1.0, "couette")
    c = _check_resolved_not_flagged(m.eigenvalues(N=96), m.eigenvalues(N=144))
    assert c.imag < 0


def test_rayleigh_benard_linearization():
    """Reid & Harris (1958): s = 5.5425059 at Ra = 2500, k = pi; neutral at (1707.762, 3.117)."""
    lin = NS.rayleigh_benard_evp(k=np.pi, Ra=2500.0, Pr=1.0)
    s = _check_resolved_not_flagged(lin.eigenvalues(N=24, k=np.pi), lin.eigenvalues(N=36, k=np.pi))
    assert abs(s - 5.5425059) < 1e-6
    lin = NS.rayleigh_benard_evp(k=3.117, Ra=1707.762, Pr=1.0)
    s = _check_resolved_not_flagged(lin.eigenvalues(N=24, k=3.117), lin.eigenvalues(N=36, k=3.117))
    assert abs(s.real) < 1e-4


def test_channel_primitive_variables():
    """Orszag's mode in primitive variables (velocity, pressure, taus): Re s = alpha Im c and
    |Im s| = alpha Re c (the sign of Im s is that of the Fourier convention)."""
    m = NS.channel(Lx=2 * np.pi, Lz=2 * np.pi, Re=1e4, pressure_gradient=2e-4)
    lin = m.linearize(about={"u": lambda x, y, z: np.stack([1 - y * y, 0 * y, 0 * y])})
    k = {"x": 1.0, "z": 0.0}
    s = _check_resolved_not_flagged(lin.eigenvalues(N=64, k=k), lin.eigenvalues(N=96, k=k))
    assert abs(s.real - 0.00373967) < 1e-8 and abs(abs(s.imag) - 0.23752649) < 1e-8


def test_half_line_laguerre():
    """s u = u'' - y^2 u on y > 0, u(0) = 0: the odd states of the oscillator, s = -(4n + 3)."""
    m = cn.Model(cn.Box(y=cn.half_line(scale=0.25)), name="half oscillator")
    m.scalar("u")
    m.eigenvalue("s", growth="real")
    m.given(V=lambda y: y ** 2)
    m.equations("s*u - lap(u) + V*u = 0")
    m.at("wall", u=0)
    s = _check_resolved_not_flagged(m.eigenvalues(N=48), m.eigenvalues(N=72))
    assert abs(s + 3) < 1e-10


def test_compound_interval():
    """s u = u'' on [0, 1] u [1, 2], u = 0 at both ends: s = -(n pi / 2)^2, per-piece tails."""
    m = cn.Model(cn.Box(z=cn.walls(0, 1, 2)), name="compound")
    m.scalar("u")
    m.eigenvalue("s", growth="real")
    m.equations("s*u - lap(u) = 0")
    m.at("walls", u=0)
    s = _check_resolved_not_flagged(m.eigenvalues(N=(24, 24)), m.eigenvalues(N=(36, 36)))
    assert abs(s + (np.pi / 2) ** 2) < 1e-10


def _set_aside_tails(text):
    """Tails of the modes the log reports as set aside by the tail test (largest growth each)."""
    return [float(t.split(")")[0]) for t in text.split("growth than every mode it kept")[1:]
            for t in [t.split("tail ")[1]]]


def test_critical_points_with_the_tail_test_only():
    """Neutral curves without a drift pencil: the published critical points; a mode set aside
    on the way is a tau mode (tail near 1: OS has growing tau modes, Im c ~ 1e3), said in the log."""
    P = _os(1e4, 1.0, "poiseuille").parametric(N=64, over=("Re", "alpha"))
    crit = critical(P, "Re", bracket=(5000, 7000), over={"alpha": (0.98, 1.06)}, xtol=1e-9)
    assert abs(crit.value - 5772.221816) < 1e-4 and abs(crit.over["alpha"] - 1.020547) < 1e-5
    text = "\n".join(crit.log)
    assert "spectral-tail test only" in text
    assert all(t > 0.5 for t in _set_aside_tails(text)), text
    lin = NS.rayleigh_benard_evp(k=3.117, Ra=1700.0, Pr=1.0)
    P = lin.parametric(N=24, over=("Ra", "k"))
    crit = critical(P, "Ra", bracket=(1500, 2000), over={"k": (2.9, 3.3)}, xtol=1e-9)
    assert abs(crit.value - 1707.762) < 1e-3 and abs(crit.over["k"] - 3.117) < 1e-3
    assert all(t > 0.5 for t in _set_aside_tails("\n".join(crit.log)))


def test_set_aside_mode_is_reported():
    """OS at N = 16: the tail test sets aside modes of larger Im c than the one it selects; the
    selection says so (no silent jump to another branch)."""
    P = _os(1e4, 1.0, "poiseuille").parametric(N=16, over=("Re",))
    mode = P.least_stable(Re=1e4)
    assert "set aside" in mode.note and "spurious='none'" in mode.note, mode.note
    assert _set_aside_tails(mode.note)
