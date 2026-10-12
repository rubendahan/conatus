"""Rayleigh-Benard onset for the three classical wall pairs, at the published digits.

Native model (thermal-diffusion units, depth 1, Pr = 1; the onset is stationary so Pr drops
out), perturbation about conduction with T = 0 on both walls:

* rigid-rigid: Ra_c = 1707.76177710 at k_c = 3.11632 (Dominguez-Lerma, Ahlers & Cannell 1984,
  Phys. Fluids 27, 856; Reid & Harris 1958: 1707.762, 3.117);
* rigid-free: Ra_c = 1100.65 at k_c = 2.682 (Chandrasekhar 1961, ch. II, Table III);
* free-free: Ra_c = 27 pi^4 / 4 = 657.511364 at k_c = pi / sqrt(2) (Rayleigh 1916).

The rigid-free pair was not tested before (measured: 1100.649606 at 2.68233); rigid-rigid
1707.761777 at 3.11633, free-free 657.511364 at 2.22145. Ra_neutral(k) is found by root bracketing of the leading
growth rate at three wave numbers around k_c, and the parabola through them gives (k_c, Ra_c): the
cubic term of Ra(k) shifts the vertex by ~1e-6 in k with dk = 0.005.
"""
import numpy as np
import pytest
import scipy.optimize as so

import conatus as cn


def _model(bottom, top, k):
    box = cn.Box(x=cn.periodic(2 * np.pi / k), z=cn.walls(0, 1))
    m = cn.Model(box, name="RB onset", Ra=1000.0, Pr=1.0)
    u, _, T = m.velocity("u"), m.pressure("p"), m.scalar("T")
    m.equations("""
        momentum:   dt(u) + grad(p) - Pr*lap(u) - Pr*Ra*T*ez = -u·grad(u)
        continuity: div(u) = 0
        heat:       dt(T) - lap(T) - u·ez = -u·grad(T)
    """)
    m.at("bottom", u=bottom, T=0)
    m.at("top", u=top, T=0)
    return m, u, T


def _neutral(bottom, top, k, lo, hi, N=24):
    m, u, T = _model(bottom, top, k)

    def growth(Ra):
        m.Ra = Ra
        res = m.linearize(about={u: 0, T: 0}).eigenvalues(N, k=k)
        i = int(np.argmax(res.values.real))
        assert res.residuals[i] < 1e-8
        return float(res.values[i].real)
    return so.brentq(growth, lo, hi, xtol=1e-9, rtol=1e-14)


def _critical(bottom, top, k0, lo, hi):
    ks = np.array([k0 - 0.005, k0, k0 + 0.005])
    ras = np.array([_neutral(bottom, top, k, lo, hi) for k in ks])
    a, b, c = np.polyfit(ks - k0, ras, 2)
    dk = -b / (2 * a)
    return c - b * b / (4 * a), k0 + dk


@pytest.mark.parametrize("pair,ref_ra,ref_k,tol_ra,tol_k,k0,lo,hi", [
    pytest.param("rigid-rigid", 1707.76177710, 3.11632, 2e-5, 2e-5, 3.1163, 1650, 1800,
                 marks=pytest.mark.slow),                 # 3.5 s each; 1e-3 already in the fast suite
    ("rigid-free", 1100.65, 2.682, 5e-3, 5e-4, 2.682, 1050, 1150),  # half a unit of the last digit
    pytest.param("free-free", 27 * np.pi ** 4 / 4, np.pi / np.sqrt(2), 2e-5, 2e-5, 2.2214, 600, 700,
                 marks=pytest.mark.slow)])
def test_onset_at_published_digits(pair, ref_ra, ref_k, tol_ra, tol_k, k0, lo, hi):
    bottom, top = {"rigid-rigid": (0, 0), "rigid-free": (0, cn.free_slip),
                   "free-free": (cn.free_slip, cn.free_slip)}[pair]
    Ra_c, k_c = _critical(bottom, top, k0, lo, hi)
    print(f"\n{pair}: Ra_c = {Ra_c:.6f} (ref {ref_ra}), k_c = {k_c:.5f} (ref {ref_k:.5f})")
    assert abs(Ra_c - ref_ra) < tol_ra and abs(k_c - ref_k) < tol_k, (Ra_c, k_c)
