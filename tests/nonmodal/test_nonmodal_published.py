"""Published non-modal values (slow; Slurm job nonmodal_study).

References (values as printed in the papers; PDFs not on the store yet, see docs/mathematics/
nonmodal.md, "to be re-checked"):
  Butler & Farrell 1992, Phys. Fluids A 4, 1637; Reddy & Henningson 1993, JFM 252, 209:
    plane Poiseuille Re = 5000, (α, β) = (0, 2.044): G_max = 4897 at t = 379;
    plane Poiseuille Re = 1000, (α, β) = (0, 2.044): G_max = 196 at t ≈ 76;
    plane Couette Re = 1000, (α, β) = (0.035, 1.60): G_max = 1185 at t = 117 (Butler & Farrell
    1992, Table 1).
  Orszag 1971, JFM 50, 689: c = 0.23752649 + 0.00373967i (Re = 10⁴, α = 1).
"""
from __future__ import annotations

import numpy as np
import pytest

from conatus.nonmodal import Y, max_growth, parallel_shear_flow, state_space

pytestmark = pytest.mark.slow


def gmax(U, kx, kz, Re, N, tb):
    S = parallel_shear_flow(U, kx=kx, kz=kz, Re=Re, walls="no-slip")
    ss = state_space(S.discretize(N), infinite_tol=1e-10)
    return max_growth(ss, tb, xtol=1e-6, n_scan=40)


@pytest.mark.parametrize("Re, tb, G_ref, t_ref", [(5000.0, (300, 460), 4897, 379),
                                                  (1000.0, (50, 110), 196, 76)])
def test_poiseuille_optimal_growth_butler_farrell(Re, tb, G_ref, t_ref):
    g48, t48 = gmax(1 - Y ** 2, 0.0, 2.044, Re, 48, tb)
    g64, t64 = gmax(1 - Y ** 2, 0.0, 2.044, Re, 64, tb)
    assert abs(g64 / g48 - 1) < 1e-9                  # converged in N
    assert abs(g64 - G_ref) <= 0.5                     # printed to the unit
    assert abs(t64 - t_ref) <= 0.5 if Re == 5000.0 else abs(t64 - t_ref) <= 1.0


def test_couette_optimal_growth_butler_farrell():
    """Quality pass 9: 1184.57 at t = 116.9 (N = 48 and 64 agree to 1e-9), the printed 1185 / 117."""
    g48, t48 = gmax(Y, 0.035, 1.6, 1000.0, 48, (60, 200))
    g64, t64 = gmax(Y, 0.035, 1.6, 1000.0, 64, (60, 200))
    assert abs(g64 / g48 - 1) < 1e-9
    assert abs(g64 - 1185) <= 0.5 and abs(t64 - 117) <= 0.5


def test_orszag_eigenvalue_converged():
    S = parallel_shear_flow(1 - Y ** 2, kx=1.0, kz=0.0, Re=1e4, walls="no-slip")
    ss = state_space(S.discretize(100), infinite_tol=1e-10)
    c = 1j * ss.eigenvalues()
    c0 = c[np.argmax(c.imag)]
    assert abs(c0 - (0.23752648882 + 0.00373967062j)) < 1e-10
