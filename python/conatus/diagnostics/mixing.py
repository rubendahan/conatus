"""Mixing diagnostics (read-only, NumPy or CuPy): scalar variance, mix-norms, background and
available potential energy by sorting, mixing rate and efficiency.


* **Variance** ``<c^2> - <c>^2`` with the grid quadrature.
* **Mix-norm** (Mathew, Mezić & Petzold 2005, Physica D 211, 23): the negative Sobolev norm
  ``||c'||_{H^{-s}}^2 = sum_{k != 0} Lambda(k) |c_k|^2`` of the fluctuation ``c' = c - <c>``,
  ``s = 1/2`` by default (MMP's choice). Two weights, declared: ``"homogeneous"``
  ``Lambda = |k|^{-2s}`` (Lin, Thiffeault & Doering 2011 for s = 1) and ``"mmp"``
  ``Lambda = (1 + |k|^2)^{-s}`` (MMP's weight, with |k| the angular wavenumber of the box; for
  the unit torus it is their ``1 + 4 pi^2 |n|^2``). Fully periodic grids only.
* **Potential energies** (Winters, Lombard, Riley & D'Asaro 1995, JFM 289, 115). With the
  buoyancy b (b = -g rho / rho_0) and height z along the declared vertical axis:
  ``E_p = -int b z dV`` and the background energy ``E_b = -int b z_* dV``, z_* being the
  height of each fluid parcel once all parcels are sorted into the stable state (b increasing
  upward). The sample volumes are the grid quadrature weights (:meth:`Grid.cell_volumes`); each
  column is read as a stack of cells of heights equal to the vertical weights, so E_p and E_b use
  the *same* cell geometry and APE = E_p - E_b >= 0 holds exactly (rearrangement inequality),
  with an O(h^2) cell-centre error against the continuum integral (declared, measured in the
  tests). Horizontal axes may be periodic or bounded; the domain is a box.
* **Mixing rate and efficiency** (Winters et al. 1995; Peltier & Caulfield 2003):
  ``Phi_d = dE_b/dt - Phi_i`` (Phi_i: the declared diffusive flux through the boundaries,
  e.g. kappa A (b_top - b_bottom) / H for a linear conductive state), instantaneous efficiency
  ``eta = Phi_d / (Phi_d + eps)`` and cumulative ``int Phi_d / int (Phi_d + eps)``. The time
  derivative is second-order centred (``numpy.gradient``, declared).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import CoordinateError, OptionError, UnsupportedError
from .grid import Grid, Periodic

__all__ = ["variance", "mix_norm", "potential_energies", "mixing_rate", "mixing_efficiency"]


def variance(grid: Grid, c):
    """Volume variance <c^2> - <c>^2 (0-d array)."""
    m = grid.mean(c)
    return grid.mean((c - m) ** 2)


def mix_norm(grid: Grid, c, *, s: float = 0.5, weight: str = "homogeneous"):
    """H^{-s} mix-norm of the fluctuation of ``c`` on a fully periodic box (see module doc):
    sqrt(sum_{k != 0} Lambda(k) |c_k|^2), |c_k| normalized so that sum_k |c_k|^2 = <c^2>."""
    if not all(isinstance(ax, Periodic) for ax in grid.axes):
        raise UnsupportedError("mix_norm needs a fully periodic grid",
                               why="the Fourier definition of Mathew et al. (2005) is on a torus",
                               fix="restrict to a periodic sub-box, or use variance()")
    if s <= 0:
        raise OptionError(f"mix_norm: s = {s} must be > 0 (s = 0 is the variance)")
    if weight not in ("homogeneous", "mmp"):
        raise OptionError(f"mix_norm weight {weight!r}: 'homogeneous' or 'mmp'")
    xp = grid.xp
    N = float(np.prod(grid.shape))
    C = xp.fft.fftn(c) / N
    k2 = 0
    for i in range(grid.ndim):
        shp = [1] * grid.ndim
        shp[i] = -1
        k2 = k2 + grid.wavenumbers(i).reshape(shp) ** 2
    k2s = xp.where(k2 == 0, 1.0, k2)
    lam = k2s ** (-s) if weight == "homogeneous" else (1.0 + k2s) ** (-s)
    lam = xp.where(k2 == 0, 0.0, lam)                        # the mean is not part of the norm
    return xp.sqrt((lam * xp.abs(C) ** 2).sum())


def potential_energies(grid: Grid, b, *, vertical_axis: int) -> dict:
    """E_p, E_b (sorting) and APE = E_p - E_b of the buoyancy ``b`` (see module doc). Returns
    dict(E_p, E_b, APE, z_ref) as floats (per unit reference density; multiply by rho_0 for
    energies). ``vertical_axis`` is required (declared); gravity points to decreasing z."""
    if not 0 <= vertical_axis < grid.ndim:
        raise CoordinateError(f"vertical_axis={vertical_axis} on a {grid.ndim}D grid")
    if tuple(b.shape) != grid.shape:
        raise CoordinateError(f"buoyancy of shape {tuple(b.shape)} on a grid of shape {grid.shape}")
    xp = grid.xp
    ax = grid.axes[vertical_axis]
    z0, z1 = ax.bounds
    H = z1 - z0
    V = grid.cell_volumes()
    area = grid.volume() / H
    # cell centres of each column: cumulative vertical weights (same geometry as the sorted state)
    wz = grid.weights(vertical_axis)
    zc = z0 + xp.cumsum(wz) - 0.5 * wz
    shp = [1] * grid.ndim
    shp[vertical_axis] = -1
    zc = zc.reshape(shp)
    Ep = -(b * zc * V).sum()
    # background state: sort all parcels by b, stack them from the bottom (volume / area)
    order = xp.argsort(b.reshape(-1), kind="stable")
    bs = b.reshape(-1)[order]
    vs = V.reshape(-1)[order]
    top = xp.cumsum(vs)
    zs = z0 + (top - 0.5 * vs) / area
    Eb = -(bs * zs * vs).sum()
    Ep, Eb = float(Ep), float(Eb)
    return dict(E_p=Ep, E_b=Eb, APE=Ep - Eb, z_ref=float(z0))


def mixing_rate(t: Sequence[float], E_b: Sequence[float], *,
                phi_i: Sequence[float] | float) -> np.ndarray:
    """Irreversible mixing rate Phi_d = dE_b/dt - Phi_i (Winters et al. 1995). ``phi_i`` (the
    boundary diffusive flux, a number or a series) is required: set 0.0 explicitly when the
    boundaries are insulating. Centred second-order derivative (numpy.gradient, declared)."""
    t = np.asarray(t, dtype=float)
    E = np.asarray(E_b, dtype=float)
    if t.shape != E.shape or t.ndim != 1 or len(t) < 3:
        raise CoordinateError("mixing_rate: t and E_b are 1D series of the same length >= 3")
    return np.gradient(E, t, edge_order=2) - np.asarray(phi_i, dtype=float)


def mixing_efficiency(phi_d: Sequence[float], eps: Sequence[float], *,
                      t: Sequence[float] | None = None) -> dict:
    """Instantaneous eta = Phi_d / (Phi_d + eps) and, if ``t`` is given, the cumulative
    efficiency int Phi_d dt / int (Phi_d + eps) dt (trapezoidal rule, declared).
    ``eps`` is the volume-integrated viscous dissipation (same units as Phi_d)."""
    pd = np.asarray(phi_d, dtype=float)
    ep = np.asarray(eps, dtype=float)
    if pd.shape != ep.shape:
        raise CoordinateError("mixing_efficiency: Phi_d and eps series of the same shape")
    den = pd + ep
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(den != 0, pd / np.where(den != 0, den, 1.0), np.nan)
    out: dict[str, Any] = {"eta": eta}
    if t is not None:
        tt = np.asarray(t, dtype=float)
        out["eta_cumulative"] = float(np.trapezoid(pd, tt) / np.trapezoid(den, tt))
    return out
