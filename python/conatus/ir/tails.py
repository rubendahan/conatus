"""Spectral-tail measures of coefficient arrays: the one place where each tail is
defined. Three measures are in use, each for its own question; they are not
interchangeable, and every message that prints a tail names its measure.

* **coefficient tail** (:func:`coefficient_tail`), an *amplitude*: ``max |a[-k:]| / max |a|``,
  the largest of the last ``k`` coefficients over the largest one. Used for sampled
  non-constant coefficients (NCC, ``NCC_TAIL_TOL``, k = 3), coordinate-map metrics (k = 3) and the
  a-posteriori check of a verification report (k = 4).
* **envelope tail** (:func:`envelope_tail`), an *amplitude*: ``E[M - n_tail] / max(E)`` with
  ``E`` the monotone envelope of ``|c|`` over the mode numbers of one direction
  (:func:`mode_envelope`) and ``n_tail = max(TAIL_MIN_MODES, ceil(TAIL_FRACTION M))``
  (:func:`tail_window`). Used by ``conatus.resolution`` (``TAIL_TARGET``, ``TAIL_MARGINAL``) and
  by the run-time check of the time steppers (``RUN_TAIL_TOL``).
* **energy tail** (:func:`energy_tails`), an *energy fraction*: the energy of the last
  ``SPURIOUS_TAIL_FRACTION`` of the wall-normal coefficients of an eigenvector over its energy
  (``SPURIOUS_TAIL_TOL``, ``UNRESOLVED_TAIL_TOL``). An energy tail t corresponds to an amplitude of
  about sqrt(t).

Mode numbers (:func:`mode_numbers`): the polynomial degree for Chebyshev/Legendre/Jacobi;
``|n|`` for Fourier (RealFourier pair index; ComplexFourier ``|fftfreq|`` with the always-zero
Nyquist entry excluded).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import UnsupportedError
from ..tolerances import SPURIOUS_SIGNIFICANT, SPURIOUS_TAIL_FRACTION

__all__ = ["coefficient_tail", "mode_numbers", "mode_envelope", "tail_window", "envelope_tail",
           "energy_tails", "TAIL_FRACTION", "TAIL_MIN_MODES"]

TAIL_FRACTION = 0.10       # the envelope-tail window is the last 10 % of the mode numbers ...
TAIL_MIN_MODES = 2         # ... and at least 2 mode numbers


def coefficient_tail(a: Any, k: int = 3) -> float:
    """Coefficient tail (an amplitude): ``max |a[-k:]| / max |a|`` of the 1-D array ``a``; 0.0 for
    an empty or identically zero array (no tail to measure)."""
    a = np.abs(np.asarray(a))
    amax = float(a.max()) if a.size else 0.0
    return float(a[-k:].max()) / amax if amax > 0 else 0.0


def basis_family(basis: Any) -> str:
    """'poly' (degree = index), 'rfourier' or 'cfourier'; UnsupportedError otherwise."""
    from .bases import ComplexFourier, Jacobi, RealFourier
    if isinstance(basis, RealFourier):
        return "rfourier"
    if isinstance(basis, ComplexFourier):
        return "cfourier"
    if isinstance(basis, Jacobi):
        return "poly"
    raise UnsupportedError(
        f"resolution report: basis {type(basis).__name__} is not supported",
        where=f"direction {basis.coord.name}",
        why="only Chebyshev/Legendre/Jacobi (degree) and Fourier (|wavenumber|) coefficients have "
            "a mode number this module interprets; guessing one would give a wrong tail",
        fix="inspect the coefficients yourself (solver.coeffs(name)) for this basis")


def mode_numbers(basis: Any) -> tuple[np.ndarray, int]:
    """Mode number of every coefficient index (-1 = excluded) and the number of mode numbers."""
    N, fam = basis.size, basis_family(basis)
    if fam == "poly":
        return np.arange(N), N
    if fam == "rfourier":
        return np.arange(N) // 2, N // 2
    m = np.abs(np.rint(np.fft.fftfreq(N) * N)).astype(int)
    if N % 2 == 0:
        m[N // 2] = -1                 # Nyquist entry, always zero by convention
    return m, (N + 1) // 2


def mode_envelope(a: np.ndarray, mnum: np.ndarray, M: int) -> np.ndarray:
    """Monotone envelope ``E[m] = max_{m' >= m} e[m']`` with ``e[m]`` the largest of the host values
    ``a[i]`` (one per coefficient index, already a max of |c| over the other axes) whose mode number
    ``mnum[i]`` is m; indices of mode number -1 are excluded. Spectra with zero odd or even modes
    (by symmetry) do not fool the tail."""
    e = np.zeros(M)
    for i, m in enumerate(mnum):
        if m >= 0:
            e[m] = max(e[m], a[i])
    return np.maximum.accumulate(e[::-1])[::-1]


def tail_window(M: int) -> int:
    """Number of mode numbers in the envelope-tail window: max(TAIL_MIN_MODES, ceil(TAIL_FRACTION
    M)), at most M."""
    return min(M, max(TAIL_MIN_MODES, math.ceil(TAIL_FRACTION * M)))


def envelope_tail(E: np.ndarray, emax: float) -> float:
    """Envelope tail (an amplitude): ``E[M - n_tail] / emax`` for the monotone envelope ``E`` of M
    mode numbers and its maximum ``emax > 0``. The caller checks that the window leaves the mean
    out (``M - tail_window(M) >= 1``): a window holding every mode number has tail 1 whatever the
    data."""
    M = E.size
    return float(E[M - tail_window(M)] / emax)


def energy_tails(layout: list, V: np.ndarray) -> np.ndarray:
    """Energy tail (an energy fraction) of each column of ``V``: for every unknown of ``layout``
    (``(name, index array (rows, nz), pieces)``, -1 = outside the symmetry subspace), the energy in
    the last ``max(2, ceil(SPURIOUS_TAIL_FRACTION n))`` coefficients of each wall-normal piece of n
    coefficients over the unknown's energy; the largest over the unknowns whose energy is at least
    ``SPURIOUS_SIGNIFICANT`` of the largest one."""
    V = np.asarray(V)
    m = V.shape[1]
    ratios, energies = [], []
    for _name, idx, pieces in layout:
        C = np.where((idx >= 0)[..., None], V[np.maximum(idx, 0)], 0.0)   # (rows, nz, m)
        E = np.abs(C) ** 2
        tot = E.sum(axis=(0, 1))
        tail = np.zeros(m)
        for a, b in pieces:
            k = max(2, int(np.ceil(SPURIOUS_TAIL_FRACTION * (b - a))))
            tail = tail + E[:, b - k:b, :].sum(axis=(0, 1))
        energies.append(tot)
        ratios.append(np.where(tot > 0, tail / np.where(tot > 0, tot, 1.0), 0.0))
    energies = np.array(energies)
    ratios = np.array(ratios)
    top = energies.max(axis=0)
    significant = energies >= SPURIOUS_SIGNIFICANT * np.where(top > 0, top, 1.0)
    return np.where(significant, ratios, 0.0).max(axis=0)
