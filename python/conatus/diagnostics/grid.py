"""Tensor-product physical grids for the diagnostics (NumPy or CuPy), spectral derivatives.

A diagnostic never touches the solver state: it reads *physical values* on a declared grid and
returns new arrays. The grid is a list of axes, each one of

* :class:`Periodic` ``(L, n, a=0)``: nodes ``a + L j / n``; derivatives by FFT (exact on the
  trigonometric interpolant). For even ``n`` the Nyquist mode of an **odd** derivative is set to
  zero (its derivative is not representable on the grid; Trefethen 2000, ch. 3) — declared in
  :meth:`Grid.explain`;
* :class:`Chebyshev` ``(a, b, n)``: Gauss (roots) nodes in ascending order,
  ``x_j = a + (b - a)(1 - cos(pi (j + 1/2) / n)) / 2`` — the convention of
  ``conatus.reference.transforms.grid`` (the dealiased solver grid). Derivatives are exact on the
  degree ``n - 1`` interpolant: values -> Chebyshev coefficients (length-2n FFT) -> coefficient
  recurrence -> values. No dense differentiation matrix is formed.

Quadrature (:meth:`Grid.integrate`, :meth:`Grid.mean`): trapezoidal (spectral) on periodic axes,
Fejér's first rule on Chebyshev–Gauss nodes (exact for polynomials of degree < n).

Array module: ``xp`` is NumPy by default, CuPy when the grid is built with ``xp=cupy`` (or from
CuPy arrays via :func:`array_module`). The identical code runs on both.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import CoordinateError, OptionError, UnsupportedError

__all__ = ["Periodic", "Chebyshev", "Grid", "array_module"]


def array_module(*arrays: Any):
    """NumPy, or CuPy if any argument is a CuPy array (CuPy is never imported otherwise)."""
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy
            return cupy
    return np


@dataclass(frozen=True)
class Periodic:
    """Periodic axis of period ``L`` with ``n`` uniform nodes starting at ``a``."""
    L: float
    n: int
    a: float = 0.0
    kind = "periodic"

    def nodes(self) -> np.ndarray:
        """Grid nodes (host array)."""
        return self.a + self.L * np.arange(self.n) / self.n

    @property
    def bounds(self) -> tuple[float, float]:
        """(a, a + L): the period."""
        return (self.a, self.a + self.L)


@dataclass(frozen=True)
class Chebyshev:
    """Bounded axis [a, b] with ``n`` Chebyshev–Gauss nodes in ascending order."""
    a: float
    b: float
    n: int
    kind = "chebyshev"

    def nodes(self) -> np.ndarray:
        """Grid nodes (host array), ascending."""
        t = -np.cos(np.pi * (np.arange(self.n) + 0.5) / self.n)
        return self.a + (t + 1.0) * (self.b - self.a) / 2.0

    @property
    def bounds(self) -> tuple[float, float]:
        """(a, b)."""
        return (self.a, self.b)

    def fejer_weights(self) -> np.ndarray:
        """Fejér first-rule weights on the nodes (sum = b - a); exact for degree < n."""
        n = self.n
        th = np.pi * (np.arange(n) + 0.5) / n
        k = np.arange(1, n // 2 + 1)
        w = (2.0 / n) * (1.0 - 2.0 * (np.cos(2.0 * np.outer(th, k)) / (4.0 * k**2 - 1.0)).sum(axis=1))
        return w * (self.b - self.a) / 2.0


Axis = Periodic | Chebyshev


def _ix(ndim: int, axis: int, s) -> tuple:
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


def _bshape(v, ndim: int, axis: int):
    shp = [1] * ndim
    shp[axis] = v.shape[0]
    return v.reshape(shp)


class Grid:
    """Declared tensor-product grid ``axes`` (sequence of :class:`Periodic` / :class:`Chebyshev`).

    Field arrays have shape ``grid.shape`` (axis order as declared); vector fields are sequences
    of such arrays (one per axis, same order)."""

    def __init__(self, axes: Sequence[Axis], xp: Any = None) -> None:
        if not axes:
            raise CoordinateError("a diagnostics grid needs at least one axis")
        for ax in axes:
            if not isinstance(ax, (Periodic, Chebyshev)):
                raise OptionError(f"grid axis {ax!r}: Periodic(L, n) or Chebyshev(a, b, n)",
                                  fix="declare every axis explicitly")
            # n an integer (np.arange(64.5) gave 65 nodes without a word), finite increasing
            # bounds (Chebyshev(1, -1, n) gave descending "ascending" nodes, Periodic(0, n) a
            # division by zero)
            if isinstance(ax.n, bool) or not isinstance(ax.n, (int, np.integer)) or ax.n < 2:
                raise CoordinateError(f"grid axis {ax!r}: n must be an integer >= 2",
                                      fix="e.g. Periodic(2*pi, 64) or Chebyshev(-1, 1, 33)")
            lo, hi = (float(v) for v in ax.bounds)
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise CoordinateError(f"grid axis {ax!r}: bounds must be finite with "
                                      "lo < hi (period L > 0)",
                                      fix="Periodic(L > 0, n, a) or Chebyshev(a < b, n)")
        self.axes = tuple(axes)
        self.xp = np if xp is None else xp
        self._cache: dict = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_bases(cls, bases: Sequence[Any], *, dealias: bool = True, xp: Any = None) -> Grid:
        """Grid of a Conatus domain (``conatus.ir.bases``): RealFourier / ComplexFourier ->
        :class:`Periodic`, Chebyshev -> :class:`Chebyshev`, on the dealiased grid by default
        (the grid of ``IVPSolver.grid``). Mapped Chebyshev bases are refused (not uniform in the
        mapped coordinate; declare the axis by hand)."""
        from ..ir.bases import Chebyshev as BCheb
        from ..ir.bases import ComplexFourier, RealFourier
        axes: list[Axis] = []
        for b in bases:
            n = b.grid_size if dealias else b.size
            a0, L = float(b.bounds[0]), float(b.length)
            if isinstance(b, (RealFourier, ComplexFourier)):
                axes.append(Periodic(L, n, a0))
            elif isinstance(b, BCheb):
                if getattr(b, "map", None) is not None:
                    raise UnsupportedError(f"{b!r}: mapped Chebyshev basis",
                                           why="the diagnostics grid differentiates in the "
                                               "unmapped coordinate only",
                                           fix="evaluate the field on an unmapped grid first")
                axes.append(Chebyshev(a0, a0 + L, n))
            else:
                raise UnsupportedError(f"diagnostics grid: basis {type(b).__name__} not supported",
                                       fix="Fourier and Chebyshev axes only (Cartesian domains)")
        return cls(axes, xp=xp)

    def on(self, xp: Any) -> Grid:
        """The same grid with another array module (e.g. ``cupy``)."""
        return Grid(self.axes, xp=xp)

    # ------------------------------------------------------------------ geometry
    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape of a scalar field."""
        return tuple(ax.n for ax in self.axes)

    def nodes(self, axis: int):
        """Nodes of ``axis`` (device array)."""
        return self.xp.asarray(self.axes[axis].nodes())

    def mesh(self) -> tuple:
        """Coordinate arrays broadcast to ``shape`` (``indexing='ij'``)."""
        return tuple(self.xp.meshgrid(*[self.nodes(i) for i in range(self.ndim)], indexing="ij"))

    def extent(self) -> list[list[float]]:
        """[[lo, hi], ...] per axis: the domain box (periodic: one period)."""
        return [list(map(float, ax.bounds)) for ax in self.axes]

    def explain(self) -> str:
        """Every axis, its nodes and the derivative/quadrature rule used on it."""
        parts = []
        for i, ax in enumerate(self.axes):
            if isinstance(ax, Periodic):
                parts.append(f"axis {i}: periodic L={ax.L:g}, {ax.n} uniform nodes from {ax.a:g}; "
                             "FFT derivative (odd-order Nyquist mode set to 0), trapezoidal rule")
            else:
                parts.append(f"axis {i}: Chebyshev [{ax.a:g}, {ax.b:g}], {ax.n} Gauss nodes "
                             "(ascending); coefficient-recurrence derivative, Fejér-1 rule")
        return "Grid(" + "; ".join(parts) + f"; backend {self.xp.__name__})"

    # ------------------------------------------------------------------ periodic spectra
    def wavenumbers(self, axis: int, *, real: bool = False):
        """Angular wavenumbers of a periodic axis (FFT order; ``real``: rfft half-spectrum)."""
        ax = self.axes[axis]
        if not isinstance(ax, Periodic):
            raise CoordinateError(f"axis {axis} is not periodic: no wavenumbers")
        f = np.fft.rfftfreq(ax.n, 1.0 / ax.n) if real else np.fft.fftfreq(ax.n, 1.0 / ax.n)
        return self.xp.asarray(2.0 * np.pi * f / ax.L)

    # ------------------------------------------------------------------ derivatives
    def d(self, f, axis: int, order: int = 1):
        """Spectral derivative of order ``order`` of the physical-values array ``f`` along grid
        axis ``axis``. ``f`` has the grid shape as its trailing dimensions (leading dimensions
        are a batch). Real input gives real output; complex input is differentiated as re + i im."""
        xp = self.xp
        if order < 0:
            raise OptionError(f"derivative order {order} < 0")
        if order == 0:
            return f
        if not 0 <= axis < self.ndim:
            raise CoordinateError(f"derivative along axis {axis}: the grid has {self.ndim} axes")
        if xp.iscomplexobj(f):
            return self.d(f.real, axis, order) + 1j * self.d(f.imag, axis, order)
        ax = self.axes[axis]
        aa = axis + f.ndim - self.ndim             # array axis
        if aa < 0 or f.shape[aa] != ax.n:
            raise CoordinateError(f"derivative along axis {axis}: array of shape {f.shape} on a "
                                  f"grid of shape {self.shape}")
        if isinstance(ax, Periodic):
            return self._d_periodic(f, aa, order, ax)
        c = self.cheb_coeffs(f, aa, ax)
        for _ in range(order):
            c = self._cheb_diff(c, aa, ax)
        return self.cheb_values(c, aa, ax)

    def grad(self, f) -> list:
        """Gradient of a scalar field: one derivative per axis."""
        return [self.d(f, i) for i in range(self.ndim)]

    def laplacian(self, f):
        """Sum of the second derivatives along every axis."""
        return sum(self.d(f, i, 2) for i in range(self.ndim))

    def _d_periodic(self, f, axis: int, order: int, ax: Periodic):
        xp = self.xp
        n = ax.n
        F = xp.fft.rfft(f, axis=axis)
        ik = 1j * xp.asarray(2.0 * np.pi * np.fft.rfftfreq(n, 1.0 / n) / ax.L)
        m = ik**order
        if order % 2 == 1 and n % 2 == 0:
            m = m.copy()
            m[-1] = 0.0                          # odd derivative of the Nyquist mode: zero (declared)
        return xp.fft.irfft(F * _bshape(m, f.ndim, axis), n=n, axis=axis)

    # ------------------------------------------------------------------ Chebyshev transforms
    def _twiddle(self, n: int, sign: float):
        key = ("tw", n, sign)
        w = self._cache.get(key)
        if w is None:
            k = np.arange(n)
            w = self._cache[key] = self.xp.asarray(np.exp(sign * 1j * np.pi * k / (2 * n)) * (-1.0) ** k)
        return w

    def cheb_coeffs(self, f, axis: int, ax: Chebyshev | None = None):
        """Chebyshev coefficients (n of them) of real values on the Gauss nodes along array axis
        ``axis`` (``ax``: its declared axis, default ``self.axes[axis]``):
        c_k = (-1)^k (2/n) sum_j f_j cos(k theta_j), halved at k = 0 (as xp_transforms)."""
        xp = self.xp
        ax = ax or self.axes[axis]
        n, nd = ax.n, f.ndim
        F = xp.fft.rfft(f, n=2 * n, axis=axis)[_ix(nd, axis, slice(0, n))]
        c = (_bshape(self._twiddle(n, -1.0), nd, axis) * F).real * (2.0 / n)
        c[_ix(nd, axis, 0)] *= 0.5
        return c

    def cheb_values(self, c, axis: int, ax: Chebyshev | None = None):
        """Inverse of :meth:`cheb_coeffs` (values on the Gauss nodes)."""
        xp = self.xp
        ax = ax or self.axes[axis]
        n, nd = ax.n, c.ndim
        h = np.full(n, 0.5)
        h[0] = 1.0
        a = _bshape(self._twiddle(n, 1.0) * xp.asarray(h), nd, axis) * c
        return xp.fft.irfft(a, n=2 * n, axis=axis)[_ix(nd, axis, slice(0, n))] * (2 * n)

    def _cheb_diff(self, c, axis: int, ax: Chebyshev):
        """Coefficients of the derivative: d_{k-1} = d_{k+1} + 2 k c_k, d_0 halved."""
        xp = self.xp
        n = ax.n
        cm = xp.moveaxis(c, axis, 0)
        d = xp.zeros_like(cm)
        if n >= 2:
            d[n - 2] = 2 * (n - 1) * cm[n - 1]
        for k in range(n - 2, 0, -1):           # O(n) vector operations, never a dense matrix
            d[k - 1] = d[k + 1] + 2 * k * cm[k]
        d[0] *= 0.5
        return xp.moveaxis(d, 0, axis) * (2.0 / (ax.b - ax.a))

    # ------------------------------------------------------------------ quadrature
    def weights(self, axis: int):
        """1D quadrature weights of ``axis`` (sum = length): a new array (the grid keeps its
        own copy: an in-place change of the returned array used to change every later
        ``integrate``, ``mean`` and ``cell_volumes`` of this grid, audit state)."""
        return self._weights(axis).copy()

    def _weights(self, axis: int):
        """The cached weights of ``axis`` (never handed out: :meth:`weights` copies them)."""
        ax = self.axes[axis]
        key = ("w", axis)
        w = self._cache.get(key)
        if w is None:
            w = np.full(ax.n, ax.L / ax.n) if isinstance(ax, Periodic) else ax.fejer_weights()
            w = self._cache[key] = self.xp.asarray(w)
        return w

    def volume(self) -> float:
        """Measure of the domain (product of the lengths)."""
        return float(np.prod([ax.bounds[1] - ax.bounds[0] for ax in self.axes]))

    def integrate(self, f):
        """Integral of a scalar field over the domain (0-d array on the field's device)."""
        out = f
        for axis in range(self.ndim - 1, -1, -1):
            out = self.xp.tensordot(out, self._weights(axis), axes=([axis], [0]))
        return out

    def mean(self, f):
        """Volume average: integral / volume."""
        return self.integrate(f) / self.volume()

    def cell_volumes(self):
        """Quadrature weight of every node (outer product of the 1D weights), shape ``shape``.
        Used as the volume of each sample when sorting (background potential energy)."""
        w = self.weights(0)                     # a copy: the 1D result is the caller's own
        for axis in range(1, self.ndim):
            w = w[..., None] * _bshape(self._weights(axis), axis + 1, axis)
        return w
