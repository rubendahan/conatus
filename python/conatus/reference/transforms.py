"""Reference grid <-> coefficient transforms (NumPy/SciPy), the oracle of the C++/CUDA ones.

Conventions (shared with ``reference.evaluate`` and ``discretize``):

- Chebyshev: Gauss–Chebyshev nodes in ascending order, t_j = -cos(pi (j + 1/2) / M),
  x = a + (t + 1) L / 2; coefficients of T_n, n < N.
- RealFourier: x_j = a + L j / M; coefficients stored as pairs
  [cos_0, msin_0, cos_1, msin_1, ...] for f = sum_n c_n cos(k_n (x-a)) + s_n (-sin(k_n (x-a))),
  n < N/2 (N even). msin_0 is always 0; the Nyquist mode is not represented.
- ComplexFourier: coefficients in numpy ``fftfreq`` order for N (N even); the
  Nyquist entry (index N/2) is always 0.
- Hermite, Laguerre, CompoundChebyshev: Gauss–Hermite / Gauss–Laguerre nodes (ascending), or the
  Gauss–Chebyshev nodes of each piece; transforms of ``conatus.discretize.families``.

The grid has M = ``basis.grid_size`` points: floor(N * dealias), or larger when the basis
declares ``grid="smooth7"`` (7-smooth FFT lengths); dealias is the basis attribute, never a
hidden default. ``forward`` from an M-point grid then keeps the first N modes: that
truncation *is* the dealiasing projection, and it is exact for coefficient data
(round trip coeff -> grid -> coeff is the identity to rounding).
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from ..errors import CoordinateError, DtypeError
from ..ir.bases import Basis, Chebyshev, ComplexFourier, RealFourier

__all__ = ["grid", "forward", "backward", "grid_size", "coeff_size"]


def grid_size(b: Basis, dealias: bool = True) -> int:
    """Number of grid points of ``b`` (dealiased or not)."""
    return b.grid_size if dealias else b.size


def coeff_size(b: Basis) -> int:
    """Number of coefficients of ``b``."""
    return b.size


def _check_even(b: Basis) -> None:
    if isinstance(b, (RealFourier, ComplexFourier)) and b.size % 2:
        from ..errors import ResolutionError
        raise ResolutionError(f"{b!r}: Fourier sizes must be even (Nyquist convention)",
                              why="a Fourier series of N modes holds the wavenumbers up to the "
                                  "Nyquist mode N/2, which needs N even",
                              fix=f"an even size for {b.coord.name}, e.g. N = {b.size + 1} or "
                                  f"{b.size - 1}", doc="docs/site/guide/bases.md")


def _family(b: Basis):
    from ..discretize.families import family_of
    return family_of(b)


def grid(b: Basis, dealias: bool = True) -> np.ndarray:
    """1D grid points of ``b`` (dealiased grid by default)."""
    fam = _family(b)
    if fam is not None:
        return fam.grid(dealias)
    M = grid_size(b, dealias)
    a, L = b.bounds[0], b.length
    if isinstance(b, (RealFourier, ComplexFourier)):
        return a + L * np.arange(M) / M
    if isinstance(b, Chebyshev):
        t = -np.cos(np.pi * (np.arange(M) + 0.5) / M)
        if getattr(b, "map", None) is not None:       # mapped basis: y = m(xi) (ir/maps.py)
            return b.map.y(t)
        return a + (t + 1) * L / 2
    raise NotImplementedError(f"reference transforms: {type(b).__name__} not supported yet")


def forward(f: np.ndarray, b: Basis, axis: int = 0, dealias: bool = True) -> np.ndarray:
    """Grid values (M points along ``axis``) -> N coefficients."""
    fam = _family(b)
    if fam is not None:
        return fam.forward(f, axis, dealias)
    _check_even(b)
    f = np.asarray(f)
    M, N = grid_size(b, dealias), b.size
    if f.shape[axis] != M:
        # typed (audit inputs: initial(u=lambda x: np.zeros(3)) ended in a bare error)
        raise CoordinateError(f"expected {M} grid points along axis {axis}, got {f.shape[axis]}",
                              fix="grid values of the shape of the grid (a function of the "
                                  "coordinate grids returns arrays of their broadcast shape)")
    f = np.moveaxis(f, axis, 0)
    if isinstance(b, Chebyshev):
        y = sfft.dct(f, type=2, axis=0) / M
        y[0] /= 2
        sign = (-1.0) ** np.arange(M)
        c = (y * sign.reshape((M,) + (1,) * (f.ndim - 1)))[:N]
    elif isinstance(b, RealFourier):
        if np.iscomplexobj(f):
            # a complex state of a real problem is refused, never made real (audit inputs: it
            # surfaced as a bare TypeError from sim.initial(u=1+1j))
            raise DtypeError("RealFourier takes real data: complex values given for a real field",
                             why="the problem was compiled real; the imaginary part is never "
                                 "dropped for you",
                             fix="real data, or a complex problem: periodic(L, complex=True)")
        F = np.fft.rfft(f, axis=0) / M
        K = N // 2
        c = np.zeros((N,) + f.shape[1:])
        c[0] = F[0].real
        c[2:N:2] = 2 * F[1:K].real
        c[3:N:2] = 2 * F[1:K].imag          # (-sin) coefficient: 2 Im(F_k)
    elif isinstance(b, ComplexFourier):
        F = np.fft.fft(f, axis=0) / M
        K = N // 2
        c = np.zeros((N,) + f.shape[1:], dtype=complex)
        c[:K] = F[:K]
        c[K + 1:] = F[M - K + 1:]
    else:
        raise NotImplementedError(f"reference transforms: {type(b).__name__} not supported yet")
    return np.moveaxis(c, 0, axis)


def backward(c: np.ndarray, b: Basis, axis: int = 0, dealias: bool = True) -> np.ndarray:
    """N coefficients -> grid values on M points (zero-padded spectrum)."""
    fam = _family(b)
    if fam is not None:
        return fam.backward(c, axis, dealias)
    _check_even(b)
    c = np.asarray(c)
    M, N = grid_size(b, dealias), b.size
    if c.shape[axis] != N:
        raise ValueError(f"expected {N} coefficients along axis {axis}, got {c.shape[axis]}")
    c = np.moveaxis(c, axis, 0)
    shape = (M,) + c.shape[1:]
    if isinstance(b, Chebyshev):
        y = np.zeros(shape, dtype=c.dtype)
        sign = (-1.0) ** np.arange(N)
        y[:N] = c * sign.reshape((N,) + (1,) * (c.ndim - 1))
        y[0] *= 2
        f = sfft.idct(y * M, type=2, axis=0)
    elif isinstance(b, RealFourier):
        if np.iscomplexobj(c):
            # Complex coefficients on a real basis (complex LBVP / EVP solutions): transform Re and
            # Im separately. Folding them into one Hermitian spectrum returned a real array with the
            # wrong values, silently.
            c = np.moveaxis(c, 0, axis)
            return (backward(c.real, b, axis, dealias) + 1j * backward(c.imag, b, axis, dealias))
        K = N // 2
        F = np.zeros((M // 2 + 1,) + c.shape[1:], dtype=complex)
        F[0] = c[0]
        F[1:K] = (c[2:N:2] + 1j * c[3:N:2]) / 2
        f = np.fft.irfft(F * M, n=M, axis=0)
    elif isinstance(b, ComplexFourier):
        K = N // 2
        F = np.zeros(shape, dtype=complex)
        F[:K] = c[:K]
        F[M - K + 1:] = c[K + 1:]
        f = np.fft.ifft(F * M, axis=0)
    else:
        raise NotImplementedError(f"reference transforms: {type(b).__name__} not supported yet")
    return np.moveaxis(f, 0, axis)
