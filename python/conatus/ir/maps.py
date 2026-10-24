"""Mapped Chebyshev bases: infinite and semi-infinite intervals.

A mapped basis expands in Chebyshev polynomials T_n(xi), xi in [-1, 1], of a computational
coordinate related to the physical one by a declared map y = m(xi) (Boyd, *Chebyshev and Fourier
Spectral Methods*, 2nd ed. 2001, ch. 17). Derivatives follow the chain rule

    d/dy = mu(xi) d/dxi,        mu = 1 / m'(xi)  (the metric),

and the discretizer multiplies by mu as a non-constant coefficient in the ultraspherical spaces
(Olver–Townsend). When mu is a polynomial (tanh map: mu = (1 - xi^2)/L; rational map for a
half-line: mu = (1 - xi)^2/(2L)) the operators stay **banded exactly**; otherwise mu is expanded in
Chebyshev series to machine precision and the number of kept coefficients (= bandwidth) and the
dropped tail are reported, never hidden.

Maps provided (all declared in ``explain()``):

- ``TanhMap(L)``: y = c + L artanh(xi) on (-inf, inf), mu = (1 - xi^2)/L (polynomial). Note that
  e^{-a|y|} becomes (1 -+ xi)^{a L / 2}: an algebraic endpoint singularity unless a L / 2 is an
  integer (Boyd §17.6): convergence in N depends on L, check it.
- ``TanMap(L)``: y = c + L tan(pi xi / 2) on (-inf, inf), mu = (2/(pi L)) cos^2(pi xi / 2) (entire,
  expanded to machine precision). Exponential decay becomes C-infinity flat at xi = +-1.
- ``RationalMap(L)``: y = y0 + L (1 + xi)/(1 - xi) on [y0, inf), mu = (1 - xi)^2/(2L) (polynomial;
  the basis is Boyd's rational Chebyshev TL_n).

``contour=delta`` (infinite maps): the path is shifted into the complex plane,
y(xi) = m(xi) - i delta (the metric is unchanged, so a polynomial metric stays exact). For
coefficients analytic in the strip -delta <= Im y <= 0 and modes decaying at both ends, the
eigenvalues of modes whose critical points lie above the path are unchanged (Cauchy), and the
discretized continuous spectrum U(y) = c moves below the real axis (for U = tanh y:
Im tanh(y - i delta) = -sin(2 delta)/(cosh 2y + cos 2 delta) < 0); this is how inviscid neutral
modes are continued through c_i = 0 without a critical-layer singularity on the path. The path,
not the physics, changes: the eigenfunction on the grid is its analytic continuation (declared).
Boundary conditions are only allowed at the endpoints (+-inf) on a shifted path.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.fft as sfft

from .bases import _REQUIRED, Basis, Chebyshev
from .coords import Coordinate
from ..errors import CoordinateError, OptionError, ResolutionError
from .tails import coefficient_tail

__all__ = ["CoordinateMap", "TanhMap", "TanMap", "RationalMap", "MappedChebyshev", "is_mapped"]

METRIC_TAIL = 1e-15      # relative size below which trailing metric coefficients are dropped (reported)


def _cheb_coeffs(f, n: int) -> np.ndarray:
    """Chebyshev coefficients of f on n Gauss–Chebyshev points (ascending), as reference.transforms."""
    t = -np.cos(np.pi * (np.arange(n) + 0.5) / n)
    v = np.asarray(f(t))
    def dct(a):
        y = sfft.dct(a, type=2) / n
        y[0] /= 2
        return y * (-1.0) ** np.arange(n)
    return dct(v.real) + 1j * dct(v.imag) if np.iscomplexobj(v) else dct(v)


class CoordinateMap:
    """y = m(xi), xi in [-1, 1]; subclasses give m, m' and (if any) exact metric coefficients."""

    kind = "abstract"
    infinite = True                # both ends at infinity (else [y0, inf))

    def __init__(self, L: float, origin: float = 0.0, contour: float = 0.0) -> None:
        if not L > 0:
            raise OptionError(f"map scale L must be > 0, got {L}", fix='a positive map scale, e.g. L=1.0')
        if contour and not self.infinite:
            raise OptionError("contour deformation is implemented for infinite maps only",
                              fix="deform the contour of an infinite map only")
        self.L, self.origin, self.contour = float(L), float(origin), float(contour)

    # -- to override
    def _y(self, xi): raise NotImplementedError
    def _dy(self, xi): raise NotImplementedError
    def _xi(self, y): raise NotImplementedError
    def _exact_metric(self) -> np.ndarray | None:
        return None

    # -- public
    @property
    def bounds(self) -> tuple[float, float]:
        """Physical interval of y: (origin, inf) on the half line, (-inf, inf) on the whole
        line."""
        return (-math.inf if self.infinite else self.origin, math.inf)

    def y(self, xi):
        """Physical coordinate (complex on a deformed contour)."""
        xi = np.asarray(xi, float)
        out = self._y(xi)
        if self.contour:
            out = out - 1j * self.contour          # constant shift: m'(xi) unchanged
        return out

    def dy(self, xi):
        """dy/dxi at ``xi``."""
        return self._dy(np.asarray(xi, float))

    def metric(self, xi):
        """mu = dxi/dy = 1/m'(xi) (finite at xi = +-1 where m' is infinite)."""
        return 1.0 / self.dy(xi)

    def xi_of(self, y: float) -> float:
        """Computational coordinate of a physical position (+-inf -> +-1)."""
        if math.isinf(y):
            return 1.0 if y > 0 else -1.0
        if self.contour:
            raise OptionError("finite positions are not on a deformed contour: only +-inf traces "
                              "are allowed (the contour meets the real axis at infinity only)",
                              fix="evaluate +-inf traces on a deformed contour (finite positions are off it)")
        return float(self._xi(float(y)))

    def metric_coeffs(self, n: int) -> tuple[np.ndarray, str]:
        """Chebyshev coefficients of mu and how they were obtained (declared)."""
        ex = self._exact_metric()
        if ex is not None:
            return ex, f"exact polynomial metric ({ex.size} coefficients, banded)"
        m = max(2 * n, 64)
        c = _cheb_coeffs(self.metric, m)
        amax = np.abs(c).max()
        keep = int(np.nonzero(np.abs(c) > METRIC_TAIL * amax)[0][-1]) + 1
        if keep > 0.9 * m:
            raise ResolutionError(f"{self!r}: the metric is not resolved by {m} Chebyshev coefficients "
                                  f"(tail {coefficient_tail(c):.1e}); choose another map",
                                  fix="more Chebyshev coefficients for the metric, or a smoother map")
        dropped = float(np.abs(c[keep:]).max() / amax) if keep < m else 0.0
        return c[:keep], (f"metric expanded to machine precision: {keep} Chebyshev coefficients "
                          f"(bandwidth {keep}), dropped tail {dropped:.1e} relative")

    @property
    def key(self) -> tuple:
        """Structural identity (kind, scale, origin, complex shift)."""
        return (self.kind, self.L, self.origin, self.contour)

    def __repr__(self) -> str:
        c = f", contour={self.contour:g}" if self.contour else ""
        o = f", origin={self.origin:g}" if self.origin else ""
        return f"{type(self).__name__}(L={self.L:g}{o}{c})"

    def formula(self) -> str:
        """y(xi) and dxi/dy as text (each map defines it)."""
        raise NotImplementedError

    def explain(self) -> str:
        """The map, its formula and interval, and the complex shift of the path if any."""
        s = f"{self!r}: {self.formula()}, y in {self.bounds}"
        if self.contour:
            s += (f"; path shifted into the complex plane, y(xi) - {self.contour:g} i: eigenvalues "
                  "of modes analytic above the path are unchanged, the continuous spectrum moves "
                  "below the real axis (analytic continuation, declared)")
        return s


class TanhMap(CoordinateMap):
    """Infinite line y in (-inf, inf): y = origin + L artanh(xi)."""
    kind = "tanh"

    def _y(self, xi):
        with np.errstate(divide="ignore"):
            return self.origin + self.L * np.arctanh(xi)

    def _dy(self, xi):
        with np.errstate(divide="ignore"):
            return self.L / (1 - xi ** 2)

    def metric(self, xi):
        """dxi/dy = (1 - xi^2)/L at ``xi``."""
        xi = np.asarray(xi, float)
        return (1 - xi ** 2) / self.L

    def _xi(self, y):
        return np.tanh((y - self.origin) / self.L)

    def _exact_metric(self):
        return np.array([0.5, 0.0, -0.5]) / self.L          # (1 - xi^2)/L = (T0 - T2)/(2L)

    def formula(self) -> str:
        """y = origin + L artanh(xi) and its metric, as text."""
        return f"y = {self.origin:g} + {self.L:g} artanh(xi), dxi/dy = (1 - xi^2)/{self.L:g}"


class TanMap(CoordinateMap):
    """Infinite line y in (-inf, inf): y = origin + L tan(pi xi / 2)."""
    kind = "tan"

    def _y(self, xi):
        with np.errstate(divide="ignore", over="ignore"):
            y = self.origin + self.L * np.tan(np.pi * xi / 2)
        return np.where(np.abs(xi) == 1, np.sign(xi) * np.inf, y)

    def _dy(self, xi):
        with np.errstate(divide="ignore"):
            return self.L * np.pi / 2 / np.cos(np.pi * xi / 2) ** 2

    def metric(self, xi):
        """dxi/dy = 2 cos^2(pi xi/2)/(pi L) at ``xi``."""
        xi = np.asarray(xi, float)
        c2 = np.cos(np.pi * xi / 2) ** 2
        return c2 / (self.L * np.pi / 2)

    def _xi(self, y):
        return 2 / np.pi * np.arctan((y - self.origin) / self.L)

    def formula(self) -> str:
        """y = origin + L tan(pi xi/2) and its metric, as text."""
        return (f"y = {self.origin:g} + {self.L:g} tan(pi xi/2), "
                f"dxi/dy = 2 cos^2(pi xi/2)/(pi {self.L:g})")


class RationalMap(CoordinateMap):
    """Half line y in [origin, inf): y = origin + L (1 + xi)/(1 - xi) (rational Chebyshev TL)."""
    kind = "rational"
    infinite = False

    def _y(self, xi):
        with np.errstate(divide="ignore"):
            return np.where(xi == 1, np.inf, self.origin + self.L * (1 + xi) / (1 - xi))

    def _dy(self, xi):
        with np.errstate(divide="ignore"):
            return 2 * self.L / (1 - xi) ** 2

    def metric(self, xi):
        """dxi/dy = (1 - xi)^2/(2 L) at ``xi``."""
        xi = np.asarray(xi, float)
        return (1 - xi) ** 2 / (2 * self.L)

    def _xi(self, y):
        if y < self.origin:
            raise CoordinateError(f"position {y} outside [{self.origin}, inf)",
                                  fix="a position inside the mapped interval")
        return (y - self.origin - self.L) / (y - self.origin + self.L)

    def _exact_metric(self):
        return np.array([0.75, -1.0, 0.25]) / self.L        # (1 - xi)^2/(2L) in T0, T1, T2

    def formula(self) -> str:
        """y = origin + L (1 + xi)/(1 - xi) and its metric, as text."""
        return (f"y = {self.origin:g} + {self.L:g} (1 + xi)/(1 - xi) (rational Chebyshev TL), "
                f"dxi/dy = (1 - xi)^2/(2 {self.L:g})")


class MappedChebyshev(Chebyshev):
    """Chebyshev T_n(xi) with a declared map y = m(xi) onto an (semi-)infinite interval.

    ``bounds`` are the physical ones (e.g. (-inf, inf)); the collocation grid is m(xi_j) at the
    Gauss–Chebyshev points xi_j. Supported by the discretizer (EVP, LBVP, NLBVP) and the reference
    evaluator; refused by the time steppers and by symmetry subspaces (explicitly)."""

    family = "Chebyshev T in a mapped coordinate"

    def __init__(self, coord: Coordinate, size: int, map: CoordinateMap,
                 dealias: Any = _REQUIRED, grid: str = "floor") -> None:
        if not isinstance(map, CoordinateMap):
            raise TypeError("map must be a CoordinateMap (TanhMap, TanMap, RationalMap)")
        self.map = map
        super().__init__(coord, size, map.bounds, dealias, grid)

    def _params(self) -> tuple:
        return super()._params() + self.map.key

    def coeff_space(self) -> str:
        """Chebyshev T_n of xi with the map, as text."""
        return f"T_n(xi), n = 0..{self.size - 1}, {self.map.explain()}"

    def __repr__(self) -> str:
        return (f"MappedChebyshev({self.coord.name}, size={self.size}, map={self.map!r}, "
                f"dealias={self.dealias:g})")


def is_mapped(b: Basis | None) -> bool:
    """True for a MappedChebyshev basis."""
    return isinstance(b, MappedChebyshev)
