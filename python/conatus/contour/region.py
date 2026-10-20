"""Regions of the complex plane for contour methods: the rectangle ``cn.box(re=(a, b), im=(c, d))``.

The boundary is walked counter-clockwise from the corner ``a + ic`` (positive orientation: the
argument principle then counts the zeros inside). A point of the boundary is addressed by a
perimeter parameter ``s`` in ``[0, 4]``: edge ``floor(s)`` (bottom, right, top, left), at the
fraction ``s - floor(s)`` of that edge. The corners are ``s = 0, 1, 2, 3`` (``s = 4`` closes the loop).
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from ..errors import OptionError

__all__ = ["ContourBox", "box"]

_DOC = "docs/site/guide/dynamics/contour.md"


def _interval(value, name: str) -> tuple[float, float]:
    """A pair of finite real numbers ``(lo, hi)`` with ``lo < hi``, refused otherwise."""
    try:
        lo, hi = value
    except (TypeError, ValueError):
        raise OptionError(f"box({name}={value!r}): expected a pair (lo, hi)",
                          fix=f"cn.box(re=(-1.0, 1.0), im=(0.0, 2.0)); {name}= is the range of "
                              f"the {'real' if name == 're' else 'imaginary'} part", doc=_DOC) from None
    for v in (lo, hi):
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            raise OptionError(f"box({name}={value!r}): bounds must be finite real numbers",
                              fix="give two finite floats, e.g. (0.0, 0.5)", doc=_DOC)
    if not lo < hi:
        raise OptionError(f"box({name}={value!r}): the lower bound must be below the upper one",
                          why="an empty or reversed range encloses no region", fix=f"{name}=({min(lo, hi)!r}, "
                          f"{max(lo, hi)!r}) if the order was swapped", doc=_DOC)
    return float(lo), float(hi)


@dataclass(frozen=True)
class ContourBox:
    """The closed rectangle ``re[0] <= Re z <= re[1]``, ``im[0] <= Im z <= im[1]`` and its
    positively oriented boundary. Built by :func:`box`."""

    re: tuple[float, float]
    im: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "re", _interval(self.re, "re"))
        object.__setattr__(self, "im", _interval(self.im, "im"))

    # ------------------------------------------------------------------ geometry
    @property
    def corners(self) -> np.ndarray:
        """The four corners, counter-clockwise from ``re[0] + i im[0]``."""
        (a, b), (c, d) = self.re, self.im
        return np.array([complex(a, c), complex(b, c), complex(b, d), complex(a, d)])

    @property
    def center(self) -> complex:
        """Centre of the box (the origin of the scaled variable of Beyn's moments)."""
        return complex(0.5 * (self.re[0] + self.re[1]), 0.5 * (self.im[0] + self.im[1]))

    @property
    def radius(self) -> float:
        """Half the diagonal: every point of the box is within it of :attr:`center`."""
        return 0.5 * math.hypot(self.re[1] - self.re[0], self.im[1] - self.im[0])

    @property
    def perimeter(self) -> float:
        """Length of the boundary (the walk's smallest step is measured against it)."""
        return 2.0 * ((self.re[1] - self.re[0]) + (self.im[1] - self.im[0]))

    def point(self, s) -> np.ndarray:
        """Boundary point(s) at perimeter parameter ``s`` in ``[0, 4]`` (module docstring)."""
        s = np.asarray(s, dtype=float)
        k = np.clip(np.floor(s).astype(int), 0, 3)
        c = self.corners
        z0, z1 = c[k], c[(k + 1) % 4]
        return z0 + (s - k) * (z1 - z0)

    def edge_length(self, k: int) -> float:
        """Length of edge ``k`` (0 bottom, 1 right, 2 top, 3 left)."""
        c = self.corners
        return float(abs(c[(k + 1) % 4] - c[k]))

    def contains(self, z) -> np.ndarray:
        """True where ``z`` lies in the closed box."""
        z = np.asarray(z, dtype=complex)
        return ((z.real >= self.re[0]) & (z.real <= self.re[1])
                & (z.imag >= self.im[0]) & (z.imag <= self.im[1]))

    def distance_to_boundary(self, z) -> np.ndarray:
        """Distance of an interior point to the boundary (0 outside or on it)."""
        z = np.asarray(z, dtype=complex)
        d = np.minimum.reduce([z.real - self.re[0], self.re[1] - z.real,
                               z.imag - self.im[0], self.im[1] - z.imag])
        return np.maximum(d, 0.0)

    def around(self, z: complex, half: float) -> ContourBox:
        """The square of half-side ``half`` centred on ``z`` (local counts of a cluster)."""
        return ContourBox((z.real - half, z.real + half), (z.imag - half, z.imag + half))

    # ------------------------------------------------------------------ text
    def describe(self) -> str:
        """The two ranges in one line (used by every message about the box)."""
        return (f"Re in [{self.re[0]:.6g}, {self.re[1]:.6g}], "
                f"Im in [{self.im[0]:.6g}, {self.im[1]:.6g}]")

    def __repr__(self) -> str:
        return f"cn.box(re={self.re!r}, im={self.im!r})"

    def explain(self) -> str:
        """The region, its orientation and how its boundary is addressed."""
        return (f"contour region: the closed rectangle {self.describe()} (centre "
                f"{self.center:.6g}, half-diagonal {self.radius:.6g}); boundary walked "
                "counter-clockwise from the lower-left corner (positive orientation: the argument "
                "principle counts the zeros inside); an eigenvalue on the boundary has no count "
                "and is refused (move the box)")


def box(*, re: tuple[float, float], im: tuple[float, float]) -> ContourBox:
    """The rectangle ``re[0] <= Re z <= re[1]``, ``im[0] <= Im z <= im[1]`` of the complex plane,
    for ``m.eigenvalues(..., region=cn.box(...))`` and :mod:`conatus.contour`. Both ranges are
    required (no default region)."""
    return ContourBox(re, im)
