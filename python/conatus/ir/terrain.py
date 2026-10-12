"""Terrain-following map of a channel between two surfaces (mapped 2D/3D domains).

The physical domain is periodic in the horizontal coordinates x (and y) and bounded in z by two
surfaces, z = b(x[, y]) (bottom) and z = t(x[, y]) (top), t > b. A computational coordinate
xi in [-1, 1] follows them (Gal-Chen & Somerville, J. Comput. Phys. 17, 1975):

    z = b + (t - b) (1 + xi) / 2,        xi = 2 (z - b) / (t - b) - 1.

Physical derivatives by the chain rule (d = x or y, primes = horizontal derivatives at fixed z):

    dz      = J dxi,                                J  = 2 / (t - b)
    dd|_z   = dd|_xi + (A_d + B_d xi) dxi,          A_d = -J (b_d + t_d) / 2,
                                                    B_d = -J (t_d - b_d) / 2
    dV      = H dxi dx [dy],                        H  = (t - b) / 2 = 1 / J.

Every metric coefficient is a polynomial of degree <= 1 in xi times a function of the horizontal
coordinates only: along xi the operators stay banded exactly (Chebyshev/ultraspherical); along
the periodic directions they are convolutions by the Fourier coefficients of J, H, A_d, B_d,
which couple the Fourier modes (bandwidth = number of kept metric coefficients, reported).

This module only describes the map (surfaces, metric, their Fourier analysis and its
resolution check). The discretization is ``conatus.discretize.terrain_plan``; the native
geometry is ``conatus.model.terrain`` (``cn.MappedBox``, ``cn.between``).
"""
from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bases import _REQUIRED, Chebyshev
from .coords import Coordinate
from ..errors import CoordinateError, ResolutionError

__all__ = ["Surface", "TerrainMap", "TerrainChebyshev", "MetricReport", "is_terrain"]

# Fourier coefficients of a metric function below this fraction of its largest one are the
# rounding of its FFT on the fine grid (double precision); they are dropped and the count of kept
# coefficients (the Fourier bandwidth of the coupled operator) is reported by explain().
METRIC_ROUNDING = 1e-15
# The surfaces are sampled on a fine grid (at least this many points per periodic direction);
# their own Fourier tail there must be at the rounding level, or the surface is refused.
FINE_MIN = 256
SURFACE_TAIL = 1e-13


@dataclass(frozen=True)
class Surface:
    """z = value (a number) or z = f(x[, y]) (a callable of the periodic coordinates, in order)."""

    value: Any
    text: str

    @property
    def flat(self) -> bool:
        return isinstance(self.value, numbers.Real)

    def __call__(self, *grids: np.ndarray) -> np.ndarray:
        if self.flat:
            shape = np.broadcast_shapes(*[np.shape(g) for g in grids]) if grids else ()
            return np.full(shape, float(self.value))
        v = np.asarray(self.value(*grids), dtype=float)
        shape = np.broadcast_shapes(*[np.shape(g) for g in grids])
        return np.broadcast_to(v, shape).copy()


@dataclass
class MetricReport:
    """Fourier analysis of one metric function at one resolution (never silent)."""

    name: str
    tail: float              # relative tail of its coefficients sampled on the N-point grid
    kept: int                # Fourier coefficients kept in the operator (|p| <= 2K, > rounding)
    total: int               # coefficients available (|p| <= 2K in every direction)
    tol: float

    def explain(self) -> str:
        return (f"metric {self.name}: {self.kept} of {self.total} Fourier coefficients kept "
                f"(Fourier bandwidth), sampled tail {self.tail:.1e} on the grid "
                f"(accepted up to ncc_tail_tol = {self.tol:g})")


class TerrainMap:
    """Map of a channel between ``bottom`` and ``top`` surfaces over periodic horizontal axes.

    ``periodic``: sequence of (Coordinate name, (a, b)) of the horizontal periodic directions, in
    the order in which the surface callables take them."""

    kind = "terrain-following (Gal-Chen & Somerville)"

    def __init__(self, bottom: Surface, top: Surface,
                 periodic: Sequence[tuple[str, tuple[float, float]]], vertical: str = "z") -> None:
        if not periodic:
            raise CoordinateError("a terrain-following map needs at least one periodic direction",
                                  fix="give the box at least one periodic direction")
        self.bottom, self.top = bottom, top
        self.periodic = tuple((str(n), (float(a), float(b))) for n, (a, b) in periodic)
        self.vertical = vertical
        self._fine: dict = {}
        self._check_order()

    # -------------------------------------------------------------------- identity
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.periodic)

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(b - a for _, (a, b) in self.periodic)

    @property
    def flat(self) -> bool:
        return self.bottom.flat and self.top.flat

    @property
    def key(self) -> tuple:
        """Hashable identity: the surfaces sampled on a fixed grid (two maps with equal samples
        are the same map for the IR, never two objects compared by id)."""
        if getattr(self, "_key", None) is None:
            g = self._grids(64)
            h = hashlib.sha1()
            for s in (self.bottom, self.top):
                h.update(np.ascontiguousarray(s(*g), dtype=float).tobytes())
            self._key = ("terrain", self.periodic, self.vertical, h.hexdigest())
        return self._key

    def _check_order(self) -> None:
        g = self._grids(FINE_MIN)
        gap = self.top(*g) - self.bottom(*g)
        if not np.all(np.isfinite(gap)):
            raise CoordinateError("terrain map: the surfaces have non-finite values",
                                  fix="give finite surface heights")
        if not float(gap.min()) > 0:
            raise CoordinateError(f"terrain map: the top surface must lie above the bottom one "
                                  f"(min(t - b) = {float(gap.min()):.3g} <= 0)",
                                  fix="give a top surface above the bottom one everywhere")

    def _grids(self, M: int | Sequence[int]) -> list[np.ndarray]:
        Ms = [M] * len(self.periodic) if isinstance(M, numbers.Integral) else list(M)
        out = []
        d = len(self.periodic)
        for i, ((_, (a, b)), m) in enumerate(zip(self.periodic, Ms, strict=True)):
            sh = [1] * d
            sh[i] = m
            out.append((a + (b - a) * np.arange(m) / m).reshape(sh))
        return out

    # -------------------------------------------------------------------- explain
    def formula(self) -> str:
        h = ", ".join(self.names)
        z = self.vertical
        return (f"{z} = b({h}) + (t({h}) - b({h})) (1 + xi)/2, xi in [-1, 1] "
                f"(b: {self.bottom.text}; t: {self.top.text})")

    def metric_text(self) -> str:
        z = self.vertical
        lines = [f"d{z} = J dxi, J = 2/(t - b)"]
        for n in self.names:
            lines.append(f"d{n}|_{z} = d{n}|_xi + (A_{n} + B_{n} xi) dxi, "
                         f"A_{n} = -J (b_{n} + t_{n})/2, B_{n} = -J (t_{n} - b_{n})/2")
        lines.append(f"dV = H dxi d{' d'.join(self.names)}, H = (t - b)/2")
        return "; ".join(lines)

    def explain(self) -> str:
        return f"{self.kind}: {self.formula()}; chain rule: {self.metric_text()}"

    def __repr__(self) -> str:
        return f"TerrainMap(bottom={self.bottom.text}, top={self.top.text})"

    # -------------------------------------------------------------------- spectral analysis
    def fine_values(self, Ms: tuple[int, ...]) -> dict[str, np.ndarray]:
        """b, t, their horizontal derivatives and the metric functions on the fine grid ``Ms``
        (derivatives spectral, from the surface coefficients; the surface tail is checked)."""
        Ms = tuple(int(m) for m in Ms)
        if Ms in self._fine:
            return self._fine[Ms]
        g = self._grids(Ms)
        out: dict[str, np.ndarray] = {}
        der: dict[str, list[np.ndarray]] = {}
        for nm, s in (("b", self.bottom), ("t", self.top)):
            v = s(*g)
            out[nm] = v
            if s.flat:
                der[nm] = [np.zeros_like(v) for _ in self.periodic]
                continue
            c = np.fft.fftn(v) / v.size
            amax = float(np.abs(c).max())
            shell = _outer_shell(c.shape)
            tail = float(np.abs(c[shell]).max()) / amax if amax > 0 else 0.0
            if tail > SURFACE_TAIL:
                raise CoordinateError(f"terrain map: the {'bottom' if nm == 'b' else 'top'} surface "
                                      f"({s.text}) is not resolved by {Ms} Fourier samples (relative tail "
                                      f"{tail:.1e} > {SURFACE_TAIL:g}): its derivatives cannot be computed "
                                      "spectrally; give a smooth periodic surface",
                                      fix="a smooth periodic surface (or more Fourier samples)")
            ds = []
            for i, ((_, (a, b)), m) in enumerate(zip(self.periodic, Ms, strict=True)):
                k = 2 * np.pi * np.fft.fftfreq(m, d=1.0 / m) / (b - a)
                if m % 2 == 0:
                    k[m // 2] = 0.0                       # Nyquist mode: derivative set to 0
                sh = [1] * len(Ms)
                sh[i] = m
                ds.append(np.real(np.fft.ifftn(1j * k.reshape(sh) * c) * v.size))
            der[nm] = ds
        b, t = out["b"], out["t"]
        J = 2.0 / (t - b)
        out["J"] = J
        out["H"] = (t - b) / 2.0
        for i, n in enumerate(self.names):
            bd, td = der["b"][i], der["t"][i]
            out[f"b_{n}"], out[f"t_{n}"] = bd, td
            out[f"A_{n}"] = -J * (bd + td) / 2.0
            out[f"B_{n}"] = -J * (td - bd) / 2.0
        self._fine[Ms] = out
        return out

    def metric_names(self) -> list[str]:
        return ["J", "H"] + [f"{p}_{n}" for n in self.names for p in ("A", "B")]

    def metric_coeffs(self, Ns: tuple[int, ...], tol: float) -> tuple[dict, list[MetricReport]]:
        """Fourier coefficients c[p], |p_d| <= 2 K_d (K_d = N_d/2 - 1), of every metric function,
        computed on a fine grid (multiple of N_d), and the check: the coefficients of each
        function sampled on the N-point grid (the resolution of the unknowns) must have a relative
        outer-shell tail <= ``tol`` (``ncc_tail_tol``), otherwise ValueError with the resolution
        to use. Returns ({name: array of shape (4K_d + 1, ...)}, reports)."""
        Ks = [n // 2 - 1 for n in Ns]
        r = [max(2, math.ceil(max(FINE_MIN, 4 * (4 * k + 1)) / n)) for k, n in zip(Ks, Ns, strict=True)]
        Ms = tuple(n * q for n, q in zip(Ns, r, strict=True))
        fine = self.fine_values(Ms)
        out, reps = {}, []
        for nm in self.metric_names():
            v = fine[nm]
            c = np.fft.fftn(v) / v.size
            # coarse samples = every r-th fine point (the N-point grid of the unknowns)
            coarse = v[tuple(slice(None, None, q) for q in r)]
            cc = np.fft.fftn(coarse) / coarse.size
            amax = float(np.abs(cc).max())
            tail = float(np.abs(cc[_outer_shell(cc.shape)]).max()) / amax if amax > 0 else 0.0
            if tail > tol and amax > 0 and not self.flat:
                raise ResolutionError(_unresolved_metric(nm, tail, tol, Ns, cc, self.names),
                                      fix="raise the resolution of the periodic directions, or smooth the surfaces")
            idx = np.ix_(*[np.arange(-2 * k, 2 * k + 1) % m for k, m in zip(Ks, Ms, strict=True)])
            a = c[idx]
            big = float(np.abs(a).max()) if a.size else 0.0
            a = np.where(np.abs(a) > METRIC_ROUNDING * big, a, 0.0)
            out[nm] = a
            reps.append(MetricReport(nm, tail, int(np.count_nonzero(a)), a.size, tol))
        return out, reps

    def z_of(self, xi: np.ndarray, grids: Sequence[np.ndarray]) -> np.ndarray:
        """Physical z at computational xi over the horizontal grids (broadcast)."""
        b, t = self.bottom(*grids), self.top(*grids)
        return b[..., None] + (t - b)[..., None] * (1.0 + np.asarray(xi)) / 2.0


def _outer_shell(shape: tuple[int, ...]) -> np.ndarray:
    """Mask of the FFT coefficients with |m_d| >= M_d/2 - 1 in some direction (the last two
    retained wavenumbers and the Nyquist mode: the Fourier analogue of the 'last 3' Chebyshev
    coefficients of the tail)."""
    mask = np.zeros(shape, dtype=bool)
    for i, m in enumerate(shape):
        f = np.abs(np.fft.fftfreq(m, d=1.0 / m))
        sel = f >= max(m // 2 - 1, 1)
        sh = [1] * len(shape)
        sh[i] = m
        mask |= np.broadcast_to(sel.reshape(sh), shape)
    return mask


def _unresolved_metric(name: str, tail: float, tol: float, Ns: tuple[int, ...], cc,
                       names: tuple[str, ...]) -> str:
    # predicted resolution: geometric decay of the 1D envelope along the direction whose outer
    # shell holds the largest coefficient (the unresolved one; in 3D not always the first)
    a = np.abs(cc)
    best = None
    for i, m in enumerate(cc.shape):
        env1 = np.moveaxis(a, i, 0).reshape(m, -1).max(axis=1)
        half = np.abs(np.fft.fftfreq(m, d=1.0 / m))
        env = np.array([env1[half == j].max() for j in range(m // 2 + 1)])
        env = env / env.max()
        shell = float(env[max(m // 2 - 1, 1):].max())
        if best is None or shell > best[0]:
            best = (shell, i, env[: m // 2], m)
    _, i, env, m = best
    k = np.arange(1, m // 2)
    slope = np.polyfit(k, np.log10(np.maximum(env[1:], 1e-300)), 1)[0] if m >= 8 else 0.0
    if slope < -1e-3:
        need = int(2 * math.ceil(math.log10(tol) / slope + 2))
        grow = f"raise N_{names[i]} to about {max(need + need % 2, Ns[i] + 2)}"
    else:
        grow = (f"raise N_{names[i]} (the coefficients do not decay along {names[i]}: is the "
                "surface smooth?)")
    grid = " x ".join(f"N_{n} = {v}" for n, v in zip(names, Ns, strict=True))
    # declared tail rounded up (a rounded-down value would be refused again)
    e = 10.0 ** math.floor(math.log10(tail))
    accept = next(q * e for q in (1, 2, 5, 10) if q * e >= tail * (1 - 1e-12))
    return (f"metric coefficient {name} of the terrain map is not resolved on the grid {grid} "
            f"(relative Fourier tail {tail:.1e} > ncc_tail_tol = {tol:g}); the coupled operator "
            f"would use a truncated metric (refused, never truncated silently). Fix: {grow}, "
            f"or lower the wall slope; or declare the accepted tail with ncc_tail_tol={accept:.0e}")


class TerrainChebyshev(Chebyshev):
    """Chebyshev T_n(xi) along the vertical of a terrain-following map (``ir.terrain``).

    ``bounds`` are the computational ones, (-1, 1): xi = -1 is the bottom surface, xi = +1 the
    top one. The physical z of a grid point depends on the horizontal position; only
    ``conatus.discretize.terrain_plan`` (``discretize(problem)``) knows how to discretize it.
    The flat-channel discretizers refuse it (they would treat xi as z)."""

    family = "Chebyshev T in a terrain-following coordinate"

    def __init__(self, coord: Coordinate, size: int, terrain: TerrainMap,
                 dealias: Any = _REQUIRED, grid: str = "floor") -> None:
        if not isinstance(terrain, TerrainMap):
            raise TypeError("terrain must be a TerrainMap")
        self.terrain = terrain
        super().__init__(coord, size, (-1.0, 1.0), dealias, grid)

    def _params(self) -> tuple:
        return super()._params() + self.terrain.key

    def coeff_space(self) -> str:
        return f"T_n(xi), n = 0..{self.size - 1}, {self.terrain.formula()}"

    def __repr__(self) -> str:
        return (f"TerrainChebyshev({self.coord.name}, size={self.size}, {self.terrain!r}, "
                f"dealias={self.dealias:g})")


def is_terrain(b: Any) -> bool:
    return isinstance(b, TerrainChebyshev)


def surface(value: Any, args: tuple[str, ...] | None = None,
            order: tuple[str, ...] = ()) -> Surface:
    """A Surface from a number or a callable whose arguments ``args`` are among the periodic
    coordinate names ``order`` (the callable is re-ordered to take all of them)."""
    if isinstance(value, Surface):
        return value
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return Surface(float(value), f"{float(value):g}")
    if callable(value):
        fn: Callable = value
        if args is not None and tuple(args) != tuple(order):
            idx = [order.index(a) for a in args]
            base = fn
            fn = lambda *g: base(*(g[i] for i in idx))  # noqa: E731
        text = getattr(value, "__name__", "f")
        text = "function" if text == "<lambda>" else text
        return Surface(fn, f"{text}({', '.join(args or order)})")
    raise TypeError(f"a surface is a number or a callable of the periodic coordinates, got {value!r}")
