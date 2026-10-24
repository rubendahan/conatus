"""Geometries of the native layer: named boundaries, no bases and no resolution.

A geometry says *where* the problem lives (coordinates, periodic directions, walls and their
names). The spectral bases are chosen from a fixed table (one choice per geometry, printed by
``explain()``) and built only when a resolution is given (``Model.simulate(N=...)``,
``Model.eigenvalues(N=...)``, ``Model.discretize(N=...)``). Nothing here is guessed from data.

  Box(x=periodic(L), z=walls(a, b))   Cartesian: RealFourier (ComplexFourier with complex=True)
                                      on periodic axes, Chebyshev T on the (single) wall axis
  Box(..., z=walls(a, b, basis="sincos"))  the wall axis in sine/cosine series: the reflection
                                      subspace of RealFourier(2 N_z) on [a, 2b - a) (no tau;
                                      parities from the wall conditions)
  Box(..., z=walls(a, c, b))          compound interval: Chebyshev T on [a, c] and [c, b], glued by
                                      automatic continuity conditions (N_z = (N_1, N_2))
  Box(..., x=line(scale=s))           whole line: Hermite functions (no boundary)
  Box(..., z=half_line(scale=s))      half line z >= 0: Laguerre functions (one wall)
  Channel(Lx, Lz=None)                Box(x=periodic(Lx), y=walls(-1, 1)[, z=periodic(Lz)])
  Disk(radius), Annulus(ri, ro)       PolarCoordinates(phi, r): RealFourier x DiskBasis/AnnulusBasis
  Pipe(radius, length) (= Cylinder)   CylindricalCoordinates(phi, r, z): RealFourier x DiskBasis x ComplexFourier
  TaylorCouette(ri, ro, length)       CylindricalCoordinates(phi, r, z): RealFourier x AnnulusBasis x ComplexFourier
  Sphere(radius)                      S2Coordinates(phi, theta): RealFourier x SphereBasis
  Ball(radius), Shell(ri, ro)         SphericalCoordinates(phi, theta, r): RealFourier x SphereBasis x BallBasis/ShellBasis
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import BoundaryConditionError, InputTypeError, ResolutionError, UnsupportedError
from ..ir import (AnnulusBasis, BallBasis, CartesianCoordinates, Chebyshev, ComplexFourier,
                  CompoundChebyshev, CylindricalCoordinates, DiskBasis, Domain, Hermite, Laguerre,
                  PolarCoordinates, RealFourier, S2Coordinates, ShellBasis, SphereBasis,
                  SphericalCoordinates)
from .axes import HalfLine, Line

__all__ = ["Periodic", "Walls", "WALL_BASES", "periodic", "walls", "Geometry", "Box", "Channel", "Disk",
           "Annulus", "Pipe", "Cylinder", "TaylorCouette", "Sphere", "Ball", "Shell", "DEFAULT_DEALIAS",
           "WEIGHTED_DEALIAS"]

# Floor of the default dealiasing factor of the native layer: dealias=None uses
# max(DEFAULT_DEALIAS, (p+1)/2) for the highest polynomial degree p of the equations
# (model.compiler.dealias_factor), printed by explain() and in the Methods paragraph;
# a value given with dealias= always wins.
DEFAULT_DEALIAS = 1.5
# Floor of the default factor on a whole-line (Hermite) or half-line (Laguerre) axis for
# equations with products of the unknowns (degree p >= 2): max(WEIGHTED_DEALIAS, (p+1)/2)
# there. No factor is alias-free on these axes (the functions carry a
# Gaussian / exponential weight); 2 instead of 3/2 lowers the aliasing error of a quadratic
# product by about one decade for unit spectra and two for decaying ones.
WEIGHTED_DEALIAS = 2.0

_DOC = "docs/site/guide/geometries.md"   # public guide page (never an internal design note)


@dataclass(frozen=True)
class Periodic:
    """A periodic axis on [a, b) (``periodic(L)`` is [0, L))."""
    bounds: tuple[float, float]
    complex: bool = False

    @property
    def length(self) -> float:
        """Period b - a."""
        return self.bounds[1] - self.bounds[0]

    def explain(self) -> str:
        """What this axis is and how it is discretized (every key object has one)."""
        a, b = self.bounds
        return (f"periodic axis [{a:g}, {b:g}) of period {self.length:g}: "
                + ("complex exponentials exp(2 pi i n x / L) (ComplexFourier)" if self.complex else
                   "real cos/sin series (RealFourier; complex=True for exp(i k x))")
                + "; N points, uniform grid, no boundary condition")


WALL_BASES = ("chebyshev", "sincos")
#: Smallest number of modes along Chebyshev walls and along a line or half line (
#: N = 2 was accepted): a declared refusal threshold, not an accuracy rule.
MIN_WALL_MODES = 4


@dataclass(frozen=True)
class Walls:
    """A bounded axis [a, b] with a wall at each end; ``basis`` is the family along it
    ("chebyshev": Chebyshev T with tau conditions; "sincos": sine/cosine series).
    ``interfaces``: interior points of a compound interval (one Chebyshev series per piece)."""
    bounds: tuple[float, float]
    basis: str = "chebyshev"
    interfaces: tuple[float, ...] = ()

    def explain(self) -> str:
        """What this axis is and how it is discretized (every key object has one)."""
        a, b = self.bounds
        if self.interfaces:
            pts = (a, *self.interfaces, b)
            return (f"compound wall axis [{a:g}, {b:g}]: one Chebyshev T series per piece "
                    + ", ".join(f"[{p:g}, {q:g}]" for p, q in zip(pts, pts[1:], strict=False))
                    + " (N is a tuple, one size per piece); continuity conditions at the "
                      "interfaces written automatically; one wall condition per derivative "
                      "order at each end")
        if self.basis == "sincos":
            return (f"wall axis [{a:g}, {b:g}], sine/cosine series (no tau): each unknown is a "
                    "pure sine or cosine series, its parity read off its homogeneous wall "
                    "conditions (f = 0: sine; dz(f) = 0: cosine)")
        return (f"wall axis [{a:g}, {b:g}], Chebyshev T series (tau method): one wall condition "
                "per derivative order of each equation, at each end")


def periodic(length: float | tuple[float, float], *more: Any, complex: bool = False) -> Periodic:
    """Periodic axis of length ``L`` ([0, L)) or on ``(a, b)``; ``complex=True`` uses complex
    exponentials (ComplexFourier) instead of the real cos/sin series (RealFourier).
    ``periodic(a, b)`` (two numbers) is refused with the two spellings it may mean."""
    if more:
        shown = ", ".join(_g(v) if isinstance(v, numbers.Real) else repr(v)
                          for v in (length,) + more)
        bounds = (length,) + more
        two = len(more) == 1 and all(isinstance(v, numbers.Real) and not isinstance(v, bool)
                                     for v in bounds)
        raise InputTypeError(
            f"periodic({shown}): one length or one (a, b) pair, not {1 + len(more)} numbers",
            why="periodic(L) is the axis [0, L); the bounds of an interval are one tuple",
            fix=("periodic(L) or periodic((a, b))" if not two else
                  f"periodic({float(bounds[1])!r}) (the axis [0, L)), or "
                  f"periodic((0, {float(bounds[1])!r}))" if bounds[0] == 0 else
                  f"periodic(({float(bounds[0])!r}, {float(bounds[1])!r})) for the interval "
                  "[a, b)"),
            doc=_DOC)
    if isinstance(length, bool):
        raise InputTypeError(f"periodic({length!r}): the length is a number, not a bool",
                             fix="periodic(2.0) for [0, 2)", doc=_DOC)
    if isinstance(length, numbers.Real):
        a, b = 0.0, float(length)
    elif (hasattr(length, "__len__") and not isinstance(length, (str, bytes))
          and len(length) == 2):
        a, b = (_num(v, f"periodic({length!r})") for v in length)
    else:
        # a string, None, a complex number or a sequence of another length (audit inputs,
        # bare ValueError/TypeError from the unpacking)
        raise InputTypeError(f"periodic({length!r}): one length or one (a, b) pair",
                             fix="periodic(2.0) for [0, 2), or periodic((a, b)) with a < b",
                             doc=_DOC)
    # finite bounds (periodic(inf) would build a grid of NaN)
    if not (b > a and math.isfinite(a) and math.isfinite(b)):
        raise InputTypeError(f"periodic({length!r}): the length must be finite and positive",
                             fix="periodic(2.0) for [0, 2), or periodic((a, b)) with a < b",
                             doc=_DOC)
    from ..errors import check_flag            # complex="no" was read as True (audit inputs)
    return Periodic((a, b), check_flag(complex, "complex", "periodic()"))


def walls(a: float, b: float, *more: float, basis: str = "chebyshev") -> Walls:
    """Bounded axis [a, b] (walls at a and b); ``walls(a, c, ..., b)``: a compound interval.

    ``basis="chebyshev"`` (default): Chebyshev T, any wall conditions (tau method).
    ``basis="sincos"``: sine/cosine series sin(n pi (z-a)/L), cos(n pi (z-a)/L), n < N; each
    unknown is a pure sine or cosine series, its parity read off its homogeneous wall conditions
    (f = 0: sine; dz(f) = 0: cosine; free-slip velocity: cosine tangential, sine normal), no tau.
    Realised as the reflection subspace of RealFourier(2N) on the doubled period [a, 2b - a).

    ``walls(a, c1, ..., b)``: a compound interval, one Chebyshev T series per piece [a, c1],
    [c1, c2], ..., [ck, b] (resolution: one size per piece, N_z = (N_1, ..., N_k+1)); at each
    interface the value and the derivatives up to the order of each equation minus one are
    continuous: those conditions and their taus are written automatically and listed by
    explain(). Use it to refine a thin interior layer or a region near one wall."""
    pts = tuple(_num(v, "walls(...)") for v in (a, b) + tuple(more))
    if more:
        if basis != "chebyshev":
            raise InputTypeError(f"walls{pts} with basis={basis!r}: a compound interval is "
                                 "Chebyshev per piece", fix=f"walls{pts} (Chebyshev pieces), or "
                                 f"walls({_g(pts[0])}, {_g(pts[-1])}, basis={basis!r})", doc=_DOC)
        if any(not q > p for p, q in zip(pts, pts[1:], strict=False)):
            raise InputTypeError(f"walls{pts}: the points must increase (a < c1 < ... < b)",
                                 fix="walls(-1, 0, 1)", doc=_DOC)
        _check_finite(pts, "walls")
        return Walls((pts[0], pts[-1]), basis, pts[1:-1])
    a, b = pts
    if not b > a:
        raise InputTypeError(f"walls({a}, {b}): need a < b", fix="walls(-0.5, 0.5)", doc=_DOC)
    _check_finite(pts, "walls")
    if basis not in WALL_BASES:
        raise InputTypeError(f"walls({_g(a)}, {_g(b)}, basis={basis!r}): unknown basis",
                             why="the basis of a wall axis is a declared choice",
                             fix='basis="chebyshev" (any conditions, tau method) or basis="sincos" '
                                 "(homogeneous parity conditions: f = 0, dz(f) = 0, free-slip)",
                             doc=_DOC)
    return Walls((a, b), basis)


def _num(v: Any, what: str) -> float:
    """A length or a position given as a real number (audit inputs: ``float(v)``
    parsed ``walls("0", "1")`` and ``Disk("1")`` from text, read ``True`` as 1, and raised a bare
    TypeError for None or a complex number)."""
    import decimal

    from ..errors import _scalar_of
    v = _scalar_of(v)                 # a 0-d array is one number; a Decimal is real
    if isinstance(v, bool) or type(v).__name__ == "bool_" \
            or not isinstance(v, (numbers.Real, decimal.Decimal)):
        raise InputTypeError(f"{what}: {v!r} is not a real number",
                             why="lengths and positions must be real numbers",
                             fix=f"{what} as a float, e.g. 1.0", doc=_DOC)
    return float(v)


def _check_finite(pts: tuple[float, ...], what: str) -> None:
    """Wall positions are finite numbers (walls(0, inf) would build a
    Chebyshev grid of NaN); an unbounded axis is declared as such."""
    if not all(math.isfinite(p) for p in pts):
        raise InputTypeError(f"{what}{pts}: the walls must be at finite positions",
                             fix="line(scale=...) for the whole line, half_line(scale=...) for "
                                 "a half line", doc=_DOC)


def _check_radii(kind: str, radii: tuple[float, ...], fix: str) -> None:
    """Radii of a curved geometry: finite, 0 < inner < outer (checked when the geometry is
    declared, with the native call to write, rather than later by the basis)."""
    ok = all(math.isfinite(r) for r in radii) and radii[0] > 0 and all(
        q > p for p, q in zip(radii, radii[1:], strict=False))
    if not ok:
        from ..errors import CoordinateError
        need = "0 < inner < outer" if len(radii) == 2 else "a finite radius > 0"
        raise CoordinateError(f"{kind}{radii}: need {need}", fix=fix, doc=_DOC)


def _g(v: float) -> str:
    """A number in the shortest ``%g`` form (labels and messages)."""
    return f"{v:g}"


class Geometry:
    """Base class: coordinate names, named boundaries, the basis table and the tau recipe.

    ``recipe`` names the tau recipe of the hand-written problems of the repository that the
    compiler reproduces:
    ``"cartesian"`` lift(tau, Chebyshev, -k) (tutorial t02a), ``"regular"`` one wall, lift in
    the derivative basis lam=2 (disk/ball/pipe NS, tests/curvilinear/curvi_cases.py),
    ``"annulus"`` two walls, lam=2 lifts -1, -2 (tests/polar/test_polar_symmetry.py),
    ``"shell"`` first-order form with rvec, tau_1 lifted in lam=1 inside the gradient and tau_2 in
    lam=2 (lifting both in lam=1 gives a spurious growing tau mode from N_r ~ 48;
    conatus.problems rotating_shell_convection uses the same form).
    """

    kind = "abstract"
    recipe = "cartesian"

    def __init__(self) -> None:
        self._axes: dict[str, Any] = {}

    # --- coordinates ---------------------------------------------------------------------
    @property
    def coord_names(self) -> tuple[str, ...]:
        """Coordinate names, in the order of the resolution tuple."""
        return tuple(self._axes)

    def coordsys(self):
        """The coordinate system of the expert layer (IR) for this geometry."""
        raise NotImplementedError

    @property
    def wall_coord(self) -> str | None:
        """Name of the (single) bounded coordinate, or None (fully periodic / sphere)."""
        return None

    def n_walls(self) -> int:
        """Number of walls along the bounded coordinate (2, or 1 when an axis/centre closes it)."""
        return 0

    @property
    def sincos_coord(self) -> str | None:
        """Name of the wall coordinate discretized in sine/cosine series, or None."""
        return None

    @property
    def bounded_coord(self) -> str | None:
        """Name of the non-periodic coordinate (walls, half line or whole line), or None."""
        return self.wall_coord

    @property
    def interfaces(self) -> tuple[float, ...]:
        """Interior interfaces of a compound wall axis (empty otherwise)."""
        return ()

    def boundaries(self) -> dict[str, list[tuple[str, str]]]:
        """Boundary name -> list of (coordinate name, 'left' | 'right')."""
        return {}

    def boundary(self, name: str) -> list[tuple[str, str]]:
        """(coordinate, side) pairs of the boundary ``name``; an unknown name is refused with
        the names of this geometry."""
        table = self.boundaries()
        if name in table:
            return table[name]
        known = ", ".join(repr(k) for k in table) or "none (no boundary in this geometry)"
        raise BoundaryConditionError(
            f"unknown boundary {name!r} for {self!r}",
            where=f"at({name!r}, ...)",
            why="boundaries are named by the geometry; a condition on an unnamed place cannot be "
                "attached to a trace",
            fix=f"use one of: {known}", doc=_DOC)

    # --- resolution ----------------------------------------------------------------------
    def resolution(self, N: Any) -> dict[str, int]:
        """Resolution per coordinate from a tuple (coordinate order), a dict by name, or an int
        (only for a geometry with one coordinate)."""
        names = self.coord_names
        if isinstance(N, Mapping):
            missing = [c for c in names if c not in N]
            extra = [c for c in N if c not in names]
            if missing or extra:
                raise ResolutionError(
                    f"resolution {dict(N)!r} does not match the coordinates {names}",
                    where="N=" + repr(dict(N)),
                    why="every coordinate needs a declared number of modes",
                    fix="N={" + ", ".join(f"'{c}': ..." for c in names) + "}", doc=_DOC)
            out = {c: N[c] for c in names}
        elif isinstance(N, numbers.Integral):
            if len(names) != 1:
                raise ResolutionError(
                    f"N={N} is one number for the {len(names)} coordinates {names}",
                    why="each coordinate needs its own number of modes",
                    fix=f"N=({', '.join('...' for _ in names)}) in the order {names}, or a dict "
                        "by coordinate name", doc=_DOC)
            out = {names[0]: N}
        elif isinstance(N, numbers.Number) or isinstance(N, str) or not hasattr(N, "__iter__"):
            # U1: N=32.0 was "'float' object is not iterable"
            raise ResolutionError(
                f"N={N!r} is not an integer number of modes",
                why="a resolution is an integer per coordinate",
                fix=(f"N={int(N)}" if isinstance(N, numbers.Real) and float(N).is_integer()
                     and len(names) == 1 else
                     f"N=({', '.join('...' for _ in names)}) (integers, in the order {names})"),
                doc=_DOC)
        else:
            N = tuple(N)
            if len(N) != len(names):
                raise ResolutionError(
                    f"N={N} has {len(N)} entries for the coordinates {names}",
                    fix=f"one entry per coordinate, in the order {names}", doc=_DOC)
            out = dict(zip(names, N, strict=True))
        for c, n in out.items():
            if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
                raise ResolutionError(f"N[{c!r}] = {n!r} is not a positive integer",
                                      fix="give an integer number of modes"
                                          + (f": {int(n)}" if isinstance(n, numbers.Real)
                                             and float(n).is_integer() and n >= 1 else ""),
                                      doc=_DOC)
            low, why = self._min_size(c)
            if n < low:
                raise ResolutionError(f"N[{c!r}] = {n} is below the smallest size of this axis "
                                      f"({low})", why=why,
                                      fix=f"N[{c!r}] >= {low} (a resolution study starts well "
                                          "above it)", doc=_DOC)
        return {c: int(n) for c, n in out.items()}

    def _min_size(self, c: str) -> tuple[int, str]:
        """(smallest accepted size, why) along coordinate ``c`` (N = 2 on
        Chebyshev walls is refused). Default: 1 (no rule)."""
        return 1, ""

    def _weighted_coords(self) -> tuple[str, ...]:
        """Coordinates on a whole line (Hermite) or half line (Laguerre): products of the
        unknowns are not alias-free there at any factor (default factor: WEIGHTED_DEALIAS)."""
        return tuple(c for c, ax in self._axes.items() if isinstance(ax, (Line, HalfLine)))

    def dealias_map(self, dealias: Any) -> dict[str, float]:
        """Dealias factor per coordinate: None -> the declared default, a number, or a dict."""
        names = self.coord_names
        if dealias is None:
            return {c: DEFAULT_DEALIAS for c in names}
        if isinstance(dealias, Mapping):
            missing = [c for c in names if c not in dealias]
            # a key that names no coordinate (a typo) was ignored (review finding 29)
            stray = [c for c in dealias if c not in names]
            if stray:
                raise ResolutionError(f"dealias={dict(dealias)!r}: {stray} not coordinate(s) of the "
                                      f"geometry {tuple(names)}",
                                      fix="one entry per coordinate: " + ", ".join(names), doc=_DOC)
            if missing:
                raise ResolutionError(f"dealias={dict(dealias)!r} misses {missing}",
                                      fix="give every coordinate, or one number for all", doc=_DOC)
            return {c: dealias[c] for c in names}
        return {c: dealias for c in names}

    def build(self, N: Mapping[str, int], dealias: Mapping[str, float]):
        """(coordinate system, {name: Basis}, full Domain) at this resolution."""
        cs = self.coordsys()
        bases = {c: self._basis(cs[c], N[c], dealias[c]) for c in self.coord_names}
        return cs, bases, Domain(cs, list(bases.values()))

    def _basis(self, coord, n: int, dealias: float):
        raise NotImplementedError

    def basis_table(self) -> list[str]:
        """One line per coordinate: the basis family chosen (for explain())."""
        raise NotImplementedError

    def explain(self) -> str:
        """The geometry, its basis table and its named boundaries."""
        lines = [f"Geometry: {self!r} ({self.kind})", "  bases (fixed per geometry, sizes given at "
                 "discretization):"]
        lines += ["    " + s for s in self.basis_table()]
        b = self.boundaries()
        if b:
            lines.append("  named boundaries: " + "; ".join(
                f"{k} = " + ", ".join(f"{c} {side}" for c, side in v) for k, v in b.items()))
        else:
            lines.append("  no boundary")
        return "\n".join(lines)


class Box(Geometry):
    """Cartesian box: ``Box(x=periodic(2.0), z=walls(-0.5, 0.5))``; 1 to 3 axes in keyword order.

    Boundaries of the wall axis ``c`` on [a, b]: ``"c=a"``, ``"c=b"``, ``"walls"`` (both), and
    ``"bottom"`` / ``"top"`` (the wall axis is the last bounded Cartesian coordinate).
    At most one axis may have walls (the Cartesian discretization has one Chebyshev direction).
    """

    kind = "Cartesian box"

    def __init__(self, **axes: Periodic | Walls | Line | HalfLine) -> None:
        super().__init__()
        if not 1 <= len(axes) <= 3:
            raise InputTypeError(f"Box needs 1 to 3 axes, got {len(axes)}",
                                 fix="Box(x=periodic(2.0), z=walls(-0.5, 0.5))", doc=_DOC)
        for k, v in axes.items():
            if not isinstance(v, (Periodic, Walls, Line, HalfLine)):
                raise InputTypeError(f"Box axis {k}={v!r}: expected periodic(...), walls(...), "
                                     "line(...) or half_line(...)",
                                     fix=f"{k}=periodic(L) or {k}=walls(a, b)", doc=_DOC)
        bounded = [k for k, v in axes.items() if not isinstance(v, Periodic)]
        if len(bounded) > 1:
            raise UnsupportedError(
                f"Box with non-periodic axes {bounded}: one non-periodic axis at most",
                why="the Cartesian discretization couples one non-periodic direction (walls, half "
                    "line or line) per Fourier group; two such directions are not implemented",
                fix="keep one walls(...)/line(...)/half_line(...) axis, the others periodic(...)",
                doc=_DOC)
        self._axes = dict(axes)

    def coordsys(self):
        """Cartesian coordinates in the declared order."""
        return CartesianCoordinates(*self.coord_names)

    @property
    def wall_coord(self) -> str | None:
        """The coordinate carrying walls: a walls(...) axis, or a half line (one wall)."""
        return next((k for k, v in self._axes.items() if isinstance(v, (Walls, HalfLine))), None)

    @property
    def bounded_coord(self) -> str | None:
        """The first non-periodic coordinate, or None."""
        return next((k for k, v in self._axes.items() if not isinstance(v, Periodic)), None)

    def n_walls(self) -> int:
        """Number of walls: 0 (all periodic), 1 (half line) or 2."""
        c = self.wall_coord
        if c is None:
            return 0
        return 1 if isinstance(self._axes[c], HalfLine) else 2

    @property
    def sincos_coord(self) -> str | None:
        """The wall coordinate when it uses sine/cosine series, else None."""
        c = self.wall_coord
        ax = self._axes[c] if c is not None else None
        return c if isinstance(ax, Walls) and ax.basis == "sincos" else None

    @property
    def interfaces(self) -> tuple[float, ...]:
        """Interface positions of a compound wall axis (empty otherwise)."""
        c = self.wall_coord
        ax = self._axes[c] if c is not None else None
        return ax.interfaces if isinstance(ax, Walls) else ()

    def boundaries(self) -> dict[str, list[tuple[str, str]]]:
        """Named boundaries -> [(coordinate, side)]: bottom/top, walls, and the positions such
        as ``z=0``; none without a wall axis."""
        c = self.wall_coord
        if c is None:
            return {}
        if isinstance(self._axes[c], HalfLine):
            a = self._axes[c].start
            return {"wall": [(c, "left")], "bottom": [(c, "left")], f"{c}={_g(a)}": [(c, "left")]}
        a, b = self._axes[c].bounds
        return {"bottom": [(c, "left")], "top": [(c, "right")],
                f"{c}={_g(a)}": [(c, "left")], f"{c}={_g(b)}": [(c, "right")],
                "walls": [(c, "left"), (c, "right")]}

    def resolution(self, N: Any) -> dict[str, Any]:
        """As Geometry.resolution; a compound wall axis takes one size per piece (a tuple)."""
        c = self.wall_coord
        pieces = len(self.interfaces) + 1
        if pieces == 1:
            return super().resolution(N)
        names = self.coord_names
        if (len(names) == 1 and not isinstance(N, (Mapping, numbers.Integral))
                and all(isinstance(n, numbers.Integral) for n in tuple(N))):
            N = (tuple(N),)                       # 1D compound axis: N=(N_1, N_2) is per piece
        if isinstance(N, Mapping):
            given = N.get(c)
        elif isinstance(N, numbers.Integral) and len(names) == 1:
            given = N                             # refused below with what to write
        elif isinstance(N, numbers.Integral) or len(tuple(N)) != len(names):
            return super().resolution(N)          # its error says what to write
        else:
            given = tuple(N)[names.index(c)]
        ax = self._axes[c]
        pts = (ax.bounds[0],) + ax.interfaces + (ax.bounds[1],)
        if isinstance(given, numbers.Integral) or not isinstance(given, (tuple, list)) or \
                len(given) != pieces or not all(isinstance(n, numbers.Integral) and n >= 2 for n in given):
            raise ResolutionError(
                f"N[{c!r}] = {given!r} for the compound interval {c} in {pts}",
                why=f"each of its {pieces} pieces has its own number of Chebyshev modes (a numerical "
                    "choice per piece: refine the piece that holds a thin layer)",
                fix=f"N[{c!r}] = ({', '.join(['...'] * pieces)}), one integer >= 2 per piece, "
                    "e.g. (" + ", ".join(["24"] * pieces) + ")", doc=_DOC)
        probe = dict(N) if isinstance(N, Mapping) else list(N)
        if isinstance(N, Mapping):
            probe[c] = 1
        else:
            probe[names.index(c)] = 1
        out = super().resolution(probe if isinstance(N, Mapping) else tuple(probe))
        out[c] = tuple(int(n) for n in given)
        return out

    def _min_size(self, c: str) -> tuple[int, str]:
        ax = self._axes.get(c)
        if isinstance(ax, Walls) and ax.basis == "chebyshev" and not ax.interfaces:
            return MIN_WALL_MODES, (f"{MIN_WALL_MODES} Chebyshev modes is the smallest "
                                    "polynomial space that holds two wall conditions of a "
                                    "second-order equation and still has interior modes; below "
                                    "it the tau rows are the whole system")
        if isinstance(ax, (Line, HalfLine)):
            return MIN_WALL_MODES, (f"{MIN_WALL_MODES} modes is the smallest Hermite/Laguerre "
                                    "series this geometry accepts (fewer have no decay to measure)")
        return 1, ""

    def _basis(self, coord, n, dealias):
        ax = self._axes[coord.name]
        if isinstance(ax, Periodic):
            B = ComplexFourier if ax.complex else RealFourier
            return B(coord, n, ax.bounds, dealias=dealias)
        if isinstance(ax, Line):
            return Hermite(coord, n, center=ax.center, scale=ax.scale, dealias=dealias)
        if isinstance(ax, HalfLine):
            return Laguerre(coord, n, start=ax.start, scale=ax.scale, dealias=dealias)
        if ax.interfaces:
            return CompoundChebyshev(coord, n, (ax.bounds[0],) + ax.interfaces + (ax.bounds[1],),
                                     dealias=dealias)
        if ax.basis == "sincos":
            # N wavenumbers n pi / L on [a, b] = the reflection subspace (about a) of the 2N-mode
            # real Fourier series on the doubled period [a, 2b - a); the dealias factor applies
            # to that doubled grid (compiler: SymmetrySubspace(Reflection(z, about=a), ...))
            a, b = ax.bounds
            return RealFourier(coord, 2 * n, (a, 2 * b - a), dealias=dealias)
        return Chebyshev(coord, n, ax.bounds, dealias=dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the box (explain())."""
        out = []
        for k, v in self._axes.items():
            if isinstance(v, Periodic):
                fam = "ComplexFourier" if v.complex else "RealFourier (cos/sin)"
                out.append(f"{k}: periodic on [{_g(v.bounds[0])}, {_g(v.bounds[1])}) -> {fam}(N_{k})")
            elif isinstance(v, Line):
                out.append(f"{k}: whole line -> Hermite functions psi_n(({k} - {_g(v.center)})/"
                           f"{_g(v.scale)}) (N_{k}; declared scale {_g(v.scale)}, Gaussian decay "
                           "built in, no boundary)")
            elif isinstance(v, HalfLine):
                out.append(f"{k}: half line {k} >= {_g(v.start)} -> Laguerre functions of ({k} - "
                           f"{_g(v.start)})/{_g(v.scale)} (N_{k}; declared scale {_g(v.scale)}, "
                           f"exponential decay built in, wall at {k} = {_g(v.start)})")
            elif v.interfaces:
                pts = (v.bounds[0],) + v.interfaces + (v.bounds[1],)
                pieces = ", ".join(f"[{_g(p)}, {_g(q)}]" for p, q in zip(pts, pts[1:], strict=False))
                out.append(f"{k}: compound interval {pieces} -> Chebyshev T per piece (N_{k} = one "
                           "size per piece), continuity of value and derivatives at "
                           + ", ".join(f"{k} = {_g(c)}" for c in v.interfaces) + " (automatic)")
            elif v.basis == "sincos":
                a, b = v.bounds
                out.append(f"{k}: walls at {_g(a)}, {_g(b)} -> sine/cosine bases (N_{k} wavenumbers "
                           f"n pi/L, n < N_{k}, L = {_g(b - a)}), realised as the reflection subspace "
                           f"of RealFourier(2 N_{k}) on [{_g(a)}, {_g(2 * b - a)}); no tau")
            else:
                out.append(f"{k}: walls at {_g(v.bounds[0])}, {_g(v.bounds[1])} -> Chebyshev T(N_{k})")
        return out

    def __repr__(self) -> str:
        parts = []
        for k, v in self._axes.items():
            if isinstance(v, Periodic):
                parts.append(f"{k}=periodic(({_g(v.bounds[0])}, {_g(v.bounds[1])})"
                             + (", complex=True" if v.complex else "") + ")")
            elif isinstance(v, Line):
                parts.append(f"{k}=line({_g(v.center)}, scale={_g(v.scale)})")
            elif isinstance(v, HalfLine):
                parts.append(f"{k}=half_line({_g(v.start)}, scale={_g(v.scale)})")
            elif v.interfaces:
                pts = (v.bounds[0],) + v.interfaces + (v.bounds[1],)
                parts.append(f"{k}=walls({', '.join(_g(p) for p in pts)})")
            else:
                parts.append(f"{k}=walls({_g(v.bounds[0])}, {_g(v.bounds[1])}"
                             + (f", basis={v.basis!r}" if v.basis != "chebyshev" else "") + ")")
        return "Box(" + ", ".join(parts) + ")"


def Channel(Lx: float = 2 * math.pi, Lz: float | None = None, *, h: float = 1.0,
            complex: bool = False) -> Box:
    """Plane channel: x periodic of length Lx, walls at y = -h and y = +h, and z periodic of
    length Lz in 3D (coordinates in the order x, y, z; boundaries "bottom"/"top"/"walls")."""
    axes: dict[str, Any] = {"x": periodic(Lx, complex=complex), "y": walls(-h, h)}
    if Lz is not None:
        axes["z"] = periodic(Lz, complex=complex)
    box = Box(**axes)
    box.kind = "plane channel"
    return box


class _Radial(Geometry):
    """Common part of the polar and spherical geometries (radius coordinate ``r``)."""

    _sides: dict[str, str] = {}

    @property
    def wall_coord(self) -> str | None:
        return "r"

    def boundaries(self) -> dict[str, list[tuple[str, str]]]:
        out = {k: [("r", s)] for k, s in self._sides.items()}
        if len(self._sides) == 2:
            out["walls"] = [("r", "left"), ("r", "right")]
        for _k, s in self._sides.items():
            out[f"r={_g(self._radii[0 if s == 'left' else 1])}"] = [("r", s)]
        return out

    def n_walls(self) -> int:
        return len(self._sides)


class Disk(_Radial):
    """Disk r <= R: PolarCoordinates(phi, r), RealFourier(phi) x DiskBasis(r) (regular at the
    axis, no axis condition). Boundary: "wall" (r = R)."""

    kind, recipe = "disk", "regular"
    _sides = {"wall": "right"}

    def __init__(self, radius: float = 1.0) -> None:
        super().__init__()
        self._axes = {"phi": None, "r": None}
        radius = _num(radius, "Disk(radius)")
        _check_radii("Disk", (float(radius),), "cn.Disk(1.0)")
        self._radii = (0.0, float(radius))

    def coordsys(self):
        """Polar coordinates (phi, r)."""
        return PolarCoordinates("phi", "r")

    def _basis(self, coord, n, dealias):
        if coord.name == "phi":
            return RealFourier(coord, n, (0, 2 * math.pi), dealias=dealias)
        return DiskBasis(coord, n, self._radii[1], dealias=dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the disk (explain())."""
        return ["phi: periodic on [0, 2 pi) -> RealFourier(N_phi)",
                f"r: [0, {_g(self._radii[1])}] -> DiskBasis(N_r) (regular at r = 0)"]

    def __repr__(self) -> str:
        return f"Disk(radius={_g(self._radii[1])})"


class Annulus(Disk):
    """Annulus ri <= r <= ro: RealFourier(phi) x AnnulusBasis(r). Boundaries "inner", "outer",
    "walls"."""

    kind, recipe = "annulus", "annulus"
    _sides = {"inner": "left", "outer": "right"}

    def __init__(self, inner: float, outer: float) -> None:
        Geometry.__init__(self)
        self._axes = {"phi": None, "r": None}
        inner, outer = _num(inner, "Annulus(inner)"), _num(outer, "Annulus(outer)")
        _check_radii("Annulus", (float(inner), float(outer)),
                     "cn.Annulus(0.5, 1.0); cn.Disk(radius) for a full disk (no inner wall)")
        self._radii = (float(inner), float(outer))

    def _basis(self, coord, n, dealias):
        if coord.name == "phi":
            return RealFourier(coord, n, (0, 2 * math.pi), dealias=dealias)
        return AnnulusBasis(coord, n, self._radii, dealias=dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the annulus (explain())."""
        return ["phi: periodic on [0, 2 pi) -> RealFourier(N_phi)",
                f"r: [{_g(self._radii[0])}, {_g(self._radii[1])}] -> AnnulusBasis(N_r) (Chebyshev T)"]

    def __repr__(self) -> str:
        return f"Annulus(inner={_g(self._radii[0])}, outer={_g(self._radii[1])})"


class Pipe(Disk):
    """Pipe (cylinder) r <= R, z periodic of length L: CylindricalCoordinates(phi, r, z),
    RealFourier(phi) x DiskBasis(r) x ComplexFourier(z). Boundary "wall"."""

    kind, recipe = "pipe", "regular"

    def __init__(self, radius: float = 1.0, length: float = 2 * math.pi) -> None:
        Geometry.__init__(self)
        self._axes = {"phi": None, "r": None, "z": None}
        radius, length = _num(radius, "Pipe(radius)"), _num(length, "Pipe(length)")
        _check_radii("Pipe", (float(radius),), "cn.Pipe(radius=1.0, length=2*np.pi)")
        _check_finite((0.0, float(length)), "Pipe length")
        if not float(length) > 0:
            raise InputTypeError(f"Pipe(length={length!r}): the length must be positive",
                                 fix="cn.Pipe(radius=1.0, length=2*np.pi)", doc=_DOC)
        self._radii = (0.0, float(radius))
        self.length = float(length)

    def coordsys(self):
        """Cylindrical coordinates (phi, r, z)."""
        return CylindricalCoordinates("phi", "r", "z")

    def _basis(self, coord, n, dealias):
        if coord.name == "z":
            # the polar compiler stores the axis of a cylinder as ComplexFourier (a real field keeps
            # m >= 0 and every k_z; polar_eval.PolarSpace); RealFourier in z is not implemented
            return ComplexFourier(coord, n, (0, self.length), dealias=dealias)
        return super()._basis(coord, n, dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the pipe (explain())."""
        return super().basis_table() + [f"z: periodic on [0, {_g(self.length)}) -> ComplexFourier(N_z)"]

    def __repr__(self) -> str:
        return f"Pipe(radius={_g(self._radii[1])}, length={_g(self.length)})"


Cylinder = Pipe


class TaylorCouette(Annulus):
    """Annular cylinder ri <= r <= ro, z periodic of length L (Taylor–Couette geometry):
    CylindricalCoordinates(phi, r, z), RealFourier(phi) x AnnulusBasis(r) x ComplexFourier(z).
    Boundaries "inner", "outer", "walls" (the two cylinders). Same annulus recipe as
    :class:`Annulus` (lam=2 lifts at modes -1 and -2, plus the divergence tau
    dot(rvec, lift(tau_u2, lam=1)) of an incompressible flow, declared by explain()), with the z
    axis of :class:`Pipe`."""

    kind, recipe = "Taylor-Couette annular cylinder", "annulus"

    def __init__(self, inner: float, outer: float, length: float = 2 * math.pi) -> None:
        inner, outer = _num(inner, "TaylorCouette(inner)"), _num(outer, "TaylorCouette(outer)")
        length = _num(length, "TaylorCouette(length)")
        # the axial period is checked like Pipe's (audit inputs: length=0 or nan was
        # accepted and gave a grid of NaN at the first resolution)
        _check_finite((0.0, length), "TaylorCouette length")
        if not length > 0:
            raise InputTypeError(f"TaylorCouette(length={length!r}): the axial period must be > 0",
                                 fix="cn.TaylorCouette(inner, outer, length=2*np.pi)", doc=_DOC)
        if not 0 < float(inner) < float(outer):
            from ..errors import CoordinateError
            raise CoordinateError(f"TaylorCouette(inner={inner}, outer={outer}): need 0 < inner < outer",
                                  fix="cn.Pipe(radius, length) for a full cylinder (no inner wall)",
                                  doc=_DOC)
        Annulus.__init__(self, inner, outer)
        self._axes = {"phi": None, "r": None, "z": None}
        self.length = float(length)

    def coordsys(self):
        """Cylindrical coordinates (phi, r, z)."""
        return CylindricalCoordinates("phi", "r", "z")

    def _basis(self, coord, n, dealias):
        if coord.name == "z":                  # as Pipe: the polar compiler stores z as ComplexFourier
            return ComplexFourier(coord, n, (0, self.length), dealias=dealias)
        return super()._basis(coord, n, dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the Taylor-Couette annulus (explain())."""
        return super().basis_table() + [f"z: periodic on [0, {_g(self.length)}) -> ComplexFourier(N_z)"]

    @property
    def eta(self) -> float:
        """The radius ratio eta = ri / ro (the name of ``taylor_couette(eta=...)``)."""
        return self._radii[0] / self._radii[1]

    @property
    def radius_ratio(self) -> float:
        """Former name of :attr:`eta` (deprecated since v0.1, removed in 0.3)."""
        import warnings
        warnings.warn("TaylorCouette.radius_ratio is deprecated since v0.1, read .eta (same value; "
                      "removed in 0.3)", DeprecationWarning, stacklevel=2)
        return self.eta

    def __repr__(self) -> str:
        return (f"TaylorCouette(inner={_g(self._radii[0])}, outer={_g(self._radii[1])}, "
                f"length={_g(self.length)})")


class Sphere(Geometry):
    """Sphere surface of radius R: S2Coordinates(phi, theta), RealFourier(phi) x SphereBasis(theta).
    N_theta = Lmax + 1, the SphereBasis size. No boundary."""

    kind, recipe = "sphere surface", "none"

    def __init__(self, radius: float = 1.0) -> None:
        super().__init__()
        self._axes = {"phi": None, "theta": None}
        radius = _num(radius, "Sphere(radius)")
        _check_radii("Sphere", (float(radius),), "cn.Sphere(1.0)")
        self.radius = float(radius)

    def coordsys(self):
        """Sphere-surface coordinates (phi, theta)."""
        return S2Coordinates("phi", "theta")

    def _basis(self, coord, n, dealias):
        if coord.name == "phi":
            return RealFourier(coord, n, (0, 2 * math.pi), dealias=dealias)
        return SphereBasis(coord, n, radius=self.radius, dealias=dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the sphere surface (explain())."""
        return ["phi: periodic on [0, 2 pi) -> RealFourier(N_phi)",
                "theta: [0, pi] -> SphereBasis(N_theta) (N_theta = Lmax + 1)"]

    def __repr__(self) -> str:
        return f"Sphere(radius={_g(self.radius)})"


class Ball(_Radial):
    """Ball r <= R: SphericalCoordinates(phi, theta, r), RealFourier x SphereBasis x BallBasis
    (regular at the centre). Boundary "wall"."""

    kind, recipe = "ball", "regular"
    _sides = {"wall": "right"}

    def __init__(self, radius: float = 1.0) -> None:
        super().__init__()
        self._axes = {"phi": None, "theta": None, "r": None}
        radius = _num(radius, "Ball(radius)")
        _check_radii("Ball", (float(radius),), "cn.Ball(1.0)")
        self._radii = (0.0, float(radius))

    def coordsys(self):
        """Spherical coordinates (phi, theta, r)."""
        return SphericalCoordinates("phi", "theta", "r")

    def _radial(self, coord, n, dealias):
        return BallBasis(coord, n, self._radii[1], dealias=dealias)

    def _basis(self, coord, n, dealias):
        if coord.name == "phi":
            return RealFourier(coord, n, (0, 2 * math.pi), dealias=dealias)
        if coord.name == "theta":
            return SphereBasis(coord, n, dealias=dealias)
        return self._radial(coord, n, dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the ball (explain())."""
        return ["phi: periodic on [0, 2 pi) -> RealFourier(N_phi)",
                "theta: [0, pi] -> SphereBasis(N_theta) (N_theta = Lmax + 1)",
                f"r: [0, {_g(self._radii[1])}] -> BallBasis(N_r) (regular at r = 0)"]

    def __repr__(self) -> str:
        return f"Ball(radius={_g(self._radii[1])})"


class Shell(Ball):
    """Spherical shell ri <= r <= ro: RealFourier x SphereBasis x ShellBasis. Boundaries "inner",
    "outer", "walls"."""

    kind, recipe = "spherical shell", "shell"
    _sides = {"inner": "left", "outer": "right"}

    def __init__(self, inner: float, outer: float) -> None:
        Geometry.__init__(self)
        self._axes = {"phi": None, "theta": None, "r": None}
        inner, outer = _num(inner, "Shell(inner)"), _num(outer, "Shell(outer)")
        _check_radii("Shell", (float(inner), float(outer)),
                     "cn.Shell(0.35, 1.0); cn.Ball(radius) for a full ball (no inner wall)")
        self._radii = (float(inner), float(outer))

    def _radial(self, coord, n, dealias):
        return ShellBasis(coord, n, self._radii, dealias=dealias)

    def basis_table(self) -> list[str]:
        """Coordinate -> basis lines of the shell (explain())."""
        return ["phi: periodic on [0, 2 pi) -> RealFourier(N_phi)",
                "theta: [0, pi] -> SphereBasis(N_theta) (N_theta = Lmax + 1)",
                f"r: [{_g(self._radii[0])}, {_g(self._radii[1])}] -> ShellBasis(N_r) (Chebyshev T)"]

    def __repr__(self) -> str:
        return f"Shell(inner={_g(self._radii[0])}, outer={_g(self._radii[1])})"
