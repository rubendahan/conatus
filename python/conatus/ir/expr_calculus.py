"""Expression tree, part 3: calculus nodes (public face: ``conatus.ir.expr``).

Differential operators (``Differentiate``, ``Gradient``, ``Divergence``, ``Laplacian``, ``Curl``,
``TimeDerivative``) with their integration-by-parts transposes, and integrals, traces and tau lifts
(``Integrate``, ``Interpolate``, ``Lift``).
"""
from __future__ import annotations

import math
import numbers
from typing import Any
from collections.abc import Iterable, Sequence


from ..errors import CoordinateError, OptionError
from .bases import Basis
from .coords import Coordinate, CoordinateSystem
from .domain import Domain
from .expr_algebra import add, comp, neg, pair
from .expr_base import Constant, Expr, Field, Parameter, TensorSig, Zero, _num_str, as_expr


# --------------------------------------------------------------------------- differential operators


def _jump(e: Expr, coord: Coordinate) -> Expr:
    """∫_{other coords} [e]_{left}^{right} along coord; zero if periodic or constant along coord."""
    if e.is_zero() or e.domain is None:
        return Zero(())
    basis = e.domain.basis(coord)
    if basis is None or basis.periodic:
        return Zero(())
    val = add(interp(e, coord, "right"), neg(interp(e, coord, "left")))
    if val.is_zero() or val.domain is None or not val.domain.coords:
        return val
    return integ(val, val.domain.coords)


class Differentiate(Expr):
    """Partial derivative along one coordinate."""

    _prec = 90

    def __init__(self, operand: Expr, coord: Coordinate) -> None:
        self._setup((operand,), operand.tensorsig, operand.domain, operand.dtype,
                    params=(coord.key,), coord=coord)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``diff`` (same coordinates
        and options)."""
        return diff(args[0], self.coord)

    def _own_order(self, coord: Coordinate) -> int:
        return 1 if coord == self.coord else 0

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        a = self.args[0]
        bt = _jump(pair(a, w), self.coord)
        adj, bt2 = a._tr(u, neg(diff(w, self.coord)))
        return adj, add(bt, bt2)

    def __str__(self) -> str:
        return f"d{self.coord.name}({self.args[0]})"

    def _tree_label(self) -> str:
        return self.coord.name

    def _notes(self) -> list[str]:
        b = self.domain.basis(self.coord) if self.domain else None
        if b is None:
            return []
        if b.periodic:
            return [f"diagonal in {type(b).__name__} coefficients (i k)"]
        return [f"sparse/banded in {type(b).__name__} coefficients via ultraspherical "
                "conversion (Olver–Townsend)"]


def diff(x: Any, coord: Coordinate) -> Expr:
    """Partial derivative of ``x`` along ``coord`` (zero if ``x`` does not depend on it)."""
    x = as_expr(x)
    if isinstance(x, (Constant, Parameter)) or x.domain is None or not x.domain.has(coord):
        return Zero(x.tensorsig)
    return Differentiate(x, coord)


def _cartesian(cs: CoordinateSystem, what: str) -> None:
    """grad/div/lap/curl: Cartesian, polar/cylindrical (regular bases,
    conatus.discretize.polar_plan) or spherical (S2, ball, shell: conatus.discretize.sphere_plan)."""
    if cs.curvilinear and cs.kind not in ("Polar", "Cylindrical", "S2", "Spherical"):
        raise NotImplementedError(f"{what} in {cs.kind} coordinates (regular bases)")


def _no_curvilinear_transpose(cs: CoordinateSystem, what: str) -> None:
    """The formal transposes below are the Cartesian ones (no metric, no Christoffel terms)."""
    if cs.curvilinear:
        raise NotImplementedError(f"formal transpose of {what} in {cs.kind} coordinates is not "
                                  "implemented (the Cartesian rule would be wrong: metric terms)")


class Gradient(Expr):
    """grad: prepends one index (the derivative direction)."""

    _prec = 90

    def __init__(self, operand: Expr, coordsys: CoordinateSystem) -> None:
        self._setup((operand,), (coordsys,) + operand.tensorsig, operand.domain, operand.dtype,
                    params=(coordsys.key,), coordsys=coordsys)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``grad`` (same coordinates
        and options)."""
        return grad(args[0], self.coordsys)

    def _own_order(self, coord: Coordinate) -> int:
        return 1 if coord.system == self.coordsys else 0

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        _no_curvilinear_transpose(self.coordsys, "grad")
        a = self.args[0]
        bt = add(*(_jump(pair(a, comp(w, i)), c) for i, c in enumerate(self.coordsys.coords)))
        adj, bt2 = a._tr(u, neg(div(w)))
        return adj, add(bt, bt2)

    def __str__(self) -> str:
        return f"grad({self.args[0]})"


def grad(x: Any, coordsys: CoordinateSystem | None = None) -> Expr:
    """Gradient of ``x`` (adds a first index) in ``coordsys``, by default that of ``x``'s domain."""
    x = as_expr(x)
    if coordsys is None:
        if x.domain is None:
            raise CoordinateError("grad of a constant needs an explicit coordinate system",
                                  fix="grad(c, cs) with the coordinate system cs", doc="docs/site/guide/operators.md")
        coordsys = x.domain.coordsys
    _cartesian(coordsys, "grad")
    if isinstance(x, (Constant, Parameter)) or x.domain is None:
        return Zero((coordsys,) + x.tensorsig)
    return Gradient(x, coordsys)


class Divergence(Expr):
    """div: contracts the derivative with the first index."""

    _prec = 90

    def __init__(self, operand: Expr) -> None:
        self._setup((operand,), operand.tensorsig[1:], operand.domain, operand.dtype)

    @property
    def coordsys(self) -> CoordinateSystem:
        """The coordinate system of the first tensor index of the argument."""
        return self.args[0].tensorsig[0]

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``div`` (same coordinates and
        options)."""
        return div(args[0])

    def _own_order(self, coord: Coordinate) -> int:
        return 1 if coord.system == self.coordsys else 0

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        _no_curvilinear_transpose(self.coordsys, "div")
        a = self.args[0]
        bt = add(*(_jump(pair(comp(a, i), w), c) for i, c in enumerate(self.coordsys.coords)))
        adj, bt2 = a._tr(u, neg(grad(w, self.coordsys)))
        return adj, add(bt, bt2)

    def __str__(self) -> str:
        return f"div({self.args[0]})"


def div(x: Any) -> Expr:
    """Divergence: contracts the derivative with the first index of ``x`` (rank >= 1)."""
    x = as_expr(x)
    if x.rank == 0:
        from ..errors import TensorError
        raise TensorError("div of a scalar is undefined",
                          fix="grad(f) for the gradient of a scalar, div(u) of a vector", doc="docs/site/guide/operators.md")
    _cartesian(x.tensorsig[0], "div")
    if isinstance(x, Constant) or x.domain is None:
        return Zero(x.tensorsig[1:])
    return Divergence(x)


class Laplacian(Expr):
    """Laplacian (component-wise in Cartesian coordinates)."""

    _prec = 90

    def __init__(self, operand: Expr, coordsys: CoordinateSystem) -> None:
        self._setup((operand,), operand.tensorsig, operand.domain, operand.dtype,
                    params=(coordsys.key,), coordsys=coordsys)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``lap`` (same coordinates and
        options)."""
        return lap(args[0], self.coordsys)

    def _own_order(self, coord: Coordinate) -> int:
        return 2 if coord.system == self.coordsys else 0

    def _tr(self, u: Field, w: Expr) -> tuple[Expr, Expr]:
        _no_curvilinear_transpose(self.coordsys, "lap")
        a = self.args[0]
        bt = add(*(_jump(add(pair(diff(a, c), w), neg(pair(a, diff(w, c)))), c)
                   for c in self.coordsys.coords))
        adj, bt2 = a._tr(u, lap(w, self.coordsys))
        return adj, add(bt, bt2)

    def __str__(self) -> str:
        return f"lap({self.args[0]})"


def lap(x: Any, coordsys: CoordinateSystem | None = None) -> Expr:
    """Laplacian of ``x`` in ``coordsys``, by default that of ``x``'s domain."""
    x = as_expr(x)
    if isinstance(x, (Constant, Parameter)) or x.domain is None:
        return Zero(x.tensorsig)
    coordsys = coordsys or x.domain.coordsys
    _cartesian(coordsys, "lap")
    return Laplacian(x, coordsys)


class Curl(Expr):
    """curl: 2D scalar->vector (skew-grad), 2D vector->scalar, 3D vector->vector."""

    _prec = 90

    def __init__(self, operand: Expr, coordsys: CoordinateSystem, tensorsig: TensorSig) -> None:
        self._setup((operand,), tensorsig, operand.domain, operand.dtype,
                    params=(coordsys.key,), coordsys=coordsys)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``curl`` (same coordinates
        and options)."""
        return curl(args[0], self.coordsys)

    def _own_order(self, coord: Coordinate) -> int:
        return 1 if coord.system == self.coordsys else 0

    def __str__(self) -> str:
        return f"curl({self.args[0]})"

    def _notes(self) -> list[str]:
        if self.coordsys.dim == 2 and self.args[0].rank == 0:
            return ["2D convention: curl(psi) = (d_y psi, -d_x psi)"]
        if self.coordsys.dim == 2:
            return ["2D convention: curl(u) = d_x u_y - d_y u_x (scalar)"]
        return []


def curl(x: Any, coordsys: CoordinateSystem | None = None) -> Expr:
    """Curl: of a vector in 3D (a vector), of a vector in 2D (a scalar), of a scalar in 2D (a vector)."""
    x = as_expr(x)
    if coordsys is None:
        if x.domain is None:
            raise CoordinateError("curl of a constant needs an explicit coordinate system",
                                  fix="curl(c, cs) with the coordinate system cs", doc="docs/site/guide/operators.md")
        coordsys = x.domain.coordsys
    _cartesian(coordsys, "curl")
    if coordsys.dim == 2 and x.rank == 0:
        sig: TensorSig = (coordsys,)
    elif coordsys.dim == 2 and x.rank == 1:
        sig = ()
    elif coordsys.dim == 3 and x.rank == 1:
        sig = (coordsys,)
    else:
        from ..errors import TensorError
        raise TensorError(f"curl undefined for rank {x.rank} in dim {coordsys.dim}",
                          fix="curl of a vector (2D: scalar vorticity; 3D: a vector), or of a "
                              "scalar streamfunction in 2D", doc="docs/site/guide/operators.md")
    if isinstance(x, (Constant, Parameter)) or x.domain is None:
        return Zero(sig)
    return Curl(x, coordsys, sig)


class TimeDerivative(Expr):
    """d/dt (IVP left-hand sides)."""

    _prec = 90

    def __init__(self, operand: Expr) -> None:
        self._setup((operand,), operand.tensorsig, operand.domain, operand.dtype)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``dt`` (same coordinates and
        options)."""
        return dt(args[0])

    def time_order(self) -> int:
        """Time order of the argument plus one."""
        return self.args[0].time_order() + 1

    def __str__(self) -> str:
        return f"dt({self.args[0]})"


def dt(x: Any) -> Expr:
    """Time derivative (IVP left-hand side, or sigma in a linearized EVP)."""
    x = as_expr(x)
    if isinstance(x, (Constant, Parameter)):
        return Zero(x.tensorsig)
    return TimeDerivative(x)


# --------------------------------------------------------------------------- integrals, traces, lifts


class Integrate(Expr):
    """Integral over a set of coordinates (Lebesgue measure: dx dy.. in Cartesian systems, the
    area / volume element r dr dphi (dz) in polar / cylindrical ones, over all their coordinates)."""

    def __init__(self, operand: Expr, coords: tuple[Coordinate, ...]) -> None:
        dom = operand.domain.without(*coords) if operand.domain is not None else None
        self._setup((operand,), operand.tensorsig, dom, operand.dtype,
                    params=tuple(c.key for c in coords), coords=coords)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``integ`` (same coordinates
        and options)."""
        return integ(args[0], self.coords)

    def __str__(self) -> str:
        return f"integ({self.args[0]}, {', '.join(c.name for c in self.coords)})"

    def _tree_label(self) -> str:
        return ",".join(c.name for c in self.coords)


def integ(x: Any, coords: Coordinate | Iterable[Coordinate] | None = None) -> Expr:
    """Integral of ``x`` over ``coords`` (all coordinates of its domain by default)."""
    x = as_expr(x)
    if x.domain is None:
        raise CoordinateError("integral of a constant: the integration length is ambiguous",
                              fix="multiply the constant by the length (or area) explicitly",
                              doc="docs/site/guide/operators.md")
    if coords is None:
        coords = x.domain.coords
    elif isinstance(coords, Coordinate):
        coords = (coords,)
    coords = tuple(coords)
    for c in coords:
        if not x.domain.has(c):
            raise CoordinateError(f"cannot integrate along {c.name}: the expression has no basis "
                                  "along it (length unknown)",
                                  fix=f"integrate only along the coordinates of its domain "
                                      f"({', '.join(d.name for d in x.domain.coords)}), and "
                                      f"multiply by the length of {c.name} if it is meant", doc="docs/site/guide/operators.md")
    order = {c: i for i, c in enumerate(x.domain.coordsys.coords)}
    coords = tuple(sorted(set(coords), key=lambda c: order[c]))
    if x.is_zero():
        return Zero(x.tensorsig)
    return Integrate(x, coords)


class Interpolate(Expr):
    """Trace / point evaluation along one coordinate: x(coord = position).

    ``side`` ('below' / 'above', default None) selects a one-sided trace at an interface of a
    CompoundChebyshev basis (the two pieces meet there; continuity conditions compare the two
    traces). None keeps the node identical to the plain trace (same hash, same key)."""

    def __init__(self, operand: Expr, coord: Coordinate, position: float, label: str,
                 side: str | None = None) -> None:
        self._setup((operand,), operand.tensorsig, operand.domain.without(coord), operand.dtype,
                    params=(coord.key, float(position)) + ((side,) if side else ()), coord=coord,
                    position=float(position), label=label, side=side)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``interp`` (same coordinates
        and options)."""
        return interp(args[0], self.coord, self.position, side=self.side)

    def __str__(self) -> str:
        return f"{self._wrap(self.args[0])}({self.coord.name}={self.label}{_SIDE.get(self.side, '')})"

    def _tree_label(self) -> str:
        return f"{self.coord.name}={self.label}{_SIDE.get(self.side, '')}"


_SIDE = {"below": "^-", "above": "^+"}


def interp(x: Any, coord: Coordinate, position: float | str, side: str | None = None) -> Expr:
    """Value of ``x`` at ``coord = position`` (a number, 'left' or 'right'): traces and BCs.
    ``side='below'/'above'``: one-sided trace at an interface of a CompoundChebyshev basis."""
    x = as_expr(x)
    if side not in (None, "below", "above"):
        raise CoordinateError(f"side must be None, 'below' or 'above', got {side!r}",
                              fix="side='below' or side='above' at an interface of a compound "
                                  "interval, otherwise no side", doc="docs/site/guide/operators.md")
    if x.domain is None or not x.domain.has(coord):
        return x  # constant along coord: the trace is the expression itself
    basis = x.domain.basis(coord)
    a, b = basis.bounds
    if isinstance(position, str):
        if position not in ("left", "right"):
            raise CoordinateError(f"position must be a number, 'left' or 'right', got {position!r}",
                                  fix="interp(u, y, 'left'), interp(u, y, 'right') or interp(u, y, 0.5)",
                                  doc="docs/site/guide/operators.md")
        pos = a if position == "left" else b
    else:
        pos = float(position)
        if not a <= pos <= b:
            raise CoordinateError(f"interpolation point {pos} outside {basis.bounds}",
                                  fix=f"a position in [{a:g}, {b:g}], or 'left' / 'right'", doc="docs/site/guide/operators.md")
    from .bases import CompoundChebyshev, Hermite, Laguerre
    if not math.isfinite(pos) and isinstance(basis, (Hermite, Laguerre)):
        # (a mapped Chebyshev basis does reach infinity: its trace there is the value at xi = +-1)
        raise CoordinateError(f"trace at {coord.name} = {pos} on {basis!r}: the functions decay "
                              "there; there is no boundary and no condition at infinity",
                              fix="drop the condition at infinity (the basis imposes the decay)",
                              doc="docs/site/guide/bases.md")
    if isinstance(basis, CompoundChebyshev):
        basis.piece_of(pos, side)          # validates: one-sided traces only at interfaces
    elif side is not None:
        raise CoordinateError(f"side={side!r}: one-sided traces exist only at the interfaces of a "
                              f"CompoundChebyshev basis, not on {basis!r}", fix="drop side=",
                              doc="docs/site/guide/operators.md")
    label = _num_str(pos)
    if x.is_zero():
        return Zero(x.tensorsig)
    return Interpolate(x, coord, pos, label, side)


class Lift(Expr):
    """Lift a tau variable (constant along basis.coord) into mode N+n (n < 0) of ``basis``
    (``lam = 0``) or of its ultraspherical derivative basis C^(lam) (``lam >= 1``, Chebyshev
    ``basis`` only; the result still lives in ``basis``'s domain, a polynomial of degree < N).
    ``piece``: on a CompoundChebyshev basis, the subinterval whose mode N_p + n receives it."""

    def __init__(self, operand: Expr, basis: Basis, n: int, domain: Domain, lam: int = 0,
                 piece: int | None = None) -> None:
        self._setup((operand,), operand.tensorsig, domain, operand.dtype,
                    params=(basis.key, n) + ((lam,) if lam else ())
                    + (("piece", piece) if piece is not None else ()),
                    basis=basis, n=n, lam=lam, piece=piece)

    def rebuild(self, args: Sequence[Expr]) -> Expr:
        """The same operator on the argument ``args[0]``, built by ``lift`` (same coordinates
        and options)."""
        return lift(args[0], self.basis, self.n, lam=self.lam, piece=self.piece)

    def __str__(self) -> str:
        sp_ = f", lam={self.lam}" if self.lam else ""
        pc = f", piece={self.piece}" if self.piece is not None else ""
        return f"lift({self.args[0]}, {self.basis.coord.name}, {self.n}{sp_}{pc})"

    def _tree_label(self) -> str:
        return (f"{self.basis.coord.name}, n={self.n}" + (f", C^({self.lam})" if self.lam else "")
                + (f", piece {self.piece}" if self.piece is not None else ""))

    def _notes(self) -> list[str]:
        space = (f"C^({self.lam}) (derivative basis of order {self.lam} of "
                 f"{type(self.basis).__name__}({self.basis.coord.name}))" if self.lam else
                 f"{type(self.basis).__name__}({self.basis.coord.name})")
        if self.piece is not None:
            N = self.basis.sizes[self.piece]
            return [f"tau term: polynomial of index N_{self.piece}{self.n} = {N + self.n} of "
                    f"piece {self.piece} of {space} (generalized tau method, multidomain)"]
        return [f"tau term: polynomial of index N{self.n} = {self.basis.size + self.n} of "
                f"{space} (generalized tau method)"]


def lift(x: Any, basis: Basis, n: int, lam: int = 0, piece: int | None = None) -> Expr:
    """Tau term: ``x`` times the polynomial of index N+n of ``basis`` (``lam=0``) or of the
    ultraspherical C^(lam) basis of the same size and interval (``lam >= 1``, Chebyshev only):
    lifting an order-q equation in C^(q) puts its taus in the last q rows of the ultraspherical
    equation, the classic Olver–Townsend tau (no spurious eigenvalues for dt lap^2-type
    operators, which T-lifts produce).
    On a CompoundChebyshev basis ``piece`` (required) names the subinterval: mode N_piece + n of
    that piece (each piece of an order-q equation receives q taus); on a Laguerre basis ``lam`` is
    the derivative level k of L_n^(k) exp(-y/2)."""
    x = as_expr(x)
    if not isinstance(lam, numbers.Integral) or lam < 0:
        raise OptionError(f"lift lam must be a non-negative integer, got {lam!r}",
                          fix="lam=0 (lift in the basis itself) or lam=q for an order-q equation "
                              "(ultraspherical C^(q))", doc="docs/site/guide/taus.md")
    from .bases import (AnnulusBasis, BallBasis, CompoundChebyshev, DiskBasis, Jacobi, Laguerre,
                        ShellBasis)
    # Jacobi covers Chebyshev T (C^(lam)) and the Legendre / Jacobi / Ultraspherical trial bases
    # (level lam = P^(a+lam, b+lam), discretize/jacobi_family.py)
    if lam and not isinstance(basis, (Jacobi, DiskBasis, AnnulusBasis, ShellBasis, BallBasis,
                                      CompoundChebyshev, Laguerre)):
        # polar/spherical radial bases: lam is the derivative level (C^(lam) on the annulus and the
        # shell, Jacobi P^(lam, .) on the disk and the ball, Dedalus's derivative_basis(lam));
        # spherical.md
        raise OptionError("lift into C^(lam): the basis must be Chebyshev (T), Legendre, Jacobi, "
                          "Ultraspherical, compound Chebyshev, Laguerre or a polar/spherical "
                          f"radial basis, not {basis!r}",
                          fix="lam=0 on this basis", doc="docs/site/guide/taus.md")
    if isinstance(basis, CompoundChebyshev):
        if piece is None or not isinstance(piece, numbers.Integral) or not 0 <= piece < len(basis.sizes):
            raise OptionError(f"lift on {basis!r}: piece={piece!r}",
                              why="on a compound interval each tau goes into the last modes of one "
                                  "subinterval",
                              fix=f"give piece=p, 0 <= p < {len(basis.sizes)}", doc="docs/site/guide/taus.md")
        size = basis.sizes[piece]
        piece = int(piece)
    else:
        if piece is not None:
            raise OptionError(f"lift(..., piece={piece}): pieces exist only on a CompoundChebyshev "
                              "basis", fix="drop piece=", doc="docs/site/guide/taus.md")
        size = basis.size
    if not isinstance(n, numbers.Integral) or n >= 0 or -n > size:
        raise OptionError(f"lift index n must be a negative integer with |n| <= basis size "
                          f"({size}), got {n!r}", fix="n=-1 for the last mode, -2 for the one "
                                                     "before, ...", doc="docs/site/guide/taus.md")
    if basis.periodic:
        raise OptionError(f"tau lifts are for non-periodic bases, not {basis!r}",
                          why="a periodic direction has no boundary, hence no condition and no tau",
                          fix="lift along the bounded coordinate (Chebyshev, Legendre, radial)",
                          doc="docs/site/guide/taus.md")
    if x.domain is not None and x.domain.has(basis.coord):
        raise OptionError(f"lifted variable must be constant along {basis.coord.name}",
                          why="a tau variable lives on the boundary: one value per mode of the "
                              "other directions",
                          fix=f"declare the tau on the domain without {basis.coord.name} "
                              f"(domain.without({basis.coord.name}))", doc="docs/site/guide/taus.md")
    if x.domain is None:
        dom = Domain(basis.coord.system, [basis])
    else:
        dom = x.domain.with_basis(basis)
    if x.is_zero():
        return Zero(x.tensorsig)
    return Lift(x, basis, int(n), dom, int(lam), piece)
