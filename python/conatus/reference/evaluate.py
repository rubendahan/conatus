"""NumPy reference evaluator (embryo of the reference backend, ROADMAP strategy 3).

Evaluates an IR expression on the collocation grid of its domain, with dense,
obviously-correct operations (FFT for Fourier, Vandermonde solves for
Chebyshev). It is an **oracle for tests**, not a solver: no dealiasing, no
sparsity, O(N^3) Chebyshev transforms. Supported bases: RealFourier,
ComplexFourier, Chebyshev (also mapped), and the bases of ``discretize.families`` (Legendre,
Jacobi, Gegenbauer, Hermite, Laguerre, compound Chebyshev) through their own N-point transforms.

Array convention: tensor component axes first, then one axis per coordinate of
the coordinate system, of length N if the domain has a basis there and 1
otherwise (so reduced domains broadcast naturally).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial import chebyshev as C

from ..ir import expr as E
from ..ir.bases import Basis, Chebyshev, ComplexFourier, RealFourier
from ..ir.coords import Coordinate
from ..ir.domain import Domain
from ..errors import EquationError, OptionError
from .._repr import compact_repr

__all__ = ["FieldData", "ReferenceEvaluator", "evaluate"]

LAYOUTS = ("grid", "coeff")


@dataclass(frozen=True)
class FieldData:
    """Numerical values of a Field, with an explicit layout (never an implicit state)."""
    __repr__ = compact_repr   # one line, arrays by shape

    field: E.Field
    data: np.ndarray
    layout: str = "grid"

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise OptionError(f"layout must be one of {LAYOUTS}", fix="layout='grid' or layout='coeff'")

    def explain(self) -> str:
        """Field name, array shape and dtype, declared layout."""
        return (f"FieldData of '{self.field.name}': array {tuple(np.shape(self.data))} "
                f"{np.asarray(self.data).dtype}, layout '{self.layout}' (declared, never inferred)")


def _is_fourier(b: Basis) -> bool:
    return isinstance(b, (RealFourier, ComplexFourier))


def _fam(b: Basis):
    from ..discretize.families import family_of
    return family_of(b)


def _nodes(b: Basis) -> np.ndarray:
    a, L, N = b.bounds[0], b.length, b.size
    fam = _fam(b)
    if fam is not None:             # Hermite / Laguerre / compound: their N-point Gauss grid
        return fam.grid(dealias=False)
    if _is_fourier(b):
        return a + L * np.arange(N) / N
    if isinstance(b, Chebyshev):
        t = -np.cos(np.pi * (np.arange(N) + 0.5) / N)  # Gauss–Chebyshev, ascending
        if getattr(b, "map", None) is not None:        # mapped basis: physical y = m(xi)
            return b.map.y(t)
        return a + (t + 1) * L / 2
    raise NotImplementedError(f"reference evaluator: {type(b).__name__} not supported yet")


def _tnodes(b: Basis) -> np.ndarray:
    """Chebyshev nodes in the computational coordinate t in [-1, 1] (xi for a mapped basis)."""
    if getattr(b, "map", None) is not None:
        return -np.cos(np.pi * (np.arange(b.size) + 0.5) / b.size)
    return 2 * (_nodes(b) - b.bounds[0]) / b.length - 1


class ReferenceEvaluator:
    """Grid evaluator over a global domain (union of all node domains)."""

    def __init__(self, domain: Domain) -> None:
        self.domain = domain
        self.coords = domain.coordsys.coords
        self.ndim = len(self.coords)

    def explain(self) -> str:
        """The basis of each coordinate and how it is evaluated."""
        parts = []
        for c in self.coords:
            b = self.domain.basis(c) if self.domain.has(c) else None
            parts.append(f"{c.name}: " + ("constant" if b is None else
                                          f"{type(b).__name__}(N={b.size}) on {b.bounds}"))
        return ("ReferenceEvaluator (dense NumPy oracle, test use only; Fourier by FFT, Chebyshev "
                "by Vandermonde on Gauss–Chebyshev nodes): " + "; ".join(parts))

    # grids ------------------------------------------------------------------
    def grid(self, coord: Coordinate) -> np.ndarray:
        """Nodes of ``coord``, shaped to broadcast along its axis."""
        b = self.domain.basis(coord)
        i = self.coords.index(coord)
        shape = [1] * self.ndim
        shape[i] = b.size
        return _nodes(b).reshape(shape)

    def shape_of(self, e: E.Expr) -> tuple[int, ...]:
        """Array shape of ``e``: tensor component axes, then one axis per coordinate (1 if absent)."""
        comps = tuple(cs.dim for cs in e.tensorsig)
        grid = tuple(self.domain.basis(c).size if (e.domain is not None and e.domain.has(c))
                     else 1 for c in self.coords)
        return comps + grid

    # 1D kernels along an axis --------------------------------------------------
    def _axis(self, coord: Coordinate, rank: int) -> int:
        return rank + self.coords.index(coord)

    def _cheb_coeffs(self, f: np.ndarray, b: Basis, axis: int) -> np.ndarray:
        t = _tnodes(b)
        V = C.chebvander(t, b.size - 1)
        fm = np.moveaxis(f, axis, 0)
        c = np.linalg.solve(V, fm.reshape(b.size, -1)).reshape(fm.shape)
        return c  # coefficient axis first

    def _cheb_eval(self, c: np.ndarray, b: Basis, axis: int) -> np.ndarray:
        t = _tnodes(b)
        V = C.chebvander(t, c.shape[0] - 1)
        out = (V @ c.reshape(c.shape[0], -1)).reshape((b.size,) + c.shape[1:])
        return np.moveaxis(out, 0, axis)

    def _fourier_coeffs(self, f: np.ndarray, b: Basis, axis: int) -> tuple[np.ndarray, np.ndarray]:
        N = b.size
        fh = np.fft.fft(f, axis=axis) / N
        k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / b.length
        if N % 2 == 0:
            k[N // 2] = 0.0
            sl = [slice(None)] * f.ndim
            sl[axis] = N // 2
            fh[tuple(sl)] = 0.0  # drop the Nyquist mode (ambiguous derivative)
        return fh, k

    def _bshape(self, v: np.ndarray, axis: int, ndim: int) -> np.ndarray:
        shape = [1] * ndim
        shape[axis] = v.size
        return v.reshape(shape)

    def _diff(self, f: np.ndarray, coord: Coordinate, rank: int, order: int = 1) -> np.ndarray:
        axis = self._axis(coord, rank)
        if f.shape[axis] == 1:
            return np.zeros_like(f)
        b = self.domain.basis(coord)
        if _is_fourier(b):
            fh, k = self._fourier_coeffs(f, b, axis)
            out = np.fft.ifft(fh * self._bshape((1j * k) ** order, axis, f.ndim), axis=axis)
            out = out * b.size
            return out.real if np.isrealobj(f) else out
        fam = _fam(b)
        if fam is not None:
            c = fam.forward(f, axis, dealias=False)
            c = np.moveaxis(c, axis, 0)
            for _ in range(order):
                c = fam.diff_coeffs(c)
            return fam.backward(np.moveaxis(c, 0, axis), axis, dealias=False)
        if getattr(b, "map", None) is not None:       # chain rule d/dy = mu(xi) d/dxi, per order
            mu = self._bshape(b.map.metric(_tnodes(b)), axis, f.ndim)
            for _ in range(order):
                f = mu * self._cheb_eval(C.chebder(self._cheb_coeffs(f, b, axis), axis=0), b, axis)
            return f
        c = self._cheb_coeffs(f, b, axis)
        for _ in range(order):
            c = C.chebder(c, axis=0) * (2 / b.length)
        return self._cheb_eval(c, b, axis)

    def _integrate(self, f: np.ndarray, coord: Coordinate, rank: int) -> np.ndarray:
        axis = self._axis(coord, rank)
        b = self.domain.basis(coord)
        if _is_fourier(b):
            return f.mean(axis=axis, keepdims=True) * b.length
        fam = _fam(b)
        if fam is not None:
            c = np.moveaxis(fam.forward(f, axis, dealias=False), axis, 0)
            return np.expand_dims(np.tensordot(fam.integ_weights(), c, axes=(0, 0)), axis)
        if getattr(b, "map", None) is not None:
            raise NotImplementedError(f"integration along the mapped interval {b.bounds}: not implemented")
        c = self._cheb_coeffs(f, b, axis)
        ci = C.chebint(c, axis=0)
        val = (C.chebval(1.0, ci) - C.chebval(-1.0, ci)) * b.length / 2
        return np.expand_dims(val, axis)

    def _interp(self, f: np.ndarray, coord: Coordinate, pos: float, rank: int,
                side: str | None = None) -> np.ndarray:
        axis = self._axis(coord, rank)
        b = self.domain.basis(coord)
        fam = _fam(b)
        if fam is not None:
            c = np.moveaxis(fam.forward(f, axis, dealias=False), axis, 0)
            return np.expand_dims(np.tensordot(fam.interp_row(pos, side), c, axes=(0, 0)), axis)
        if _is_fourier(b):
            fh, k = self._fourier_coeffs(f, b, axis)
            ph = self._bshape(np.exp(1j * k * (pos - b.bounds[0])), axis, f.ndim)
            val = (fh * ph).sum(axis=axis, keepdims=True)
            return val.real if np.isrealobj(f) else val
        c = self._cheb_coeffs(f, b, axis)
        t = (b.map.xi_of(pos) if getattr(b, "map", None) is not None
             else 2 * (pos - b.bounds[0]) / b.length - 1)
        return np.expand_dims(C.chebval(t, c), axis)

    # evaluation -----------------------------------------------------------------
    def evaluate(self, e: E.Expr, bindings: Mapping[E.Field, Any],
                 params: Mapping[str, Any] | None = None) -> np.ndarray:
        """Grid values of ``e`` with the fields bound to arrays or FieldData (parameters by name)."""
        params = dict(params or {})
        cache: dict[E.Expr, np.ndarray] = {}

        def ev(x: E.Expr) -> np.ndarray:
            if x in cache:
                return cache[x]
            r = self._eval_node(x, ev, bindings, params)
            cache[x] = r
            return r

        return ev(e)

    def _eval_node(self, x: E.Expr, ev, bindings, params) -> np.ndarray:
        r = x.rank
        if isinstance(x, E.Field):
            if x not in bindings:
                raise KeyError(f"no data bound for field {x.name}")
            d = bindings[x]
            if isinstance(d, FieldData):
                if d.layout != "grid":
                    raise NotImplementedError("reference evaluator takes grid-layout data only")
                d = d.data
            d = np.asarray(d)
            try:
                return np.broadcast_to(d, self.shape_of(x))
            except ValueError:
                raise OptionError(f"data for {x.name} has shape {d.shape}, not broadcastable to "
                                  f"{self.shape_of(x)}",
                                  fix="give the data on the field's grid (shape_of(field)) or a "
                                      "broadcastable array") from None
        if isinstance(x, E.Parameter):
            v = params.get(x.name, x.value)
            if v is None:
                from ..errors import ParameterError
                raise ParameterError(f"no value for parameter {x.name}",
                                     where=f"Parameter('{x.name}') in the evaluated expression",
                                     why="expressions are evaluated with numbers",
                                     fix=f"give it a value, Parameter('{x.name}', ...), or pass "
                                         f"params={{'{x.name}': ...}}")
            return np.asarray(v)
        if isinstance(x, E.UnitVector):
            out = np.zeros((x.coordsys.dim,) + (1,) * self.ndim)
            out[x.index] = 1.0
            return out
        if isinstance(x, E.Constant):
            if x.rank:
                return np.zeros(tuple(cs.dim for cs in x.tensorsig) + (1,) * self.ndim)
            return np.asarray(x.value)
        if isinstance(x, E.Add):
            out = ev(x.args[0])
            for a in x.args[1:]:
                out = out + ev(a)
            return out
        if isinstance(x, E.Mul):
            out = np.asarray(1.0)
            for a in x.args:
                out = out * ev(a)
            return out
        if isinstance(x, E.Power):
            return ev(x.args[0]) ** x.exponent
        if isinstance(x, E.UnaryFunction):
            a = ev(x.args[0])
            with np.errstate(all="ignore"):
                out = x.numpy_func(a)
            if not np.all(np.isfinite(out)):
                raise OptionError(f"{x}: non-finite values (e.g. log/sqrt of a non-positive "
                                  "argument or overflow) — refused, not clipped",
                                  fix="check the argument range of log/sqrt/power in the expression (values "
                                      "are never clipped)")
            return out
        if isinstance(x, E.Conj):
            return np.conj(ev(x.args[0]))
        if isinstance(x, E.Component):
            return ev(x.args[0])[x.index]
        if isinstance(x, (E.Dot, E.Outer)):
            a, b = x.args
            A, B = ev(a), ev(b)
            if isinstance(x, E.Outer):
                return self._outer(A, a.rank, B, b.rank)
            dim = a.tensorsig[-1].dim
            out = 0
            for i in range(dim):
                out = out + self._outer(np.take(A, i, axis=a.rank - 1), a.rank - 1, B[i], b.rank - 1)
            return out
        if isinstance(x, E.Differentiate):
            return self._diff(ev(x.args[0]), x.coord, r)
        if isinstance(x, E.Gradient):
            f = ev(x.args[0])
            f = np.broadcast_to(f, self._full(f, x.args[0].rank))
            return np.stack([self._diff(f, c, x.args[0].rank) for c in x.coordsys.coords])
        if isinstance(x, E.Divergence):
            f = ev(x.args[0])
            cs = x.args[0].tensorsig[0]
            return sum(self._diff(f[i], c, r) for i, c in enumerate(cs.coords))
        if isinstance(x, E.Laplacian):
            f = ev(x.args[0])
            return sum(self._diff(f, c, r, order=2) for c in x.coordsys.coords)
        if isinstance(x, E.Curl):
            return self._curl(x, ev(x.args[0]))
        if isinstance(x, E.Integrate):
            f = ev(x.args[0])
            for c in x.coords:
                f = self._integrate(f, c, r)
            return f
        if isinstance(x, E.Interpolate):
            return self._interp(ev(x.args[0]), x.coord, x.position, r, getattr(x, "side", None))
        if isinstance(x, E.Lift):
            f = ev(x.args[0])
            b = x.basis
            axis = self._axis(b.coord, r)
            t = _tnodes(b)
            if getattr(x, "lam", 0):          # C^(lam)_{N+n}, standard Gegenbauer normalization
                from scipy.special import eval_gegenbauer
                vals = eval_gegenbauer(b.size + x.n, float(x.lam), t)
            else:
                coeffs = np.zeros(b.size + x.n + 1)
                coeffs[-1] = 1.0
                vals = C.chebval(t, coeffs)
            poly = self._bshape(vals, axis, max(f.ndim, r + self.ndim))
            return f * poly
        if isinstance(x, E.TimeDerivative):
            raise NotImplementedError("dt() cannot be evaluated pointwise; it is a solver concept")
        raise NotImplementedError(f"reference evaluator: {type(x).__name__}")

    def _full(self, f: np.ndarray, rank: int) -> tuple[int, ...]:
        comps = f.shape[:rank]
        grid = tuple(max(f.shape[rank + i] if f.ndim > rank + i else 1, 1)
                     for i in range(self.ndim))
        return comps + grid

    def _outer(self, A: np.ndarray, ra: int, B: np.ndarray, rb: int) -> np.ndarray:
        A = np.asarray(A)
        B = np.asarray(B)
        A2 = A.reshape(A.shape[:ra] + (1,) * rb + A.shape[ra:])
        B2 = B.reshape((1,) * ra + B.shape)
        return A2 * B2

    def _curl(self, x: E.Curl, f: np.ndarray) -> np.ndarray:
        cs = x.coordsys.coords
        rin = x.args[0].rank
        if len(cs) == 2 and rin == 0:
            return np.stack([self._diff(f, cs[1], 0), -self._diff(f, cs[0], 0)])
        if len(cs) == 2 and rin == 1:
            return self._diff(f[1], cs[0], 0) - self._diff(f[0], cs[1], 0)
        d = lambda i, j: self._diff(f[i], cs[j], 0)  # noqa: E731  d_j f_i
        return np.stack([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)])


def _global_domain(e: E.Expr) -> Domain:
    dom = None
    for x in e.walk():
        dom = Domain.union(dom, x.domain)
    if dom is None:
        raise EquationError("expression has no domain",
                            fix="evaluate an expression that contains at least one field (a pure constant "
                                "has no grid)")
    return dom


def evaluate(e: E.Expr, bindings: Mapping[E.Field, Any], params: Mapping[str, Any] | None = None,
             domain: Domain | None = None) -> np.ndarray:
    """Evaluate ``e`` on the grid of ``domain`` (default: union of all node domains)."""
    dom = domain or _global_domain(e)
    return ReferenceEvaluator(dom).evaluate(e, bindings, params)
