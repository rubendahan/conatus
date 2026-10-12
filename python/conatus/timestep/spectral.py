"""Pseudospectral evaluator of explicit right-hand sides (NumPy prototype).

Evaluates an IR expression **in coefficient space**: linear operators (derivatives,
integrals, traces) act on coefficients exactly; products go through the *dealiased*
grid declared by each basis (``dealias`` is a basis attribute, never a default here):
coefficients -> M-point grid -> pointwise product -> forward transform keeping the first
N modes (that truncation is the dealiasing projection).

Array convention (same as ``reference``): tensor component axes first, then one axis per
coordinate of the coordinate system, of length N (coefficients) if the node depends on
that coordinate and 1 otherwise. A size-1 axis holds the *zero mode* only: when such an
array is added to a dependent one it is embedded in mode 0, never broadcast over modes.

This is the reference implementation of F; the C++/CUDA port of F (fused kernels, §60)
comes after. The evaluator is vectorized over all modes (no loop over groups).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..discretize.families import NEW_FAMILIES, family_of, has_family
from ..errors import ParameterError, SpectralError, UndeclaredVariableError
from ..ir import expr as E
from ..ir.bases import Basis, Chebyshev, ComplexFourier, RealFourier
from ..ir.domain import Domain
from ..reference import transforms as T
from . import xp_transforms as XT

__all__ = ["SpectralEvaluator", "SpectralError"]


def _ncomp_shape(e: E.Expr) -> tuple[int, ...]:
    return tuple(cs.dim for cs in e.tensorsig)


class SpectralEvaluator:
    """Coefficient-space evaluator on a global domain (one basis per coordinate)."""

    def __init__(self, domain: Domain, xp: Any = None) -> None:
        self.domain = domain
        self.xp = np if xp is None else xp      # numpy (reference) or cupy (GPU backend)
        self._gpu = self.xp is not np
        self._mats: dict = {}
        self.coords = domain.coordsys.coords
        self.ndim = len(self.coords)
        self.bases: list[Basis | None] = [domain.basis(c) for c in self.coords]
        for b in self.bases:
            if b is not None and not isinstance(b, (Chebyshev, RealFourier, ComplexFourier)
                                                + NEW_FAMILIES) and not has_family(b):
                raise NotImplementedError(f"pseudospectral RHS: {type(b).__name__} not supported yet")
            if getattr(b, "map", None) is not None or getattr(b, "terrain", None) is not None:
                # a mapped Chebyshev axis is a Chebyshev subclass: the affine derivative 2/L
                # below would be wrong there (0 on an unbounded map), so it is refused here
                # rather than relying on every caller
                raise NotImplementedError(
                    f"pseudospectral RHS on the mapped axis {b!r}: not supported (the IVP "
                    "solvers refuse mapped intervals; TerrainSimulation has its own evaluator)")
        self.n_products = 0          # counters (explain/profiling)
        self.n_transforms = 0

    def explain(self) -> str:
        """Backend, bases with their dealiased grids, and the counters (every key object has one: this one, which evaluates F, had none)."""
        grids = ", ".join(f"{c.name}: {type(b).__name__}({b.size}) on {b.grid_size} points"
                          if b is not None else f"{c.name}: constant"
                          for c, b in zip(self.coords, self.bases, strict=True))
        return (f"SpectralEvaluator ({'GPU, CuPy' if self._gpu else 'CPU, NumPy'}): coefficient-"
                f"space derivatives, products on the dealiased grid, projection back by the "
                f"forward transforms; {grids}; {self.n_products} product(s), "
                f"{self.n_transforms} transform(s) so far")

    # ------------------------------------------------------------------ helpers
    def _axis(self, rank: int, coord) -> int:
        return rank + self.coords.index(coord)

    def _basis(self, coord) -> Basis:
        b = self.domain.basis(coord)
        if b is None:
            raise SpectralError(f"no basis along {coord.name}")
        return b

    def _embed(self, a: np.ndarray, rank: int, shape: tuple[int, ...]) -> np.ndarray:
        """Embed ``a`` (size-1 grid axes = zero mode) into the coefficient shape ``shape``."""
        xp = self.xp
        a = xp.asarray(a)
        if a.ndim < len(shape):
            a = a.reshape(a.shape + (1,) * (len(shape) - a.ndim)) if a.ndim else \
                a.reshape((1,) * len(shape))
        a = xp.broadcast_to(a, tuple(shape[:rank]) + a.shape[rank:])
        if a.shape == tuple(shape):
            return a
        for sa, st in zip(a.shape[rank:], shape[rank:], strict=True):
            if sa not in (1, st):
                raise SpectralError(f"cannot embed coefficient shape {a.shape} into {shape}")
        out = xp.zeros(shape, dtype=np.result_type(a.dtype, float))
        out[tuple([slice(None)] * rank + [slice(0, sa) for sa in a.shape[rank:]])] = a
        return out

    def _common_shape(self, arrays, rank: int) -> tuple[int, ...]:
        comps = None
        grid = [1] * self.ndim
        for a in arrays:
            a = self.xp.asarray(a)
            if a.ndim == 0:
                continue
            c = a.shape[:rank]
            if comps is None or (len(c) and c != comps):
                comps = c if comps is None else np.broadcast_shapes(comps, c)
            for i in range(self.ndim):
                if a.ndim > rank + i:
                    grid[i] = max(grid[i], a.shape[rank + i])
        comps = comps if comps is not None else ()
        return tuple(comps) + tuple(grid)

    def to_grid(self, c: np.ndarray, rank: int) -> np.ndarray:
        """Coefficients -> dealiased grid (size-1 axes stay constant)."""
        g = self.xp.asarray(c)
        for i, b in enumerate(self.bases):
            ax = rank + i
            if b is not None and g.ndim > ax and g.shape[ax] == b.size:
                g = (XT.backward(self.xp, g, b, axis=ax) if self._gpu
                     else T.backward(g, b, axis=ax, dealias=True))
                self.n_transforms += 1
        return g

    def to_coeffs(self, g: np.ndarray, rank: int) -> np.ndarray:
        """Dealiased grid -> coefficients (first N modes kept: dealiasing projection)."""
        xp = self.xp
        c = xp.asarray(g)
        for i, b in enumerate(self.bases):
            ax = rank + i
            if b is not None and c.ndim > ax and c.shape[ax] == b.grid_size and c.shape[ax] > 1:
                if isinstance(b, RealFourier) and xp.iscomplexobj(c):
                    if float(xp.max(xp.abs(c.imag))) > 1e-13 * max(1.0, float(xp.max(xp.abs(c.real)))):
                        raise SpectralError("complex values on a RealFourier grid")
                    c = c.real
                c = (XT.forward(xp, c, b, axis=ax) if self._gpu
                     else T.forward(c, b, axis=ax, dealias=True))
                self.n_transforms += 1
        return c

    # 1D coefficient operators ------------------------------------------------
    def _diff(self, c: np.ndarray, coord, rank: int) -> np.ndarray:
        ax = self._axis(rank, coord)
        if c.ndim <= ax or c.shape[ax] == 1:
            return self.xp.zeros_like(c)
        b = self._basis(coord)
        if self._gpu:
            A = self._host(("diff", coord), b, coord, lambda r, e: r._diff(e, coord, 0))
            return self._apply_1d(("diff", coord), A, c, ax)
        N = b.size
        c = np.moveaxis(c, ax, 0)
        fam = family_of(b)
        if fam is not None:            # Hermite / Laguerre / compound (discretize/families.py)
            out = fam.diff_coeffs(c)
        elif isinstance(b, ComplexFourier):
            k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / b.length
            k[N // 2] = 0.0
            out = (1j * k).reshape((N,) + (1,) * (c.ndim - 1)) * c
        elif isinstance(b, RealFourier):
            k = 2 * np.pi * np.arange(N // 2) / b.length
            kk = k.reshape((N // 2,) + (1,) * (c.ndim - 1))
            out = np.zeros_like(c)
            cs, ms = c[0::2], c[1::2]
            out[0::2] = -kk * ms
            out[1::2] = kk * cs
        else:  # Chebyshev: c'_{n} from the backward recurrence
            out = np.zeros_like(c)
            if N > 1:
                out[N - 2] = 2 * (N - 1) * c[N - 1]
                for n in range(N - 3, -1, -1):
                    out[n] = out[n + 2] + 2 * (n + 1) * c[n + 1]
                out[0] = out[0] / 2
            out = out * (2.0 / b.length)
        return np.moveaxis(out, 0, ax)

    def _integrate(self, c: np.ndarray, coord, rank: int) -> np.ndarray:
        ax = self._axis(rank, coord)
        b = self._basis(coord)
        if c.ndim <= ax or c.shape[ax] == 1:
            return c * b.length                  # constant along coord: value x length
        if self._gpu:
            A = self._host(("integ", coord), b, coord, lambda r, e: r._integrate(e, coord, 0))
            return self._apply_1d(("integ", coord), A, c, ax)
        cm = np.moveaxis(c, ax, 0)
        fam = family_of(b)
        if fam is not None:
            val = np.tensordot(fam.integ_weights(), cm, axes=(0, 0))
        elif isinstance(b, (RealFourier, ComplexFourier)):
            val = cm[0] * b.length
        else:
            n = np.arange(b.size, dtype=float)
            w = np.zeros(b.size)
            even = n % 2 == 0
            w[even] = 2.0 / (1.0 - n[even] ** 2) * (b.length / 2)
            val = np.tensordot(w, cm, axes=(0, 0))
        return np.expand_dims(val, ax)

    def _interp(self, c: np.ndarray, coord, pos: float, rank: int, side: str | None = None) -> np.ndarray:
        ax = self._axis(rank, coord)
        if c.ndim <= ax or c.shape[ax] == 1:
            return c
        b = self._basis(coord)
        if self._gpu:
            key = ("interp", coord, float(pos), side)
            A = self._host(key, b, coord, lambda r, e: r._interp(e, coord, pos, 0, side))
            return self._apply_1d(key, A, c, ax)
        N = b.size
        cm = np.moveaxis(c, ax, 0)
        x = pos - b.bounds[0]
        fam = family_of(b)
        if fam is not None:
            w = fam.interp_row(pos, side)
        elif isinstance(b, ComplexFourier):
            k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / b.length
            w = np.exp(1j * k * x)
            w[N // 2] = 0.0
        elif isinstance(b, RealFourier):
            k = 2 * np.pi * np.arange(N // 2) / b.length
            w = np.empty(N)
            w[0::2] = np.cos(k * x)
            w[1::2] = -np.sin(k * x)
        else:
            t = 2 * x / b.length - 1
            w = np.cos(np.arange(N) * np.arccos(np.clip(t, -1, 1)))
        val = np.tensordot(w, cm, axes=(0, 0))
        return np.expand_dims(val, ax)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, e: E.Expr, coeffs: Mapping[E.Field, np.ndarray],
                 params: Mapping[str, Any] | None = None) -> np.ndarray:
        """Coefficients of ``e`` (shape: components + one axis per coordinate)."""
        params = dict(params or {})
        cache: dict[E.Expr, np.ndarray] = {}

        def ev(x: E.Expr) -> np.ndarray:
            r = cache.get(x)
            if r is None:
                r = self._node(x, ev, coeffs, params)
                cache[x] = r
            return r

        return ev(e)

    def _is_scalar_const(self, x: E.Expr) -> bool:
        return (isinstance(x, (E.Parameter, E.Constant)) and x.rank == 0) or (
            not x.free_fields() and x.domain is None and x.rank == 0
            and not any(isinstance(n, E.UnitVector) for n in x.walk()))

    def _node(self, x: E.Expr, ev, coeffs, params) -> np.ndarray:
        r = x.rank
        if isinstance(x, E.Field):
            if x not in coeffs:
                raise UndeclaredVariableError(
                    f"no coefficient data for field {x.name}",
                    where=f"field '{x.name}' in the explicit (RHS) terms",
                    why="a field that is not a variable of the IVP is data; its coefficients must "
                        "be given, they are never assumed to be zero",
                    fix=f"declare '{x.name}' as a variable of the problem if it evolves, or pass "
                        "its data as a non-constant coefficient (discretize(pb, ncc={field: "
                        "data}))")
            return self.xp.asarray(coeffs[x])
        if isinstance(x, E.Parameter):
            v = params.get(x.name, x.value)
            if v is None:
                raise ParameterError(
                    f"no value for parameter {x.name}", where=f"Parameter('{x.name}') in the RHS",
                    why="explicit terms are evaluated with numbers",
                    fix=f"give it a value, Parameter('{x.name}', ...), or pass "
                        f"discretize(pb, params={{'{x.name}': ...}}); a time-dependent parameter "
                        "is declared with IVPSolver(..., time_parameter='t')")
            return self._scalar(v)
        if isinstance(x, E.Constant):
            if x.rank:
                return self.xp.zeros(_ncomp_shape(x) + (1,) * self.ndim)
            return self._scalar(x.value)
        if isinstance(x, E.UnitVector):
            out = self.xp.zeros((x.coordsys.dim,) + (1,) * self.ndim)
            out[x.index] = 1.0
            return out
        if isinstance(x, E.Add):
            parts = [ev(a) for a in x.args]
            shape = self._common_shape(parts, r)
            out = None
            for p in parts:
                p = self._embed(p, r, shape)
                out = p if out is None else out + p
            return out
        if isinstance(x, E.Mul):
            scal = [a for a in x.args if self._is_scalar_const(a)]
            fields = [a for a in x.args if not self._is_scalar_const(a)]
            s = 1.0
            for a in scal:
                s = s * ev(a)
            if len(fields) <= 1:
                if not fields:
                    return self._scalar(s)
                return s * ev(fields[0])
            return s * self._product([ev(a) for a in fields], [a.rank for a in fields], r,
                                     outer=False)
        if isinstance(x, (E.Dot, E.Outer)):
            a, b = x.args
            A, B = ev(a), ev(b)
            if isinstance(x, E.Outer):
                return self._product([A, B], [a.rank, b.rank], r, outer=True)
            ga, gb = self.to_grid(A, a.rank), self.to_grid(B, b.rank)
            dim = a.tensorsig[-1].dim
            out = 0
            for i in range(dim):
                Ai = self.xp.take(ga, i, axis=a.rank - 1)
                out = out + self._outer_grid(Ai, a.rank - 1, gb[i], b.rank - 1)
            self.n_products += 1
            return self.to_coeffs(out, r)
        if isinstance(x, E.Cross):
            # (a x b)_k = sum sign a_i b_j on the dealiased grid (feat2; frame of
            # cartesian_cross_table: e_x x e_y = e_z for coordinates named x, y, z)
            from ..ir.expr_algebra import cartesian_cross_table
            a, b = x.args
            ga, gb = self.to_grid(ev(a), 1), self.to_grid(ev(b), 1)
            terms: dict = {}
            for (i, j), (k, sg) in cartesian_cross_table(a.tensorsig[0]).items():
                t = sg * (ga[i] * gb[j])
                terms[k] = t if k not in terms else terms[k] + t
            shape = np.broadcast_shapes(*[np.shape(v) for v in terms.values()])
            out = self.xp.stack([self.xp.broadcast_to(terms[k], shape) for k in range(3)])
            self.n_products += 1
            return self.to_coeffs(out, r)
        if isinstance(x, E.Component):
            return ev(x.args[0])[x.index]
        if isinstance(x, E.Differentiate):
            c = ev(x.args[0])
            return self._diff(self._full(c, x.args[0]), x.coord, r)
        if isinstance(x, E.Gradient):
            c = self._full(ev(x.args[0]), x.args[0])
            ra = x.args[0].rank
            return self.xp.stack([self._diff(c, co, ra) for co in x.coordsys.coords])
        if isinstance(x, E.Divergence):
            c = self._full(ev(x.args[0]), x.args[0])
            cs = x.args[0].tensorsig[0]
            return sum(self._diff(c[i], co, r) for i, co in enumerate(cs.coords))
        if isinstance(x, E.Laplacian):
            c = self._full(ev(x.args[0]), x.args[0])
            return sum(self._diff(self._diff(c, co, r), co, r) for co in x.coordsys.coords)
        if isinstance(x, E.Curl):
            c = self._full(ev(x.args[0]), x.args[0])
            cs = x.coordsys.coords
            rin = x.args[0].rank
            if len(cs) == 2 and rin == 0:
                return self.xp.stack([self._diff(c, cs[1], 0), -self._diff(c, cs[0], 0)])
            if len(cs) == 2 and rin == 1:
                return self._diff(c[1], cs[0], 0) - self._diff(c[0], cs[1], 0)
            d = lambda i, j: self._diff(c[i], cs[j], 0)  # noqa: E731
            return self.xp.stack([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)])
        if isinstance(x, E.Integrate):
            c = self._full(ev(x.args[0]), x.args[0])
            for co in x.coords:
                c = self._integrate(c, co, r)
            return c
        if isinstance(x, E.Interpolate):
            return self._interp(self._full(ev(x.args[0]), x.args[0]), x.coord, x.position, r,
                                getattr(x, "side", None))
        if isinstance(x, E.Power):
            g = self.to_grid(self._full(ev(x.args[0]), x.args[0]), r)
            self.n_products += 1
            out = g ** x.exponent
            # a negative or fractional power of a field that vanishes somewhere is inf/nan on the
            # grid: refused like a pointwise function (a positive integer power of finite values
            # cannot overflow to it here, so no device synchronisation is added for u**2)
            e = x.exponent                # numeric; a complex exponent is checked as well
            whole = isinstance(e, (int, float, np.integer, np.floating)) and e >= 0 \
                and float(e).is_integer()
            if not whole and not bool(self.xp.all(self.xp.isfinite(out))):
                raise SpectralError(f"{x}: non-finite values on the grid (refused, not clipped)")
            return self.to_coeffs(out, r)
        if isinstance(x, E.UnaryFunction):
            # pointwise function on the (dealiased) grid; not alias-free (declared by the IR node)
            g = self.to_grid(self._full(ev(x.args[0]), x.args[0]), r)
            self.n_products += 1
            if E.is_user_function(x.func):
                out = self.xp.asarray(x.numpy_func(g))   # the user's function (pointwise())
            else:
                out = getattr(self.xp, x.numpy_func.__name__)(g)
            if not bool(self.xp.all(self.xp.isfinite(out))):
                raise SpectralError(f"{x}: non-finite values on the grid (refused, not clipped)")
            return self.to_coeffs(out, r)
        if isinstance(x, E.Conj):
            g = self.to_grid(self._full(ev(x.args[0]), x.args[0]), r)
            return self.to_coeffs(self.xp.conj(g), r)
        if isinstance(x, (E.Lift, E.TimeDerivative)):
            raise SpectralError(f"{type(x).__name__} cannot appear in an explicit RHS")
        raise SpectralError(f"pseudospectral RHS: node {type(x).__name__} not supported")

    # GPU helpers: 1D operators as small dense matrices, built once by applying the NumPy
    # reference operator to unit vectors (same conventions by construction)
    def _scalar(self, v):
        a = np.asarray(v)
        if not self._gpu:
            return a
        return a.item() if a.ndim == 0 else self.xp.asarray(a)

    def _unit(self, b: Basis, coord, j: int) -> np.ndarray:
        shape = [1] * self.ndim
        ax = self.coords.index(coord)
        shape[ax] = b.size
        e = np.zeros(shape, dtype=complex if isinstance(b, ComplexFourier) else float)
        idx = [0] * self.ndim
        idx[ax] = j
        e[tuple(idx)] = 1.0
        return e

    def _host_matrix(self, op, b: Basis, coord) -> np.ndarray:
        """Matrix A (rows: output modes along ``coord``) with op(c) = A c, from the reference."""
        ref = SpectralEvaluator(self.domain)
        cols = [np.asarray(op(ref, self._unit(b, coord, j))).reshape(-1) for j in range(b.size)]
        return np.stack(cols, axis=1)

    def _host(self, key, b: Basis, coord, op) -> np.ndarray:
        A = self._mats.get(key)
        if A is None:
            A = self._mats[key] = self._host_matrix(op, b, coord)
        return A

    def _apply_1d(self, key, A: np.ndarray, c, ax: int):
        """out[.., n, ..] = sum_m A[n, m] c[.., m, ..] along axis ``ax`` (A: host matrix; on the
        GPU a CSR kernel, capturable in a CUDA graph, unlike cuBLAS/cuSPARSE through CuPy)."""
        import scipy.sparse as sp

        from .device import DeviceCsr
        xp = self.xp
        dt = np.result_type(A.dtype, c.dtype)
        dkey = (key, np.dtype(dt).str)
        D = self._mats.get(dkey)
        if D is None:
            D = self._mats[dkey] = (sp.csr_matrix(A.astype(dt)) if xp is np
                                    else DeviceCsr(sp.csr_matrix(A), dt))
        cm = xp.moveaxis(c, ax, 0)
        sh = cm.shape
        out = D @ cm.reshape(sh[0], -1)
        return xp.moveaxis(out.reshape((A.shape[0],) + sh[1:]), 0, ax)

    def _full(self, c: np.ndarray, e: E.Expr) -> np.ndarray:
        """Give ``c`` its full axis count (components + ndim), size-1 where constant."""
        c = self.xp.asarray(c)
        want = e.rank + self.ndim
        if c.ndim == want:
            return c
        if c.ndim == 0:
            return c.reshape((1,) * want)
        return c.reshape(c.shape + (1,) * (want - c.ndim))

    def _outer_grid(self, A, ra, B, rb):
        A2 = A.reshape(A.shape[:ra] + (1,) * rb + A.shape[ra:])
        B2 = B.reshape((1,) * ra + B.shape)
        return A2 * B2

    def _product(self, arrays, ranks, rank_out, outer: bool) -> np.ndarray:
        xp = self.xp
        grids = [self.to_grid(xp.asarray(a).reshape(xp.asarray(a).shape + (1,) * (ra + self.ndim - xp.asarray(a).ndim)),
                              ra) for a, ra in zip(arrays, ranks, strict=True)]
        if outer:
            out, ro = grids[0], ranks[0]
            for g, rg in zip(grids[1:], ranks[1:], strict=True):
                out = self._outer_grid(out, ro, g, rg)
                ro += rg
        else:
            # Mul: at most one tensor factor; scalars broadcast over its components
            out = None
            for g, rg in zip(grids, ranks, strict=True):
                if out is None:
                    out, ro = g, rg
                elif rg == 0:
                    out = out * g.reshape((1,) * ro + g.shape) if ro else out * g
                else:
                    out = g * out.reshape((1,) * rg + out.shape) if ro == 0 else out * g
                    ro = max(ro, rg)
        self.n_products += 1
        return self.to_coeffs(out, rank_out)
