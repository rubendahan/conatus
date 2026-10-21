"""Spherical plans, part 2: compilation of IR expressions into sparse blocks for one group.

``_SphereCompiler`` walks an expression and returns, per output tensor component, the blocks of each
degree l: fields, grad / div / lap / curl (spin components with frame terms on the shell, regularity
components in the ball), components, traces, integrals, tau lifts, and products with non-constant
coefficients (NCC). The rules are listed in the docstring of ``conatus.discretize.sphere_plan``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from ..ir import expr as E
from .compiler import DiscretizationError
from .curvilinear_common import const_value
from .sphere_eval import comps
from .sphere_ops import cross_table, spin_of
from .sphere_blocks import _SQ2, SphereGroup, _Ent, _relabel, _sum_blocks

if TYPE_CHECKING:
    from .sphere_plan import SpherePlan


# ----------------------------------------------------------------------------------------------
# compiler (one group)
# ----------------------------------------------------------------------------------------------

class _SphereCompiler:
    def __init__(self, plan: SpherePlan, group: SphereGroup) -> None:
        self.P, self.g, self.m = plan, group, group.m
        self.S = plan.space
        self.ops = plan.space.ops
        self.nc = plan.space.nc
        self.vars = frozenset(plan.variables)
        self.mode = "L"
        self._lay: dict = {}

    # ---------------------------------------------------------------- layouts
    def lay(self, comp: tuple, bulk: bool) -> list[tuple[int, int]]:
        """[(l, rows)] of an entry of component ``comp``: valid degrees of the group (spin/regularity
        rule l >= max(|m|, |s|)), n_l radial rows in the 3D bulk, one row otherwise."""
        key = (comp, bulk)
        if key not in self._lay:
            s = spin_of(comp)
            S = self.S
            if S.ball and len(comp) == 1:
                ok = lambda ell: S.valid_reg(self.m, comp, ell)       # noqa: E731
            else:
                ok = lambda ell: ell >= max(abs(self.m), abs(s))      # noqa: E731
            self._lay[key] = [(ell, S.rsize(ell) if (bulk and not S.s2) else 1)
                              for ell in self.g.ells if ok(ell)]
        return self._lay[key]

    def nrows(self, comp: tuple, bulk: bool) -> int:
        return sum(n for _, n in self.lay(comp, bulk))

    def bd(self, comp_out, bulk_out, comp_in, bulk_in, f) -> sp.csr_matrix:
        """Block-diagonal (in l) operator from an entry's rows to new rows; ``f(l, n_out, n_in)``
        returns the (n_out, n_in) block (sparse, dense or scalar) or None."""
        lo, li = self.lay(comp_out, bulk_out), self.lay(comp_in, bulk_in)
        if len(lo) == 1 and len(li) == 1 and lo[0][0] == li[0][0]:
            # one degree (separable 3D groups): the block itself (fast path, same result)
            (ell, n_o), (_, n_i) = lo[0], li[0]
            B = f(ell, n_o, n_i)
            if B is None or (np.isscalar(B) and B == 0):
                return sp.csr_matrix((n_o, n_i), dtype=complex)
            if np.isscalar(B):
                if n_o != n_i:
                    raise DiscretizationError("internal: scalar block between different sizes")
                return sp.identity(n_o, dtype=complex, format="csr") * B
            return sp.csr_matrix(B, dtype=complex)
        oo, oi = {}, {}
        off = 0
        for ell, n in lo:
            oo[ell] = (off, n)
            off += n
        no = off
        off = 0
        for ell, n in li:
            oi[ell] = (off, n)
            off += n
        ni = off
        rows, cols, vals = [], [], []
        for ell in oo:
            if ell not in oi:
                continue
            (ro, n_o), (ci, n_i) = oo[ell], oi[ell]
            B = f(ell, n_o, n_i)
            if B is None:
                continue
            if np.isscalar(B):
                if B == 0:
                    continue
                if n_o != n_i:
                    raise DiscretizationError("internal: scalar block between different sizes")
                B = B * sp.identity(n_o, format="coo")
            B = sp.coo_matrix(B)
            rows.append(B.row + ro)
            cols.append(B.col + ci)
            vals.append(B.data.astype(complex))
        if not rows:
            return sp.csr_matrix((no, ni), dtype=complex)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(no, ni))

    def full(self, comp_out, bulk_out, comp_in, bulk_in, f) -> sp.csr_matrix:
        """Operator coupling degrees: ``f(l_out, l_in, n_out, n_in)`` -> block or None."""
        lo, li = self.lay(comp_out, bulk_out), self.lay(comp_in, bulk_in)
        blocks = [[f(a, b, na, nb) for b, nb in li] for a, na in lo]
        grid = [[(sp.coo_matrix(B) if B is not None else sp.coo_matrix((na, nb)))
                 for B, (b, nb) in zip(row, li)] for row, (a, na) in zip(blocks, lo)]
        if not lo or not li:
            return sp.csr_matrix((self.nrows(comp_out, bulk_out), self.nrows(comp_in, bulk_in)),
                                 dtype=complex)
        return sp.bmat(grid, format="csr").astype(complex)

    # ---------------------------------------------------------------- sums
    def add(self, ents: list, comp: tuple) -> _Ent:
        ents = [e for e in ents if e is not None]
        if not any(e.bulk for e in ents):
            return _Ent(False, comp, None, None, _sum_blocks(ents))
        level = max(e.level for e in ents if e.bulk)
        ps = [e.p for e in ents if e.bulk and e.p is not None]
        p = max(ps) if ps else 0
        return _Ent(True, comp, level, p, _sum_blocks([self.normalize(e, comp, level, p) for e in ents]))

    def normalize(self, e: _Ent, comp: tuple, level: int, p: int) -> _Ent:
        """Bring an entry to a row's (level, p): conversion (bidiagonal / banded) and, on the shell,
        multiplication by r^(p - p_e); point quantities are broadcast as constants in r."""
        if not e.bulk:
            e = self.broadcast(e, comp)
        out = e
        if e.level != level:
            out = e.map(self.bd(comp, True, comp, True,
                                lambda ell, no, ni: self._convert(comp, ell, e.level, level, no)))
        if e.p is not None and e.p != p:
            if p < e.p:
                raise DiscretizationError("internal: r-power above the row power")
            out = out.map(self.bd(comp, True, comp, True,
                                  lambda ell, no, ni: self.ops.multiply_r(level, p - e.p, no)))
        return _Ent(True, comp, level, p, out.blocks)

    def _convert(self, comp, ell, lev_from, lev_to, n):
        if self.S.s2:
            return 1.0
        if self.S.ball:
            return self.ops.convert(ell + spin_of(comp), lev_from, lev_to, n)
        return self.ops.convert(lev_from, lev_to, n)

    def broadcast(self, e: _Ent, comp: tuple) -> _Ent:
        """A point quantity (surface value or constant) inside a bulk equation: constant in r. On
        S2 the rows coincide (one per l). In the ball only l = 0 is regular (rho^l needed)."""
        if self.S.s2:
            return _Ent(True, comp, 0, 0, dict(e.blocks))
        if self.S.ball:
            rows = [i for i, (ell, _) in enumerate(self.lay(comp, False)) if ell > 0]
            if rows and any(b[rows].nnz for b in e.blocks.values()):
                raise DiscretizationError(
                    f"[{self.g.label}] a quantity without radial dependence and degree l > 0 enters a "
                    "ball equation: it is not regular at the centre (use lift() for taus)")
            f = lambda ell, no, ni: sp.csr_matrix(self.ops.const_col(no)[:, None]) if ell == 0 else None  # noqa: E731
        else:
            f = lambda ell, no, ni: sp.csr_matrix(self.ops.const_col()[:, None])  # noqa: E731
        return e.map(self.bd(comp, True, comp, False, f), bulk=True, level=0, p=0)

    def _param(self, p: E.Parameter):
        v = self.P.params.get(p.name, p.value)
        if v is None:
            raise DiscretizationError(f"parameter {p.name} has no value")
        return v

    def _const(self, f: E.Expr):
        return const_value(f, self._param)

    # ---------------------------------------------------------------- entry point
    def compile(self, x: E.Expr, mode: str) -> dict:
        self.mode = mode
        return self._c(x)

    def _c_L(self, x: E.Expr) -> dict:
        saved, self.mode = self.mode, "L"
        try:
            return self._c(x)
        finally:
            self.mode = saved

    def _c(self, x: E.Expr) -> dict:
        V, ev = self.vars, self.P.eigenvalue
        if not x.depends_on(V) and not isinstance(x, E.TimeDerivative):
            raise DiscretizationError(f"'{x}' does not depend on the unknowns: it belongs on the RHS")
        if isinstance(x, E.Field):
            if self.mode == "M":
                # a bare unknown has no dt() and no eigenvalue: it never enters M (the defect found
                # in the Cartesian and polar compilers, avoided here)
                return {}
            return self._field(x)
        if isinstance(x, E.TimeDerivative):
            if self.mode != "M":
                return {}
            arg = x.args[0]
            if any(isinstance(n, E.TimeDerivative) for n in arg.walk()):
                raise DiscretizationError(f"nested time derivative in '{x}': not supported")
            out = self._c_L(arg)
            if not any(e.blocks for e in out.values()):
                raise DiscretizationError(f"'{x}' compiles to an empty mass operator")
            return out
        if self.mode == "M" and not any(isinstance(n, E.TimeDerivative) for n in x.walk()) and (
                ev is None or ev not in x.free_parameters()):
            return {}
        if isinstance(x, E.Add):
            parts = [self._c(t) for t in x.args if t.depends_on(V) or isinstance(t, E.TimeDerivative)]
            out = {}
            for c in {c for p in parts for c in p}:
                out[c] = self.add([p[c] for p in parts if c in p], c)
            return out
        if isinstance(x, E.Mul):
            return self._mul(x)
        if isinstance(x, (E.Gradient, E.Divergence, E.Laplacian, E.Curl)):
            inner = self._c(x.args[0])
            if self.S.ball:
                return self._ball_op(x, inner)
            if isinstance(x, E.Gradient):
                return self._grad(inner, x)
            if isinstance(x, E.Divergence):
                return self._contract(self._grad(inner, x), x.rank)
            if isinstance(x, E.Laplacian):
                return self._contract(self._grad(self._grad(inner, x), x), x.rank)
            return self._curl(self._grad(inner, x), x)
        if isinstance(x, E.Component):
            return self._component(self._c(x.args[0]), x)
        if isinstance(x, E.Differentiate):
            if x.coord == self.S.phi:
                if x.args[0].rank:
                    raise DiscretizationError(f"'{x}': d/dphi of a tensor (frame rotates): use grad")
                return {c: e.scale(1j * self.m) for c, e in self._c(x.args[0]).items()}
            raise DiscretizationError(
                f"'{x}': partial derivatives along theta or r are not regular operators here; use "
                "grad/div/lap/curl, or interp(diff(., r), r, r0) for a boundary condition")
        if isinstance(x, E.Interpolate):
            if self.S.s2 or x.coord != self.S.r:
                raise DiscretizationError(f"'{x}': interpolation along {x.coord.name} couples all "
                                          "its wavenumbers (not separable)")
            return self._trace(x.args[0], x.position, 0)
        if isinstance(x, E.Integrate):
            return self._integrate(x)
        if isinstance(x, E.Lift):
            return self._lift(x)
        if isinstance(x, (E.Dot, E.Cross, E.Outer)):
            return self._product(x)
        raise DiscretizationError(f"cannot discretize node {type(x).__name__} ('{x}') on a spherical domain")

    def _need_bulk(self, e: _Ent, x: E.Expr) -> None:
        if not e.bulk:
            raise DiscretizationError(f"'{x}': derivative of a quantity without dependence on the "
                                      "innermost coordinate")

    def _p(self, e: _Ent, x: E.Expr) -> int:
        if e.p is None:
            raise DiscretizationError(f"'{x}': derivative of a lifted tau term")
        return e.p

    # ---------------------------------------------------------------- fields
    def _field(self, x: E.Field) -> dict:
        """Selection of the unknown's coefficients into the entry layout (identity where the unknown
        has the block, zero rows where its domain lacks phi or theta)."""
        out = {}
        bulk = self.S.bulk(x.domain)
        if self.S.ball and bulk and x.rank > 1:
            raise DiscretizationError(f"{x.name}: ball unknowns of rank >= 2 are not implemented")
        for c in comps(x.rank, self.nc):
            rows = self.lay(c, bulk)
            cols = self.P.obj_layout(x.domain, c, self.g)
            cpos, off = {}, 0
            for ell, n in cols:
                cpos[ell] = (off, n)
                off += n
            r_i, c_i, roff = [], [], 0
            for ell, n in rows:
                if ell in cpos:
                    o, nn = cpos[ell]
                    r_i.extend(range(roff, roff + nn))
                    c_i.extend(range(o, o + nn))
                roff += n
            blk = sp.csr_matrix((np.ones(len(r_i)), (r_i, c_i)), shape=(roff, off), dtype=complex)
            out[c] = _Ent(bulk, c, 0 if bulk else None, 0 if bulk else None, {(x, c): blk})
        return out

    # ---------------------------------------------------------------- S2 / shell (spin, Leibniz)
    def _grad(self, inner: dict, x: E.Expr) -> dict:
        S = self.S
        parts: dict = {}

        def put(c, e):
            parts.setdefault(c, []).append(e)

        def over_r(e: _Ent, comp_out: tuple, coef_fn) -> _Ent:
            """(coef(l) / r) * e into component comp_out (S2: / R, level and p unchanged)."""
            A = self.bd(comp_out, True, e.comp, True, lambda ell, no, ni: coef_fn(ell))
            if S.s2:
                return e.map(A / S.R, comp=comp_out)
            return e.map(A, comp=comp_out, p=self._p(e, x) + 1)

        for c, e in inner.items():
            self._need_bulk(e, x)
            s = spin_of(c)
            for d, sigma in ((0, -1), (1, 1)):
                G = S.H.G(sigma, self.m, s)
                put((d,) + c, over_r(e, (d,) + c, lambda ell, G=G: G[ell]))
            if S.s2:
                continue
            A = self.bd((2,) + c, True, c, True,     # bd calls f eagerly; e bound for clarity
                        lambda ell, no, ni, e=e: self.ops.dr(e.level, self._p(e, x), no)[0])
            put((2,) + c, e.map(A, comp=(2,) + c, level=e.level + 1, p=self._p(e, x) + 1))
        if not S.s2:                                   # frame terms, index by index
            for c in inner:
                for i, t in enumerate(c):
                    rep = lambda j, c=c, i=i: c[:i] + (j,) + c[i + 1:]      # noqa: E731
                    one = lambda ell: 1.0                          # noqa: E731
                    if t == 0 and rep(2) in inner:
                        put((1,) + c, over_r(inner[rep(2)], (1,) + c, one))
                    if t == 1 and rep(2) in inner:
                        put((0,) + c, over_r(inner[rep(2)], (0,) + c, one))
                    if t == 2:
                        if rep(1) in inner:
                            put((1,) + c, over_r(inner[rep(1)], (1,) + c, one).scale(-1.0))
                        if rep(0) in inner:
                            put((0,) + c, over_r(inner[rep(0)], (0,) + c, one).scale(-1.0))
        return {c: self.add(es, c) for c, es in parts.items()}

    def _contract(self, g: dict, rank_out: int) -> dict:
        out = {}
        G = self.S.G
        for c in comps(rank_out, self.nc):
            # the contracted pairs (-,+), (+,-), (r,r) have total spin 0: same row layout as c
            terms = [_relabel(g[(d, t) + c], c).scale(G[d, t])
                     for d in range(self.nc) for t in range(self.nc) if G[d, t] and (d, t) + c in g]
            if terms:
                out[c] = self.add(terms, c)
        return out

    def _curl(self, g: dict, x: E.Expr) -> dict:
        rin = x.args[0].rank
        if self.S.s2:
            if rin == 0:
                out = {}
                if (0,) in g:
                    out[(0,)] = g[(0,)].scale(1j)
                if (1,) in g:
                    out[(1,)] = g[(1,)].scale(-1j)
                return out
            terms = []
            if (1, 0) in g:
                terms.append(_relabel(g[(1, 0)], ()).scale(1j))
            if (0, 1) in g:
                terms.append(_relabel(g[(0, 1)], ()).scale(-1j))
            return {(): self.add(terms, ())} if terms else {}
        if rin != 1:
            raise DiscretizationError("curl in 3D: vectors only")
        parts: dict = {}
        for (d, t), (k, cst) in cross_table().items():
            if (d, t) in g:
                parts.setdefault((k,), []).append(_relabel(g[(d, t)], (k,)).scale(cst))
        return {c: self.add(es, c) for c, es in parts.items()}

    def _component(self, inner: dict, x: E.Component) -> dict:
        i = x.index
        S = self.S
        if S.s2 or i != 2:
            raise DiscretizationError(
                f"'{x}': tangential physical components (u_phi, u_theta) are not regular on the sphere; "
                "use dot products or grad/div/curl (the radial component, index 2, is allowed in 3D)")
        if S.ball and any(e.bulk for e in inner.values()):
            raise DiscretizationError(
                f"'{x}': the radial component u_r is not regular in the ball (it mixes regularity "
                "l-1 and l+1: u = e_z gives u_r = cos(theta)); use it at the boundary (interp)")
        if S.ball and x.args[0].rank == 1:
            return self._ball_radial(inner)
        out = {}
        for c, e in inner.items():
            if c and c[0] == 2:
                out[c[1:]] = _relabel(e, c[1:])
        return out

    def _ball_radial(self, inner: dict) -> dict:
        """u_r of a ball vector held on a surface in regularity components: sum_sigma Q[r, sigma] U_sigma
        (the ball labels its surface vectors by regularity, sphere_eval.valid_reg)."""
        terms = []
        for (j,), e in inner.items():
            A = self.bd((), False, (j,), False,
                        lambda ell, no, ni, j=j: complex(self.P.Q(self.m, ell)[2, j]) or None)
            if A.nnz:
                terms.append(e.map(A, comp=()))
        return {(): _Ent(False, (), None, None, _sum_blocks(terms))} if terms else {}

    # ---------------------------------------------------------------- ball (regularity components)
    def _ball_op(self, x: E.Expr, inner: dict) -> dict:
        rin = x.args[0].rank
        if rin > 1 or x.rank > 1:
            raise DiscretizationError(f"'{x}': ball operators on tensors of rank >= 2 are not "
                                      "implemented (rank <= 1)")
        if isinstance(x, E.Laplacian):
            out = {}
            for c, e in inner.items():
                self._need_bulk(e, x)
                k0 = spin_of(c)
                A = self.bd(c, True, c, True, lambda ell, no, ni, k0=k0, e=e: self.ops.lap(ell + k0, e.level, no))
                out[c] = e.map(A, level=e.level + 2)
            return out
        kind = {E.Gradient: "grad", E.Divergence: "div", E.Curl: "curl"}[type(x)]
        parts: dict = {}
        cin = comps(rin, 3)
        cout = comps(x.rank, 3)
        for jc, c in enumerate(cin):
            if c not in inner:
                continue
            e = inner[c]
            self._need_bulk(e, x)
            for io, co in enumerate(cout):
                def f(ell, no, ni, io=io, jc=jc, c=c, e=e):
                    mp = self.P.ball_map(kind, self.m, ell)
                    if (io, jc) not in mp:
                        return None
                    op, coef = mp[(io, jc)]
                    k_in = ell + spin_of(c)
                    M = self.ops.up(k_in, e.level, ni) if op == "up" else self.ops.down(k_in, e.level, ni)
                    return coef * M
                A = self.bd(co, True, c, True, f)
                if A.nnz:
                    parts.setdefault(co, []).append(e.map(A, comp=co, level=e.level + 1))
        return {c: self.add(es, c) for c, es in parts.items()}

    # ---------------------------------------------------------------- point functionals
    def _trace(self, x: E.Expr, pos: float, d: int) -> dict:
        """Value (d = 0) or radial derivative (d = 1) at r = pos, as surface rows (spin components;
        ball: regularity components, one component per row, the right-hand side being mixed by Q^H
        in rhs_map)."""
        if isinstance(x, E.Component):
            if x.index != 2:
                raise DiscretizationError(f"'{x}': tangential components are not regular on the sphere")
            inner = self._trace(x.args[0], pos, d)
            if self.S.ball and x.args[0].rank == 1:
                return self._ball_radial(inner)
            return {c[1:]: _relabel(e, c[1:]) for c, e in inner.items() if c and c[0] == 2}
        if isinstance(x, E.Differentiate) and x.coord == self.S.r:
            if d:
                raise DiscretizationError(f"'{x}': second radial derivative at a point")
            return self._trace(x.args[0], pos, d + 1)
        if isinstance(x, E.Add):
            parts = [self._trace(t, pos, d) for t in x.args if t.depends_on(self.vars)]
            return {c: _Ent(False, c, None, None, _sum_blocks([p[c] for p in parts if c in p]))
                    for c in {c for p in parts for c in p}}
        if isinstance(x, E.Mul):
            dep = [a for a in x.args if a.depends_on(self.vars)]
            rest = [a for a in x.args if not a.depends_on(self.vars)]
            vals = [self._const(a) for a in rest]
            if len(dep) == 1 and all(v is not None for v in vals):
                s = np.prod(vals) if vals else 1.0
                return {c: e.scale(s) for c, e in self._trace(dep[0], pos, d).items()}
        if isinstance(x, E.Dot) and not x.args[0].depends_on(self.vars) and \
                not self.S.has(x.args[0].domain)[2]:
            return self._trace_dot(x, pos, d)
        inner = self._c(x)
        S = self.S
        out = {}
        for c, e in inner.items():
            if not e.bulk:
                if d:
                    raise DiscretizationError(f"radial derivative of '{x}' (no radial dependence)")
                out[c] = e
                continue
            k0 = spin_of(c)

            def f(ell, no, ni, e=e, k0=k0):
                if S.ball:
                    return sp.csr_matrix(self.ops.point_row(ell + k0, e.level, pos, d, ni)[None, :])
                return sp.csr_matrix(self.ops.point_row(e.level, self._p(e, x), pos, d, ni)[None, :])
            out[c] = e.map(self.bd(c, False, c, True, f), bulk=False, level=None, p=None)
        return out

    def _trace_dot(self, x: E.Dot, pos: float, d: int) -> dict:
        """dot(C, trace) with C a rotation-invariant rank-2 tensor given on the sphere (constant
        spin components, plan.surface_tensor): out_s' = sum_{s,t} C[s', s] G[s, t] X_t. In the
        ball the trace rows are regularity components (spin = Q reg), so the block of degree l
        is Q^H M Q. Used by the curved free-slip condition (model/conditions.py)."""
        C, X = x.args
        if C.rank != 2 or X.rank != 1:
            raise DiscretizationError(f"'{x}': at a boundary, dot(coefficient, vector) needs a "
                                      "rank-2 coefficient and a vector")
        M = self.P.surface_tensor(C) @ self.S.G
        # threshold relative to the tensor's largest entry (review finding 35: an absolute
        # 1e-14 dropped every row of a tensor of size 1e-15)
        scale = float(np.max(np.abs(M), initial=0.0))
        inner = self._trace(X, pos, d)
        ball = self.S.ball
        out = {}
        for co in comps(1, 3):
            terms = []
            for ci, e in inner.items():
                if ball:
                    def f(ell, no, ni, i=co[0], j=ci[0]):
                        Q = self.P.Q(self.m, ell)
                        w = (Q.conj().T @ M @ Q)[i, j]
                        return complex(w) if abs(w) > 1e-14 * scale else None
                    A = self.bd(co, False, ci, False, f)
                    if A.nnz:
                        terms.append(e.map(A, comp=co))
                elif abs(M[co[0], ci[0]]) > 1e-14 * scale:
                    terms.append(_relabel(e, co).scale(M[co[0], ci[0]]))
            if terms:
                out[co] = _Ent(False, co, None, None, _sum_blocks(terms))
        return out

    def _integrate(self, x: E.Integrate) -> dict:
        S = self.S
        if x.args[0].rank:
            raise DiscretizationError(f"'{x}': integral of a tensor on a spherical domain")
        coords = set(x.coords)
        full = {S.phi, S.theta} | (set() if S.s2 else {S.r})
        if coords != full:
            raise DiscretizationError(f"'{x}': on a spherical domain integ() is over the whole domain")
        e = self._c(x.args[0])[()]
        if not e.bulk:
            raise DiscretizationError(f"'{x}': integral of a quantity without the full dependence")
        ang = 2 * np.pi * _SQ2                          # int dOmega Phi_0^{0,0} = 2 pi sqrt(2)

        def f(ell, no, ni):
            if self.m != 0 or ell != 0:
                return None
            if S.s2:
                return ang * S.R ** 2
            if S.ball:
                return sp.csr_matrix(ang * self.ops.integral_row(e.level, ni)[None, :])
            return sp.csr_matrix(ang * self.ops.integral_row(e.level, self._p(e, x), ni)[None, :])
        # one row (the constant), laid out like a point scalar of this group
        return {(): e.map(self.bd((), False, (), True, f), bulk=False, level=None, p=None)}

    def _lift(self, x: E.Lift) -> dict:
        S = self.S
        if S.s2 or x.basis.coord != S.r or x.basis.size != S.N:
            raise DiscretizationError(f"'{x}': the lift basis must be the radial basis of the problem")
        level = int(getattr(x, "lam", 0))
        inner = self._c(x.args[0])
        if any(e.bulk for e in inner.values()):
            raise DiscretizationError(f"'{x}': the lifted quantity depends on r")
        out = {}
        for c, e in inner.items():
            A = self.bd(c, True, c, False,
                        lambda ell, no, ni: sp.csr_matrix(self.ops.lift_col(level, x.n, no)[:, None]))
            # the lifted function is r^-lam C^(lam)_{N+n} on the shell (Dedalus's derivative basis
            # of level lam, held at p = lam: an r^lam-multiplied row takes the bare column) and
            # rho^k Q_{N_l+n}^(lam, k+1/2) in the ball
            out[c] = e.map(A, bulk=True, level=level, p=(level if S.shell else 0))
        return out

    # ---------------------------------------------------------------- products
    def _mul(self, x: E.Mul) -> dict:
        V, ev = self.vars, self.P.eigenvalue
        dep = [a for a in x.args if a.depends_on(V)]
        rest = [a for a in x.args if not a.depends_on(V)]
        if len(dep) != 1:
            raise DiscretizationError(f"'{x}' is not linear in the unknowns")
        has_ev = ev is not None and any(a == ev for a in rest)
        if ev is not None:
            if sum(1 for a in rest if a == ev) > 1 or any(a != ev and ev in a.free_parameters() for a in rest):
                raise DiscretizationError(f"'{x}': eigenvalue must appear linearly")
            if has_ev and ev in dep[0].free_parameters():
                raise DiscretizationError(f"'{x}': eigenvalue must appear linearly")
            if has_ev and self.mode != "M":
                return {}
            rest = [a for a in rest if a != ev]
        # s * (...) : the factor goes to M, compiled as an L-form; c * (s u + ...) (eigenvalue nested
        # in the unknown factor): compiled in the current mode, each term to its own matrix (a
        # nested s u was dropped from M before)
        form = self._c_L(dep[0]) if (self.mode == "M" and has_ev) else self._c(dep[0])
        scalar, nccs = 1.0, []
        for f in rest:
            v = self._const(f)
            if v is not None:
                scalar = scalar * v
            else:
                nccs.append(f)
        out = {c: e.scale(scalar) for c, e in form.items()}
        for f in nccs:
            if f.rank and dep[0].rank:
                raise DiscretizationError(f"'{x}': tensor times tensor, use dot() or cross()")
            out = self._ncc_apply(f, out, lambda a, b: [(a + b, 1.0)])
        return out

    def _product(self, x: E.Expr) -> dict:
        a, b = x.args
        V = self.vars
        if a.depends_on(V) and b.depends_on(V):
            raise DiscretizationError(f"'{x}' is quadratic in the unknowns")
        left_unknown = a.depends_on(V)
        form = self._c(a if left_unknown else b)
        f = b if left_unknown else a
        G = self.S.G
        if isinstance(x, E.Outer):
            def rule(cc, uc):
                return [((uc + cc) if left_unknown else (cc + uc), 1.0)]
        elif isinstance(x, E.Dot):
            def rule(cc, uc):
                if left_unknown:        # dot(u, c): u's last index with c's first
                    w = G[uc[-1], cc[0]]
                    return [(uc[:-1] + cc[1:], w)] if w else []
                w = G[cc[-1], uc[0]]
                return [(cc[:-1] + uc[1:], w)] if w else []
        else:
            tab = cross_table()

            def rule(cc, uc):
                key = (uc[0], cc[0]) if left_unknown else (cc[0], uc[0])
                if key not in tab:
                    return []
                k, cst = tab[key]
                return [((k,), cst)]
        return self._ncc_apply(f, form, rule)

    def _ncc_apply(self, f: E.Expr, form: dict, rule) -> dict:
        """Multiply compiled entries by a non-constant coefficient ``f`` (m = 0): for each coefficient
        component cc (spin s_c) and entry component uc, ``rule(cc, uc)`` gives [(out comp, weight)];
        the block is sum over the coefficient's degrees l_c of the angular coupling (Gaunt integral,
        sphere_ops.SphereHarmonics.coupling) times its radial multiplication."""
        S = self.S
        data = self.P.ncc_data(f)                   # {comp: {l_c: radial coefficients}}
        parts: dict = {}
        for uc, e in form.items():
            if not e.bulk:
                raise DiscretizationError(f"non-constant coefficient '{f}' multiplying a quantity "
                                          "without the full dependence")
            s_in = spin_of(uc)
            for cc, per_l in data.items():
                for oc, w in rule(cc, uc):
                    s_c = spin_of(cc)
                    if S.ball:
                        if cc != () or set(per_l) != {0}:
                            raise DiscretizationError(f"ball: only spherically symmetric scalar "
                                                      f"coefficients are implemented ('{f}')")
                        leg = self.P.ball_ncc_legendre(f)

                        def fb(ell, no, ni, e=e, uc=uc, leg=leg):
                            return self.ops.ncc(leg, ell + spin_of(uc), e.level, ni)
                        A = self.bd(oc, True, uc, True, fb)
                    else:
                        def fcpl(lo, li, no, ni, e=e, s_in=s_in, s_c=s_c, per_l=per_l):
                            acc = None
                            for lc, rad in per_l.items():
                                if abs(lo - li) > lc:
                                    continue
                                cpl = S.H.coupling(self.m, s_in, s_c, lc)[lo, li]
                                if cpl == 0:
                                    continue
                                blk = cpl * (rad if S.s2 else self.ops.ncc(rad, e.level, ni))
                                acc = blk if acc is None else acc + blk
                            return acc
                        if self.g.single and not S.s2:
                            A = self.bd(oc, True, uc, True, lambda ell, no, ni, fcpl=fcpl: fcpl(ell, ell, no, ni))
                        else:
                            A = self.full(oc, True, uc, True, fcpl)
                    if A.nnz:
                        parts.setdefault(oc, []).append(e.map(w * A, comp=oc))
        return {c: self.add(es, c) for c, es in parts.items()}
