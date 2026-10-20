"""Batched right-hand side, batched linear algebra and gather/scatter for ensembles.

An ensemble is a batch of B states X_b (array (B, n)) of *one* discretized problem, advanced
together: the same step arithmetic as :class:`conatus.timestep.IVPSolver`, vectorized over a
leading batch axis. Three pieces, each an explicit, declared choice:

* :class:`BatchedRHS` — F(X_b) for all members at once. The right-hand sides are expanded once
  into the polynomial form of the fused evaluator,

      F_(eq, comp) = sum_m  c_m  C( prod_j  G(D^{a_mj} f_mj) ),

  (or its *general form*, products of linear forms under post-operators, when an RHS holds a
  derivative of a product: the channel/Couette divergence form), and every transform, derivative
  and product carries the batch axis as an extra leading axis (the evaluator's 1D operators take
  the number of leading axes as their ``rank``). Same operations as the fused/linearized paths,
  so the numbers equal ``IVPSolver.F`` (tree evaluator) **to rounding, not bit for bit** (sums of
  monomials are formed in another order); the member-by-member agreement is measured, never
  assumed (tests/ensembles). Per-member parameter values of the RHS become per-member monomial
  coefficients (a (B, 1, ..., 1) array); identical values stay a Python scalar (same arithmetic as
  a single run).
  :meth:`BatchedRHS.jvp` gives F'(X) V for a batch of tangent vectors per state (Lyapunov).
* :class:`BatchedSystem` — M X, L X and the solves (a M + b L) X = R for the whole batch. The
  factors of every member are stacked in **one** batched banded solver per bucket (one C++ call or
  one GPU launch per bucket for the whole ensemble). Members whose L and M are equal (parameters
  that do not enter the LHS) reuse the same factor matrices, *replicated* in the stacked solver:
  memory B x (one factorization), declared by ``explain()``; the per-mode arithmetic is that of
  the single-run solver. Members with their own L (a parameter in the LHS, e.g. Re in the viscous
  term) get their own factors.
* gather/scatter: the index maps of the member's ``GlobalSystem`` applied along the state axis.

Nothing here loops over members in Python except, optionally, :class:`BatchedRHS` in
``mode="loop"`` (the oracle: ``IVPSolver.F`` member by member, bit-identical to single runs).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from ..errors import OptionError, SpectralError, UnsupportedError
from ..ir.bases import Chebyshev
from ..runtime.banded import solve_operand
from ..timestep.fused import _formkey, expand_general, expand_polynomial

__all__ = ["BatchedRHS", "BatchedSystem", "BatchMaps"]


# ---------------------------------------------------------------------------- gather / scatter
def _host(system):
    """Host GlobalSystem of a GlobalSystem or DeviceSystem (the single source of the maps)."""
    return getattr(system, "host", system)


class BatchMaps:
    """Gather/scatter index maps of a ``GlobalSystem`` on the array module ``xp`` (built once),
    applied along the state axis of a batch."""

    def __init__(self, system, xp) -> None:
        h = self.host = _host(system)
        self.xp = xp
        self.cols = [(xp.asarray(sel), xp.asarray(h.cols_flat[sel])) for sel in h._col_sel]
        self.rows = {key: (xp.asarray(sel), xp.asarray(h.rows_flat[sel]))
                     for key, sel in h._row_sel.items()}
        self._conv: dict = {}

    def scatter(self, X, lead: tuple, only: frozenset | None = None) -> list:
        """Batch of global vectors X (lead + (n,)) -> per variable coefficient arrays lead + shape
        (same maps as ``GlobalSystem.scatter``; ``only``: names needed, others None)."""
        h, xp = self.host, self.xp
        X2 = X.reshape((-1, h.n))
        nb = X2.shape[0]
        out = []
        for (sel, flat), v, shape in zip(self.cols, h.variables, h.var_shapes, strict=True):
            if only is not None and v.name not in only:
                out.append(None)
                continue
            a = xp.zeros((nb, int(np.prod(shape))), dtype=h.dtype)
            a[:, flat] = X2[:, sel]
            a = a.reshape(lead + tuple(shape))
            out.append(a if h.complex else a.real)
        return out

    def gather_rhs(self, eq_coeffs: Sequence, nb: int):
        """Equation RHS coefficients ``(nb, *eq_shape)`` -> rows ``(nb, n)``, each row converted to its
        equation's C^(lam) space (same maps and conversion as ``GlobalSystem.gather_rhs``)."""
        h, xp = self.host, self.xp
        b = xp.zeros((nb, h.n), dtype=h.dtype)
        zc = h.plan.layout.zcoord
        coords = h.plan.domain.coordsys.coords
        for (k, lam), (sel, flat) in self.rows.items():
            c = eq_coeffs[k]
            if c is None:
                continue
            c = xp.asarray(c)
            if lam > 0:
                eq = h.plan.problem.equations[k]
                ax = 1 + eq.lhs.rank + coords.index(zc)          # +1: batch axis
                if c.shape[ax] > 1:
                    S = self._conv.get(lam)
                    if S is None:
                        S = h.plan.layout.zconv(lam).tocsr()
                        if xp is not np:
                            from ..timestep.device import DeviceCsr
                            S = DeviceCsr(S, h.dtype)
                        self._conv[lam] = S
                    cm = xp.moveaxis(c, ax, 0)
                    sh = cm.shape
                    c = xp.moveaxis((S @ xp.ascontiguousarray(cm.reshape(sh[0], -1))).reshape(sh),
                                    0, ax)
            b[:, sel] = c.reshape(nb, -1)[:, flat]
        return b


# ---------------------------------------------------------------------------- batched RHS
class BatchedRHS:
    """F(X_b) (and F'(X_b) V) for a batch of states of one discretized problem.

    ``solvers``: one :class:`IVPSolver` per *distinct parameter set* and ``member_of[b]`` the index
    of member b's solver (a single solver and ``member_of=None``: every member shares it).
    ``mode="batched"`` (vectorized, polynomial/general form) or ``"loop"`` (oracle: ``solver.F``
    member by member, bit-identical to single runs, not vectorized)."""

    def __init__(self, solvers: Sequence, member_of: np.ndarray, mode: str = "batched") -> None:
        if mode not in ("batched", "loop"):
            raise OptionError(f"BatchedRHS: mode must be 'batched' or 'loop', got {mode!r}",
                              fix="'batched' (vectorized over members) or 'loop' (oracle)")
        s0 = solvers[0]
        self.solvers, self.member_of = list(solvers), np.asarray(member_of, int)
        self.B = len(self.member_of)
        self.mode = mode
        self.xp, self.system, self.ev = s0.xp, s0.system, s0.evaluator
        self.gpu = self.xp is not np
        self.coords = list(s0.plan.domain.coordsys.coords)
        self.nd = len(self.coords)
        self.bases = [s0.plan.domain.basis(c) for c in self.coords]
        self.variables = list(self.system.variables)
        self._vidx = {v: k for k, v in enumerate(self.variables)}
        self.maps = BatchMaps(self.system, self.xp)
        for s in solvers:
            if s.time_parameter:
                raise UnsupportedError("ensembles: time-dependent right-hand side (time_parameter) "
                                       "not supported yet (explicit refusal)")
        if mode == "loop":
            return
        eqs = s0.plan.problem.equations
        exprs = s0._rhs_exprs
        # same switch as LinearizedRHS / FusedRHS: general form for all equations as soon as one
        # RHS holds a derivative of a product
        general = False
        for e in exprs:
            if e is not None and e.free_fields():
                try:
                    expand_polynomial(e, self.coords, dict(s0.plan.params))
                except SpectralError as err:
                    if "derivative of a product" not in str(err):
                        raise
                    general = True
                    break
        self.form = "general" if general else "polynomial"
        fourier_axes = [i for i, b in enumerate(self.bases) if not isinstance(b, Chebyshev)]
        # per equation: {comp: [(coef (scalar or per-member array), key)]}, keys = factor tuples
        # (polynomial) or (post, forms) (general); the constant RHS (no free field) stays with the
        # evaluator, evaluated once per distinct solver and broadcast
        self.terms: list = []
        self.const_rows = None
        const_needed = False
        for k, (_eq, e) in enumerate(zip(eqs, exprs, strict=True)):
            if e is None:
                self.terms.append(None)
                continue
            if not e.free_fields():
                self.terms.append(None)
                const_needed = True
                continue
            per_solver = []
            for s in solvers:
                # each solver's own expression: its Parameters may carry their own values
                es, p = s._rhs_exprs[k], dict(s.plan.params)
                per_solver.append(expand_general(es, self.coords, p, fourier_axes) if general
                                  else expand_polynomial(es, self.coords, p))
            self.terms.append(self._merge(per_solver, general))
        if const_needed:
            rows = []
            for s in solvers:
                out = [None if (e is None or e.free_fields()) else
                       s.evaluator._embed(s.evaluator.evaluate(e, dict(s._ncc), dict(s.plan.params)),
                                          eq.lhs.rank, shape)
                       for e, eq, shape in zip(s._rhs_exprs, eqs, self.system.eq_shapes, strict=True)]
                rows.append(s.system.gather_rhs(out))
            self.const_rows = self.xp.stack(rows)[self.xp.asarray(self.member_of)] \
                if len(rows) > 1 else rows[0]
        self._ncc = dict(s0._ncc)
        for s in solvers[1:]:
            for f, c in s._ncc.items():
                if not bool(self.xp.all(c == self._ncc[f])):
                    raise UnsupportedError(f"ensembles: non-constant coefficient {f.name} differs "
                                           "between members (not supported yet)")

    # -------------------------------------------------------------- compile
    def _coef(self, vals: list):
        """Scalar if every solver has the same value, else a per-member array (B,) on xp."""
        if all(v == vals[0] for v in vals):
            return vals[0]
        return self.xp.asarray(np.asarray(vals)[self.member_of])

    def _merge(self, per_solver: list, general: bool) -> dict:
        """Union of the monomials/terms of every solver, coefficient per member (0 if absent)."""
        out: dict = {}
        comps = sorted({c for P in per_solver for c in P})
        for comp in comps:
            keys: dict = {}
            for i, P in enumerate(per_solver):
                for t in P.get(comp, []):
                    if general:
                        c, post, forms = t
                        key = (post, tuple(_formkey(f) for f in forms))
                        payload = (post, forms)
                    else:
                        c, fs = t
                        key = tuple((f[0].name, f[1], f[2]) for f in fs)
                        payload = fs
                    ent = keys.setdefault(key, [payload, [0.0] * len(per_solver)])
                    ent[1][i] = ent[1][i] + c
            out[comp] = [(self._coef(v), payload) for payload, v in keys.values()
                         if any(x != 0 for x in v)]
        return out

    # -------------------------------------------------------------- helpers
    def _full(self, c, nl: int):
        """Coefficient array with nl leading axes and one axis per coordinate (size 1 where the
        field does not depend on it: constant along trailing coordinates)."""
        if c.ndim < nl + self.nd:
            c = c.reshape(c.shape + (1,) * (nl + self.nd - c.ndim))
        return c

    def _factor(self, data: dict, f, nl: int):
        """D^a f_comp with nl leading axes (derivatives: the evaluator's own 1D operators)."""
        field, comp, a = f
        c = data[field]
        c = c[(slice(None),) * nl + tuple(comp)] if comp else c
        c = self._full(c, nl)
        for coord, n in zip(self.coords, a, strict=True):
            for _ in range(n):
                c = self.ev._diff(c, coord, nl)
        return c

    def _data(self, X, lead: tuple) -> dict:
        """Coefficient data of the variables (lead + shape) and of the NCC (leading axes of size 1)."""
        nl = len(lead)
        data = {f: self.xp.asarray(c).reshape((1,) * nl + tuple(np.shape(c)))
                for f, c in self._ncc.items()}
        data.update(zip(self.variables, self.maps.scatter(X, lead), strict=True))
        return data

    def _cshape(self, coef, nl: int):
        """Per-member coefficient broadcast against arrays with nl leading axes (member axis first)."""
        if np.isscalar(coef) or getattr(coef, "ndim", 0) == 0:
            return coef
        return coef.reshape(coef.shape + (1,) * (nl - 1 + self.nd))

    def _form(self, data: dict, form, nl: int, full: tuple, linear: bool):
        """Linear form sum cf · D^a f in the full coefficient shape (general form). ``linear``:
        drop the constant and the atoms absent from ``data`` (tangent: NCC atoms have no
        derivative)."""
        out = None
        for cf, a in form:
            if a is None:
                if linear:
                    continue
                t = self.xp.zeros((1,) * nl + full)
                t[(0,) * (nl + self.nd)] = cf
            elif a[0] not in data:
                if linear:
                    continue
                raise KeyError(f"no data for {a[0].name}")
            else:
                c = self._factor(data, a, nl)
                t = cf * self.ev._embed(c, nl, c.shape[:nl] + full)
            out = t if out is None else out + t
        return out

    # -------------------------------------------------------------- evaluation
    def __call__(self, X):
        """F for the batch X (B, n) -> (B, n)."""
        xp = self.xp
        if self.mode == "loop":
            return xp.stack([self.solvers[self.member_of[b]].F(X[b], 0.0) for b in range(self.B)])
        nb = X.shape[0]
        data = self._data(X, (nb,))
        out = self._equations(data, None, nb)
        F = self.maps.gather_rhs(out, nb)
        if self.const_rows is not None:
            F = F + self.const_rows
        return F

    def jvp(self, X, V):
        """F'(X_b) V_bj for states X (Bt, n) and tangent vectors V (Bt, k, n) -> (Bt, k, n).
        Linearization of the same form (polynomial or general), term by term."""
        if self.mode == "loop":
            raise OptionError("BatchedRHS.jvp needs mode='batched'")
        nb, k = V.shape[0], V.shape[1]
        base = self._data(X.reshape(nb, 1, -1), (nb, 1))
        dV = dict(zip(self.variables, self.maps.scatter(V, (nb, k)), strict=True))
        out = self._equations(base, dV, nb * k, lead=(nb, k))
        return self.maps.gather_rhs([None if o is None else o.reshape((nb * k,) + o.shape[2:])
                                     for o in out], nb * k).reshape(nb, k, -1)

    def _many(self, items: dict, nl: int, to: str) -> dict:
        """Transforms of many arrays at once: arrays of equal shape are stacked on an extra leading
        axis and transformed in one call per shape (same 1D transforms line by line)."""
        ev, xp = self.ev, self.xp
        groups: dict = {}
        for key, a in items.items():
            groups.setdefault((a.shape, a.dtype.str), []).append(key)
        out = {}
        for keys in groups.values():
            S = xp.stack([items[k] for k in keys])
            S = ev.to_grid(S, nl + 1) if to == "grid" else ev.to_coeffs(S, nl + 1)
            for i, k in enumerate(keys):
                out[k] = S[i]
        return out

    def _equations(self, data: dict, dV: dict | None, nb: int, lead: tuple | None = None) -> list:
        """Per-equation coefficient arrays lead + eq_shape of F (dV None) or F'·dV.

        Three passes: (1) coefficients of every distinct factor / linear form (base state and, for
        the tangent, the directions), transformed to the grid together; (2) products per term,
        summed per (equation, component, output shape) (polynomial form) or kept per term
        (general form, post-operators act on each projected product); (3) forward transforms of
        all outputs together, then post-operators and embedding."""
        xp, ev = self.xp, self.ev
        lead = (nb,) if lead is None else lead
        nl = len(lead)
        full = tuple(b.size for b in self.bases)
        poly = self.form == "polynomial"
        # (1) factor / form coefficients -> grids
        base_c, dir_c = {}, {}
        for T in self.terms:
            for terms in (T or {}).values():
                for _, payload in terms:
                    facs = payload if poly else payload[1]
                    for _j, f in enumerate(facs):
                        key = f if poly else _formkey(f)
                        if dV is None or len(facs) > 1:
                            if key not in base_c:
                                base_c[key] = (self._factor(data, f, nl) if poly
                                               else self._form(data, f, nl, full, False))
                        if dV is not None and key not in dir_c:
                            if poly:
                                dir_c[key] = (self._factor(dV, f, nl) if f[0] in self._vidx
                                              else None)
                            else:
                                dir_c[key] = self._form(dV, f, nl, full, True)
        grids = self._many(base_c, nl, "grid")
        dgrids = self._many({k: v for k, v in dir_c.items() if v is not None}, nl, "grid")
        # (2) products
        eqs = self.system.plan.problem.equations
        outs: dict = {}                                        # (k, comp, tag) -> grid
        posts: list = []                                       # general: (k, comp, tag, coef, post)
        for k, T in enumerate(self.terms):
            for comp, terms in (T or {}).items():
                for it, (cm, payload) in enumerate(terms):
                    facs = payload if poly else payload[1]
                    keys = [f if poly else _formkey(f) for f in facs]
                    g = self._product(keys, grids, dgrids, dV is not None, nl)
                    if g is None:
                        continue
                    if poly:
                        g = self._cshape(cm, nl) * g
                        tag = (k, comp, g.shape[nl:])
                        outs[tag] = g if tag not in outs else outs[tag] + g
                    else:
                        tag = (k, comp, it)
                        outs[tag] = g
                        posts.append((tag, cm, payload[0]))
        # (3) forward transforms, post-operators, embedding
        coefs = self._many(outs, nl, "coeffs")
        res = []
        for T, shape in zip(self.terms, self.system.eq_shapes, strict=True):
            res.append(None if T is None else
                       xp.zeros(lead + tuple(shape), dtype=np.result_type(self.system.dtype, float)))
        if poly:
            for (k, comp, _), c in coefs.items():
                r0 = eqs[k].lhs.rank
                idx = (slice(None),) * nl + tuple(comp)
                res[k][idx] = res[k][idx] + ev._embed(c, nl, lead + tuple(self.system.eq_shapes[k][r0:]))
            return res
        for tag, cm, (dc, ig) in posts:
            k, comp, _ = tag
            r = coefs[tag]
            for ax, flag in enumerate(ig):                     # integrals (zero mode)
                if flag:
                    r = xp.take(r, xp.asarray([0]), axis=nl + ax) * self.bases[ax].length
            for ax, n in enumerate(dc):                        # then derivatives
                for _ in range(n):
                    r = ev._diff(r, self.coords[ax], nl)
            r0 = eqs[k].lhs.rank
            idx = (slice(None),) * nl + tuple(comp)
            res[k][idx] = res[k][idx] + self._cshape(cm, nl) * ev._embed(
                r, nl, lead + tuple(self.system.eq_shapes[k][r0:]))
        return res

    def _product(self, keys: list, grids: dict, dgrids: dict, tangent: bool, nl: int):
        """Product of the factor grids of one term (F), or its linearization
        sum_j dg_j prod_{i != j} g_i (factors without a direction grid: NCC / constants)."""
        if not keys:                                           # constant term
            return None if tangent else self.xp.ones((1,) * (nl + self.nd))
        if not tangent:
            g = grids[keys[0]]
            for key in keys[1:]:
                g = g * grids[key]
            return g
        tot = None
        for j, key in enumerate(keys):
            if key not in dgrids:
                continue
            term = dgrids[key]
            for i, h in enumerate(keys):
                if i != j:
                    term = term * grids[h]
            tot = term if tot is None else tot + term
        return tot

    def explain(self) -> str:
        if self.mode == "loop":
            return (f"BatchedRHS(loop): IVPSolver.F member by member ({self.B} members; oracle, "
                    "bit-identical to single runs, not vectorized)")
        nt = sum(len(t) for T in self.terms if T for t in T.values())
        per = sum(1 for T in self.terms if T for t in T.values() for c, _ in t
                  if not (np.isscalar(c) or getattr(c, "ndim", 0) == 0))
        return (f"BatchedRHS({self.form} form, {'GPU/CuPy' if self.gpu else 'CPU/NumPy'}): "
                f"{nt} term(s), {per} with per-member coefficients; transforms, derivatives and "
                "products vectorized over a leading member axis (same operations as the fused/"
                "linearized forms; equal to IVPSolver.F to rounding, not bit for bit)"
                + ("; constant RHS rows from the evaluator, broadcast" if self.const_rows is not None
                   else ""))


# ---------------------------------------------------------------------------- batched system
class BatchedSystem:
    """M X, L X and (a M + b L) X = R for a batch (B, n) whose member b uses system
    ``systems[member_of[b]]`` (host GlobalSystem, or DeviceSystem on the GPU)."""

    def __init__(self, systems: Sequence, member_of: np.ndarray) -> None:
        self.systems = list(systems)
        self.member_of = np.asarray(member_of, int)
        self.B = len(self.member_of)
        s0 = self.systems[0]
        self.gpu = hasattr(s0, "host")
        self.xp = s0.cp if self.gpu else np
        self.hosts = [_host(s) for s in self.systems]
        h0 = self.hosts[0]
        self.n, self.dtype = h0.n, h0.dtype
        for h in self.hosts[1:]:
            if h.n != h0.n or len(h._buckets) != len(h0._buckets) or \
                    len(h._cbuckets) != len(h0._cbuckets) or len(h._zbuckets) != len(h0._zbuckets):
                raise OptionError("ensemble members must share the discretization (same unknowns, "
                                  "same groups): only parameter values may differ")
            for (_i0, r0, c0), (_i1, r1, c1) in zip(h0._buckets, h._buckets, strict=True):
                if not (np.array_equal(r0, r1) and np.array_equal(c0, c1)):
                    raise OptionError("ensemble members: different group orderings")
        xp = self.xp
        self._buckets = [(idx, xp.asarray(rows), xp.asarray(cols)) for idx, rows, cols in h0._buckets]
        self._cbuckets = [(idx, *(xp.asarray(a) for a in arrs)) for idx, *arrs in h0._cbuckets]
        self._zbuckets = [(keys, xp.asarray(rows), xp.asarray(cols), [complex(w) for w in wr],
                           [complex(w) for w in wc]) for keys, rows, cols, wr, wc in h0._zbuckets]
        # M, L: one shared matrix (all members equal) or a block-diagonal matrix over the batch
        same = all(self._same(h, h0) for h in self.hosts[1:])
        self.shared = same
        if same:
            self._M, self._L = s0.M, s0.L
        else:
            Ms = [self.hosts[i].M for i in self.member_of]
            Ls = [self.hosts[i].L for i in self.member_of]
            M = sp.block_diag(Ms, format="csr").astype(self.dtype)
            L = sp.block_diag(Ls, format="csr").astype(self.dtype)
            if self.gpu:
                from ..timestep.device import DeviceCsr
                M, L = DeviceCsr(M, self.dtype), DeviceCsr(L, self.dtype)
            self._M, self._L = M, L
        from ..timestep.system import FactorCache
        self._factors = FactorCache()      # bounded LRU (audit state)

    @staticmethod
    def _same(a, b) -> bool:
        return (a.M.shape == b.M.shape and (a.M != b.M).nnz == 0 and (a.L != b.L).nnz == 0)

    # -------------------------------------------------------------- products
    def _apply(self, A, X):
        """A X_b for every member (X: (B, n) or (..., n) flattened in member-major order)."""
        xp = self.xp
        sh = X.shape
        X2 = X.reshape(-1, self.n)
        if self.shared:
            # (A @ X2ᵀ)ᵀ: row sums in the same order as the single-vector product
            if self.gpu:
                out = (A @ xp.ascontiguousarray(X2.T)).T
            else:
                out = (A @ X2.T).T
            return xp.ascontiguousarray(out).reshape(sh)
        if X2.shape[0] != self.B:
            # tangent batches (B, k, n): block-diagonal over members, k vectors each
            k = X2.shape[0] // self.B
            Xk = xp.ascontiguousarray(X.reshape(self.B, k, self.n).transpose(0, 2, 1)).reshape(-1, k)
            out = (A @ Xk).reshape(self.B, self.n, k).transpose(0, 2, 1)
            return xp.ascontiguousarray(out).reshape(sh)
        return (A @ X2.reshape(-1)).reshape(sh)

    def M(self, X):
        return self._apply(self._M, X)

    def L(self, X):
        return self._apply(self._L, X)

    # -------------------------------------------------------------- factors
    def _mats(self, h, a: float, b: float) -> list:
        """Per bucket: the list of (a M + b L) group matrices of host system h, in the
        permutations of GlobalSystem.factor (real, complexified, multi-complex buckets)."""
        out = []
        for idx, _, _ in h._buckets:
            out.append([((a * h.slots[i].M + b * h.slots[i].L).tocsr()[h.slots[i].rperm]
                         [:, h.slots[i].cperm]).astype(h.dtype) for i in idx])
        for idx, *_ in h._cbuckets:
            czs = [h.slots[i].cz for i in idx]
            out.append([(a * cz.M + b * cz.L).tocsr()[cz.rperm][:, cz.cperm].astype(np.complex128)
                        for cz in czs])
        for keys, *_ in h._zbuckets:
            vs = [h.slots[i].zs[vi] for i, vi in keys]
            out.append([(a * v.M + b * v.L).tocsr()[v.rperm][:, v.cperm].astype(np.complex128)
                        for v in vs])
        return out

    def _colborders(self, h) -> list:
        """Column border flag per bucket (spherical shells, timestep/colborder.py; else False)."""
        return ([bool(getattr(h.slots[idx[0]], "col_border", 0)) for idx, _, _ in h._buckets]
                + [False] * (len(h._cbuckets) + len(h._zbuckets)))

    def _borders(self, h) -> list:
        return ([h.slots[idx[0]].n_border for idx, _, _ in h._buckets]
                + [h.slots[idx[0]].cz.n_border for idx, *_ in h._cbuckets]
                + [h.slots[keys[0][0]].zs[keys[0][1]].n_border for keys, *_ in h._zbuckets])

    def factor(self, a: float, b: float, reps: int = 1):
        """Stacked factors for the batch (member-major, each member's groups in bucket order),
        ``reps`` copies per member (tangent batches: k vectors per state). Cached by (a, b, reps)."""
        key = (float(a), float(b), int(reps))
        f = self._factors.get(key)
        if f is not None:
            return f
        per_sys = [self._mats(h, a, b) for h in self.hosts]
        borders = self._borders(self.hosts[0])
        colb = self._colborders(self.hosts[0])
        nr = len(self._buckets)
        f = []
        for j, nbord in enumerate(borders):
            mats = [m for i in self.member_of for _ in range(reps) for m in per_sys[i][j]]
            dt = self.dtype if j < nr else np.complex128
            if self.gpu:
                from ..runtime import cuda
                f.append(cuda.BandedBatch(mats, list(range(nbord)), pivot=self.systems[0].pivot,
                                          dtype=dt, col_border=colb[j]))
            else:
                from ..runtime import BorderedBandedBatch
                f.append(BorderedBandedBatch(mats, list(range(nbord)), dtype=dt, col_border=colb[j]))
        self._factors[key] = f
        return f

    def _bsolve(self, batch, Z):
        """Solve the stacked systems in place (GPU) or return the solution (CPU); Z (nb, P, m)."""
        nb, P, m = Z.shape
        if self.gpu:
            cp = self.xp
            Z = solve_operand(Z.reshape(nb * P, m), batch.dtype, "Ensemble solve (GPU)", cp)
            batch._s.solve_ptr(int(Z.data.ptr), int(Z.shape[0]), int(Z.shape[1]),
                               int(cp.cuda.get_current_stream().ptr))
            return Z.reshape(nb, P, m)
        return batch.solve(np.ascontiguousarray(Z.reshape(nb * P, m))).reshape(nb, P, m)

    def solve(self, a: float, b: float, rhs):
        """X with (a M_b + b L_b) X_b = rhs_b for every member; rhs (B, n) or (B, k, n) (k tangent
        vectors per member, factors replicated k times). Mirrors ``GlobalSystem.solve``."""
        xp = self.xp
        sh = rhs.shape
        reps = 1 if rhs.ndim == 2 else sh[1]
        R = rhs.reshape(-1, self.n)
        nb = R.shape[0]
        f = self.factor(a, b, reps)
        x = (xp.zeros if self._zbuckets else xp.empty)((nb, self.n), dtype=self.dtype)
        nr, nc = len(self._buckets), len(self._cbuckets)
        for (_idx, rows, cols), batch in zip(self._buckets, f[:nr], strict=True):
            x[:, cols] = self._bsolve(batch, R[:, rows])
        for (_idx, rc, rs, cc, cs), batch in zip(self._cbuckets, f[nr:nr + nc], strict=True):
            z = self._bsolve(batch, R[:, rc] + 1j * R[:, rs])
            x[:, cc] = z.real
            x[:, cs] = z.imag
        for (_keys, rows, cols, wr, wc), batch in zip(self._zbuckets, f[nr + nc:], strict=True):
            z = self._bsolve(batch, sum(w * R[:, r] for w, r in zip(wr, rows, strict=True)).astype(np.complex128))
            for w, c in zip(wc, cols, strict=True):
                x[:, c] += (w * z).real
        return x.reshape(sh)

    def explain(self) -> str:
        nf = len({k[:2] for k in self._factors})
        return (f"BatchedSystem: {self.B} member(s), n = {self.n}, "
                + ("M and L shared by all members (one sparse product over the batch); "
                   if self.shared else f"{len(self.systems)} distinct (M, L) (block-diagonal "
                   "products over the batch); ")
                + f"{'GPU' if self.gpu else 'CPU'} stacked bordered-banded solves, one call per "
                  f"bucket for the whole batch; factors of each member stacked (members with the "
                  f"same L share the same matrices, replicated in the stack: memory B x one "
                  f"factorization); {nf} (a, b) pair(s) factored")
