"""Exact tangent F'(X) V and adjoint F'(X)† W of the spherical-shell right-hand side.

The shell IVP state X is complex (spin components per stored azimuthal wavenumber m >= 0, the
modes -m of a real field being their complex conjugates, ``sphere_eval``), so the discrete maps
are only **real-linear**: the synthesis to the grid conjugates the stored modes to build the
modes -m. The adjoint is therefore taken for the **real pairing** of C^n viewed as R^2n,

    <a, b> = Re sum_i conj(a_i) b_i ,

under which a complex-linear map A has the adjoint A^H (conjugate transpose) and the
conjugation is its own adjoint. ``FlowMap`` uses the same pairing for the linear solves and the
mass/stiffness products of a spherical solver (declared by ``FlowMap.explain()``).

Tangent: the IR linearization of every right-hand side, le = dF/dX[dX] (``Expr.linearize``),
evaluated by the solver's own :class:`SphereEvaluator` (same grids, same r^k projection, same
time parameter): the derivative of the *discrete* F, exact up to rounding (the products are
formed on the same dealiased grid, the sums are linear). On a solver with f_backend='fused' the
tangent and the passive values of the adjoint walk are evaluated by the planned evaluator
(``timestep/sphere_fused``): the derivative of the fused F, real-field directions (B2).

Adjoint: the same expression walked backwards. Every evaluator stage used by le is linear in the
active field once the passive factors (the state X, the coefficients, a_g(t)) are fixed, and its
explicit adjoint is written below next to the forward formula it transposes:

* radial synthesis / projection on the shell (Chebyshev values, d/dr, 1/r at the Gauss nodes;
  Gauss-Chebyshev projection): right products by fixed matrices B -> products by B^H;
* angular synthesis / projection (spin-weighted Gauss-Legendre, one matrix per (m, s)) -> B^H;
* azimuthal synthesis (inverse FFT of the stored modes plus the conjugates of the flipped spins
  for -m) -> FFT, the conjugate part sent back to the flipped spin; projection (FFT / M_phi,
  stored modes kept) -> inverse FFT of the zero-padded modes;
* spin-basis gradient (diagonal G_+- in l, frame terms 1/r), divergence, curl, Laplacian,
  d/dphi, products, dot, cross, outer, components and conjugation -> their adjoints;
* the r^k F projection of the shell rows (k = rhs_rpow) -> conj(r^k) on the grid;
* the equation rows (``SphereGlobalSystem.gather_rhs``, one sparse map per equation) -> R^H;
  the scatter of the state into coefficient arrays -> the gather of the same indices.

Each transposed stage is checked by a dot-product test in ``tests/adjoint/test_sphere_adjoint.py``.
Time dependence (``time_parameter``): the passive values (base flow U = sum_g a_g(t) U_g with the
a_g evaluated at the stage time, Coriolis NCC, ...) are evaluated at the stage time t given by
FlowMap, as the forward F; no split is needed because a_g(t) enters as a passive scalar factor.
Refused (clear error): the ball and S2 (regularity components / other transforms: not written),
symmetry-reduced solvers (folded grids), pointwise functions of the state in le.
"""
from __future__ import annotations

import numpy as np

from ..discretize._xp import array_module
from ..discretize.sphere_eval import _R2, _C, _G, _X, _flip, comps, embed_coeffs
from ..discretize.sphere_ops import cross_table, spin_of
from ..ir import expr as E
from ..timestep.spectral import SpectralError

__all__ = ["SphereLinearizedRHS", "SphereTransposes", "unbroadcast"]


def unbroadcast(a, shape: tuple):
    """Adjoint of broadcasting an array of ``shape`` to ``a.shape``: sum over the added leading
    axes and over the axes of length 1 in ``shape``."""
    xp = array_module(a)
    shape = tuple(shape)
    if a.shape == shape:
        return a
    if a.ndim > len(shape):
        a = a.sum(axis=tuple(range(a.ndim - len(shape))))
    axes = tuple(i for i, (sa, s) in enumerate(zip(a.shape, shape)) if s == 1 and sa != 1)
    if axes:
        a = a.sum(axis=axes, keepdims=True)
    return xp.ascontiguousarray(a).reshape(shape)


# ----------------------------------------------------------------------------------------------
# adjoints of the SphereSpace transforms (shell, no fold), real pairing
# ----------------------------------------------------------------------------------------------
class SphereTransposes:
    """Adjoints (real pairing) of the four stages of ``SphereSpace`` used by the evaluator:
    coeff_to_mixed, mixed_to_grid, grid_to_mixed, mixed_to_coeff (``physical=False``, dealiased
    grid, no fold). The forward matrices are the space's own cached ones; their conjugate
    transposes are cached here once per (stage, m-set, spin, size)."""

    def __init__(self, space) -> None:
        if not space.shell:
            raise SpectralError("spherical adjoint: the spherical shell only for now (the ball's "
                                "regularity components and S2 have no transposed transforms yet)")
        self.S = space
        self._cache: dict = {}
        # the transform matrices are real: products by them (and by their transposes) are real
        # GEMMs on the interleaved (re, im) data, the angular ones restricted to the existing
        # degrees per m-chunk, as in the planned F (timestep/sphere_fused; B2)
        from ..timestep.sphere_fused import default_ang_chunks
        self.nchunk = default_ang_chunks(space, None)    # CPU rule; the GPU rule set by the caller

    def _ang_apply(self, which: str, H, y, ms, s: int, xp):
        """H y for a transposed real angular stack H: 'bwd' = synthesis-like (nm, Mt, L+1) with
        zero columns l < max(|m|, |s|), 'fwd' = projection-like (nm, L+1, Mt) with zero rows."""
        from ..timestep.sphere_fused import _ang_real, ang_chunks
        key = ("chunks", tuple(int(m) for m in ms), s)
        if key not in self._cache:
            self._cache[key] = ang_chunks(np.asarray(ms), s, self.S.L, self.nchunk)
        return _ang_real(which, H, y, self._cache[key], xp)

    def _H(self, key, build):
        """Conjugate transpose of a forward matrix (built once, kept in the module of use)."""
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    # -------------------------------------------------------- radial (Chebyshev on the shell)
    def coeff_to_mixed_T(self, y, dom, rank: int, what: str = "val"):
        """Adjoint of ``coeff_to_mixed(c, dom, rank, True, what)``: y = c @ B  ->  c̄ = ȳ @ B^H."""
        S = self.S
        xp = array_module(y)
        hp, ht, hr = S.has(dom)
        out = xp.zeros(S.coeff_shape(dom, rank), dtype=complex)
        if not hr:
            if what != "val":
                raise SpectralError("radial derivative of a quantity without r-dependence")
            return out + y                                    # identity (Mr = 1)
        Mr = S.sizes(True)[2]
        BH = self._H(("bwd", what, Mr, xp is np),
                     lambda: xp.ascontiguousarray(xp.asarray(S._shell_mat("bwd", what, Mr)).T))
        from ..timestep.sphere_fused import _right_real
        out[...] = _right_real(xp.asarray(y, dtype=complex), BH, xp)    # real matrix: B^H = B^T
        return out

    def mixed_to_coeff_T(self, z, dom, rank: int):
        """Adjoint of ``mixed_to_coeff(x, dom, rank, True)``: z = x @ F  ->  x̄ = z̄ @ F^H."""
        S = self.S
        xp = array_module(z)
        hp, ht, hr = S.has(dom)
        if not hr:
            return xp.array(z, dtype=complex)
        Mr = S.sizes(True)[2]
        FH = self._H(("fwd", Mr, xp is np),
                     lambda: xp.ascontiguousarray(xp.asarray(S._shell_mat("fwd", "val", Mr)).T))
        from ..timestep.sphere_fused import _right_real
        return _right_real(xp.asarray(z, dtype=complex), FH, xp)        # real matrix: F^H = F^T

    # -------------------------------------------------------- angular + azimuthal
    def _ang_H(self, which: str, ms, s: int, Mt: int, xp):
        """Conjugate transpose of the stacked angular synthesis ('bwd', (nm, Mt, L+1)) or
        projection ('fwd', (nm, L+1, Mt)) matrices of spin s."""
        key = ("ang", which, tuple(int(m) for m in ms), s, Mt, xp is np)
        return self._H(key, lambda: xp.ascontiguousarray(
            xp.swapaxes(self.S._ang_stack(which, ms, s, Mt, xp), 1, 2).conj()))

    def mixed_to_grid_T(self, gbar, dom, rank: int):
        """Adjoint of ``mixed_to_grid(x, dom, rank, True, physical=False)``.

        Forward, per spin component: angular synthesis y = B_(m, s) x (or the l = 0 value
        x / sqrt(2) without theta), then S[m] = y_comp, S[-m] = conj(y_flip(comp)[m]) (m > 0,
        real fields), g = M_phi ifft(S). Adjoint: S̄ = fft(ḡ); ȳ_comp[m] += S̄[m] and
        ȳ_flip[m] += conj(S̄[-m]) (m > 0); x̄ = B^H ȳ."""
        S = self.S
        xp = array_module(gbar)
        hp, ht, hr = S.has(dom)
        Mt = S.sizes(True)[1]
        ms = S._m_of(dom)
        if hp and S.real and ht and len(S.ms) > 1 and np.array_equal(S.ms, np.arange(len(S.ms))):
            return self._mixed_to_grid_T_real(gbar, rank, ms, Mt, xp)
        ybar = {c: None for c in comps(rank, S.nc)}

        def add(c, v):
            ybar[c] = v if ybar[c] is None else ybar[c] + v
        for c in comps(rank, S.nc):
            gb = gbar[c]
            if hp:
                Mp = gb.shape[0]
                Sb = xp.fft.fft(gb, axis=0)
                if S.real:
                    add(c, Sb[S.ms])
                    part = xp.zeros_like(Sb[S.ms])
                    part[1:] = xp.conj(Sb[(-S.ms[1:]) % Mp])
                    add(_flip(c), part)
                else:
                    add(c, Sb[S.ms % Mp])
            else:
                add(c, gb)
        Mr = gbar.shape[-1]
        out = xp.zeros((S.nc,) * rank + (len(ms), S.L + 1 if ht else 1, Mr), dtype=complex)
        for c in comps(rank, S.nc):
            y = ybar[c]
            s = spin_of(c)
            if ht:                       # B^H: projection-like (zero rows l < max(|m|, |s|))
                out[c] = self._ang_apply("fwd", self._ang_H("bwd", ms, s, Mt, xp), y, ms, s, xp)
            else:
                out[c] = y * _R2 if s == 0 else 0 * y
        return out

    def _mixed_to_grid_T_real(self, gbar, rank: int, ms, Mt: int, xp):
        """``mixed_to_grid_T`` for RealFourier (stored m = 0 .. nm-1), all components at once:
        phi moved last (one copy), one batched FFT, then ybar_c = S̄_c[m] + [0, conj(S̄_flip(c)[-m])]
        (the same two terms as the per-component code; their sum is order-independent)."""
        S = self.S
        nc, nm = S.nc, len(S.ms)
        Mp = gbar.shape[rank]
        G = xp.ascontiguousarray(xp.moveaxis(xp.asarray(gbar, dtype=complex), rank, -1))
        Sb = xp.fft.fft(G, axis=-1)                                   # (comps..., Mt, Mr, Mp)
        Y = xp.ascontiguousarray(Sb[..., :nm])                        # modes m >= 0
        neg = Sb[..., Mp - 1:Mp - nm:-1]                              # modes -1 .. -(nm-1)
        perm = [_flip((i,))[0] for i in range(nc)]                    # spin flip per index
        for ax in range(rank):
            neg = xp.take(neg, xp.asarray(perm), axis=ax)
        Y[..., 1:] += xp.conj(neg)
        out = xp.zeros((nc,) * rank + (nm, S.L + 1, Sb.shape[-2]), dtype=complex)
        for c in comps(rank, nc):
            y = xp.moveaxis(Y[c] if c else Y, -1, 0)                 # (nm, Mt, Mr)
            s = spin_of(c)
            v = self._ang_apply("fwd", self._ang_H("bwd", ms, s, Mt, xp), y, ms, s, xp)
            if c:
                out[c] = v
            else:
                out[...] = v
        return out

    def grid_to_mixed_T(self, xbar, dom, rank: int):
        """Adjoint of ``grid_to_mixed(g, dom, rank, True, physical=False)``.

        Forward, per component: y = (fft(g) / M_phi)[stored m], x = F_(m, s) y (or y sqrt(2) at
        l = 0 without theta). Adjoint: ȳ = F^H x̄, ḡ = ifft(zero-padded ȳ) (the adjoint of
        fft / M_phi is ifft)."""
        S = self.S
        xp = array_module(xbar)
        hp, ht, hr = S.has(dom)
        M = S.sizes(True)
        Mp, Mt = M[0], M[1]
        ms = S._m_of(dom)
        if hp and S.real and ht and len(S.ms) > 1 and np.array_equal(S.ms, np.arange(len(S.ms))):
            # all components at once: zero-padded modes phi-last, one batched inverse FFT
            nm = len(S.ms)
            Sb = None
            for c in comps(rank, S.nc):
                s = spin_of(c)
                y = self._ang_apply("bwd", self._ang_H("fwd", ms, s, Mt, xp), xbar[c], ms, s, xp)
                if Sb is None:
                    Sb = xp.zeros((S.nc,) * rank + y.shape[1:] + (Mp,), dtype=complex)
                (Sb[c] if c else Sb)[..., :nm] = xp.moveaxis(y, 0, -1)
            g = xp.fft.ifft(Sb, axis=-1)
            return xp.ascontiguousarray(xp.moveaxis(g, -1, rank))
        out = xp.zeros(S.grid_shape(dom, rank, True), dtype=complex)
        for c in comps(rank, S.nc):
            s = spin_of(c)
            y = xbar[c]
            if ht:                       # F^H: synthesis-like (zero columns l < max(|m|, |s|))
                y = self._ang_apply("bwd", self._ang_H("fwd", ms, s, Mt, xp), y, ms, s, xp)
            else:
                y = y / _R2 if s == 0 else 0 * y
            if hp:
                Sb = xp.zeros((Mp,) + y.shape[1:], dtype=complex)
                Sb[S.ms % Mp] = y
                y = xp.fft.ifft(Sb, axis=0)
            out[c] = y
        return out

    # -------------------------------------------------------- full transforms
    def backward_T(self, gbar, dom, rank: int):
        """Adjoint of ``backward(c, dom, rank, True, physical=False)`` (coefficients -> grid)."""
        return self.coeff_to_mixed_T(self.mixed_to_grid_T(gbar, dom, rank), dom, rank, "val")

    def forward_T(self, cbar, dom, rank: int):
        """Adjoint of ``forward(g, dom, rank, True, physical=False)`` (grid -> coefficients)."""
        return self.grid_to_mixed_T(self.mixed_to_coeff_T(cbar, dom, rank), dom, rank)


# ----------------------------------------------------------------------------------------------
# linearized right-hand side and its adjoint
# ----------------------------------------------------------------------------------------------
class _Meta:
    """Structure of the evaluator value of an active node (state-independent): representation
    ('C' coefficients, 'X' mixed space, 'G' dealiased grid), domain, rank, array shape and the
    shape of its grid (``_grid``)."""
    __slots__ = ("kind", "dom", "rank", "shape", "gshape")

    def __init__(self, kind, dom, rank, shape, gshape):
        self.kind, self.dom, self.rank, self.shape, self.gshape = kind, dom, rank, shape, gshape


class SphereLinearizedRHS:
    """F'(X, t) V (``apply``) and F'(X, t)† W (``transpose``, real pairing) of the right-hand
    sides of a spherical-shell ``IVPSolver`` (CPU NumPy or GPU CuPy: same code). Module doc."""

    pairing = "real"

    def __init__(self, solver, planned: bool = True) -> None:
        s = solver
        self.solver, self.system, self.ev = s, s.system, s.evaluator
        self.S = self.ev.S
        self.xp = s.xp
        self.gpu = self.xp is not np
        if getattr(self.ev, "fold", None) is not None:
            raise SpectralError(
                "spherical adjoint: symmetry-reduced solver (folded grid) not supported yet; build "
                "the IVPSolver with reduced=False (the full space; the symmetric subspace is then "
                "invariant under the flow map and its adjoint)")
        self.T = SphereTransposes(self.S)               # refuses the ball and S2 (clear message)
        from ..timestep.sphere_fused import default_ang_chunks
        self.T.nchunk = default_ang_chunks(self.S, self.xp)   # same rule as the planned F
        self.variables = list(self.system.variables)
        self.tname = getattr(s, "time_parameter", None)
        self.dvars = {v: E.Field("d" + v.name, v.domain, v.tensorsig, v.dtype) for v in self.variables}
        self._dset = set(self.dvars.values())
        self._dname = {d: v for v, d in self.dvars.items()}
        direction = {v: d for v, d in self.dvars.items()}
        self.lin: list = []
        for e in s._rhs_exprs:
            if e is None or not (set(e.free_fields()) & set(self.variables)):
                self.lin.append(None)                   # constant RHS (BCs, forcing): F' = 0
                continue
            le = e.linearize(None, direction)
            self.lin.append(None if le.is_zero() else le)
        self.rpow = list(s.plan.rhs_rpow)
        self._lin_vars = frozenset(f.name for le in self.lin if le is not None
                                   for f in le.free_fields() if f not in self._dset)
        # names of the variables whose direction enters some le (scatter only those)
        self._lin_dsrc = frozenset(self._dname[f].name for le in self.lin if le is not None
                                   for f in le.free_fields() if f in self._dset)
        # adjoint of gather_rhs: R_i^H per equation (host CSR, or device CSR on the GPU)
        host = self.system.host if self.gpu else self.system
        self.host = host
        if getattr(host, "reduced", False):
            raise SpectralError("spherical adjoint: symmetry-reduced system not supported yet "
                                "(reduced=False)")
        self._RH = []
        for R, le in zip(host._rhs_maps, self.lin):
            if le is None or R.nnz == 0:
                self._RH.append(None)
                continue
            RH = R.conj().T.tocsr()
            if self.gpu:
                from ..timestep.device import DeviceCsr
                RH = DeviceCsr(RH, host.dtype)
            self._RH.append(RH)
        asx = self.xp.asarray
        self._cols = [(asx(sel), asx(host.cols_flat[sel])) for sel in host._col_sel]
        self._meta: dict = {}
        # f_backend='fused' (B2): the tangent is the planned evaluation of the same linearized
        # expressions (common sub-expressions, cached constants, real-field transforms, one
        # product kernel): the derivative of the fused F itself. Like that F it uses the reality
        # of the fields, so directions must be real fields (SphereAdmissibleProjector); on them it
        # equals the reference linearization to rounding. The adjoint stays the explicit one.
        self._fused = None
        if getattr(s, "_sfused", None) is not None:
            from ..timestep.sphere_fused import _check_real, _FusedEvaluator, distribute_derivatives
            state = frozenset(self.variables) | frozenset(self._dset)
            lin2 = []
            for i, le in enumerate(self.lin):
                if le is None:
                    lin2.append(None)
                    continue
                _check_real(le, f"linearized right-hand side of equation {i}")
                lin2.append(distribute_derivatives(le, state)[0])
            self._fused = (_FusedEvaluator(s.plan.space, s.xp, state), lin2)
            # passive values of the adjoint walk (grid values of u, grad u, U, ...): the same
            # planned evaluator (fresh memo per value; constants cached)
            self._fused_passive = _FusedEvaluator(s.plan.space, s.xp, frozenset(self.variables))
            # The planned F reads its inputs through real-field transforms, i.e. it evaluates
            # F(P X) with P the projection of the m = 0 coefficients onto real fields (Re of the
            # grid values, timestep/real_projection.py); its tangent is F'(P X) P V. The explicit
            # adjoint below gives F'(P X)^dagger W (passive values from the planned evaluator), so
            # the exact adjoint of the planned tangent is P F'(P X)^dagger W (P is self-adjoint
            # for the real pairing). Without this P the pair was off by 2.5 % on directions with
            # a non-real m = 0 part (gpu1: reproduced on CPU).
            from ..timestep.real_projection import M0RealProjection
            pr = getattr(s, "_m0proj", None)
            pr = pr if getattr(pr, "active", False) else M0RealProjection(self.system)
            if not pr.active:
                raise SpectralError("spherical adjoint with f_backend='fused': the real projection "
                                    f"of the planned F could not be built ({pr.reason}); use "
                                    "f_backend='evaluator' for the adjoint")
            self._real_P = pr
        # planned adjoint (feat2): on fused solvers the transpose follows the planned path of the
        # tangent stage by stage (adjoint/sphere_planned.py); equations outside its scope keep the
        # explicit walk below. planned=False keeps the explicit walk everywhere (the oracle).
        self._planned = None
        self._planned_note = "explicit walk (f_backend='evaluator')" if self._fused is None else \
            "explicit walk (planned=False)"
        if self._fused is not None and planned:
            from .sphere_planned import PlannedSphereTranspose, check_scope
            why = check_scope(self)
            if why is None:
                self._planned = PlannedSphereTranspose(self)
                self._planned_note = self._planned.explain()
            else:
                self._planned_note = f"explicit walk ({why})"

    # ------------------------------------------------------------------ data
    def _params(self, t):
        params = dict(self.solver.plan.params)
        if self.tname:
            if t is None:
                raise SpectralError(f"time-dependent F ({self.tname}): the stage time is required")
            params[self.tname] = t
        return params

    def _state_data(self, X) -> dict:
        data = dict(self.solver._ncc)
        data.update((v, c) for v, c in zip(self.variables, self.system.scatter(X, only=self._lin_vars))
                    if c is not None)
        return data

    def apply(self, X, V, t=None):
        """F'(X, t) V: the linearized RHS evaluated by the solver's evaluator (row vector)."""
        data = self._state_data(X)
        data.update((self.dvars[v], c) for v, c in
                    zip(self.variables, self.system.scatter(V, only=self._lin_dsrc)) if c is not None)
        params = self._params(t)
        if self._fused is not None:                       # planned tangent (f_backend='fused')
            fev, lin2 = self._fused
            fev.begin()
            out = [None if le is None else embed_coeffs(fev.evaluate(le, data, params, k), shape)
                   for le, k, shape in zip(lin2, self.rpow, self.host.eq_shapes)]
            fev.begin()
            return self.system.gather_rhs(out)
        out = [None if le is None else embed_coeffs(self.ev.evaluate(le, data, params, k), shape)
               for le, k, shape in zip(self.lin, self.rpow, self.host.eq_shapes)]
        return self.system.gather_rhs(out)

    # ------------------------------------------------------------------ structure of active nodes
    def _active(self, x) -> bool:
        return bool(set(x.free_fields()) & self._dset)

    def _meta_of(self, x, data, params) -> _Meta:
        m = self._meta.get(id(x))
        if m is None:
            v = self.ev._ev(x, data, params)
            if isinstance(v, _C):
                m = _Meta("C", v.dom, v.rank, v.a.shape, self.S.grid_shape(v.dom, v.rank, True))
            elif isinstance(v, _X):
                m = _Meta("X", v.dom, v.rank, v.x.shape, self.S.grid_shape(v.dom, v.rank, True))
            elif isinstance(v, _G):
                m = _Meta("G", v.dom, v.rank, v.g.shape, v.g.shape)
            else:
                raise SpectralError(f"spherical adjoint: node '{x}' depends on the state but "
                                    "evaluates to a number")
            self._meta[id(x)] = (x, m)              # keep x alive: its id cannot be reused
            return m
        return m[1]

    def _pgrid(self, x, ctx):
        """Passive value of node x as a grid array (or a number), cached for this state/time."""
        key = id(x)
        hit = ctx["vals"].get(key)
        if hit is None:
            # planned evaluator on fused solvers: its memo (common sub-expressions) is reset at
            # the start and the end of each transpose() call (one state, one time)
            ev = self.ev if self._fused is None else self._fused_passive
            v = ev._ev(x, ctx["data"], ctx["params"])
            hit = v if not isinstance(v, (_C, _X, _G)) else ev._grid(v).g
            ctx["vals"][key] = hit
        return hit

    # ------------------------------------------------------------------ conversion adjoints
    def _grid_T(self, m: _Meta, gbar):
        """Adjoint of ``_grid`` from the representation of m: grid cotangent -> m's cotangent."""
        if m.kind == "G":
            return gbar
        if m.kind == "X":
            return self.T.mixed_to_grid_T(gbar, m.dom, m.rank)
        return self.T.backward_T(gbar, m.dom, m.rank)

    def _coeff_T(self, m: _Meta, cbar):
        """Adjoint of ``_coeff`` (to level-0 coefficients) from the representation of m."""
        if m.kind == "C":
            return cbar
        if m.kind == "X":
            return self.T.mixed_to_coeff_T(cbar, m.dom, m.rank)
        full = self.T.forward_T(cbar, m.dom, m.rank)
        return unbroadcast(full, m.shape)

    # ------------------------------------------------------------------ derivative adjoints
    def _grad_T(self, gbar, dom, rank: int):
        """Adjoint of ``SphereEvaluator._grad`` (3D): cotangent of the spin-basis gradient (rank+1,
        mixed space) -> cotangent of the level-0 coefficients of the operand.

        Forward (sphere_ops module doc): (grad u)_(-1, c) += G_- u_c / r, (grad u)_(+1, c) +=
        G_+ u_c / r, (grad u)_(r, c) += d_r u_c, and the frame terms of every index of c
        (+ u_r / r on the (+, -) and (-, +) slots, - u_(+-) / r on the (+-, r) slots)."""
        S = self.S
        xp = array_module(gbar)
        ms = S._m_of(dom)
        shape = gbar.shape[1:]
        Ab = xp.zeros(shape, dtype=complex)
        Db = xp.zeros(shape, dtype=complex)
        for c in comps(rank, S.nc):
            s = spin_of(c)
            L1 = shape[rank + 1]
            for d, sigma in ((0, -1), (1, 1)):
                G = S.Gstack(sigma, ms, s, xp)[:, :L1]
                Ab[c] += xp.conj(G)[..., None] * gbar[(d,) + c]
            Db[c] += gbar[(2,) + c]
            for i, t in enumerate(c):
                def rep(j, c=c, i=i):
                    return c[:i] + (j,) + c[i + 1:]
                if t == 0:
                    Ab[rep(2)] += gbar[(1,) + c]
                elif t == 1:
                    Ab[rep(2)] += gbar[(0,) + c]
                else:
                    Ab[rep(1)] -= gbar[(1,) + c]
                    Ab[rep(0)] -= gbar[(0,) + c]
        return (self.T.coeff_to_mixed_T(Ab, dom, rank, "over_r")
                + self.T.coeff_to_mixed_T(Db, dom, rank, "dr"))

    def _contract_T(self, cot, xshape):
        """Adjoint of the metric contraction of the first two indices (``_contract``)."""
        xp = array_module(cot)
        G = self.S.G
        out = xp.zeros(xshape, dtype=complex)
        for d in range(self.S.nc):
            for t in range(self.S.nc):
                if G[d, t]:
                    out[d, t] = np.conj(G[d, t]) * cot
        return out

    def _curl_T(self, cot, xshape):
        """Adjoint of ``_curl_of_grad_raw`` in 3D (rank-1 operand): out[k] += cst x[d, t]."""
        xp = array_module(cot)
        out = xp.zeros(xshape, dtype=complex)
        for (d, t), (k, cst) in cross_table().items():
            out[d, t] += np.conj(cst) * cot[k]
        return out

    # ------------------------------------------------------------------ backward walk
    def _T(self, x, cot, ctx) -> None:
        """Propagate the cotangent ``cot`` of node x's value (its representation) to the
        direction fields (accumulated in ctx['acc'])."""
        if x in self._dset:
            acc = ctx["acc"]
            acc[x] = cot if x not in acc else acc[x] + cot
            return
        data, params = ctx["data"], ctx["params"]
        xp = array_module(cot)
        S = self.S
        if isinstance(x, (E.Gradient, E.Divergence, E.Curl, E.Laplacian, E.Differentiate)):
            arg = x.args[0]
            am = self._meta_of(arg, data, params)
            cm_dom, cm_rank = am.dom, am.rank
            if isinstance(x, E.Differentiate):
                ms = S._m_of(cm_dom)
                nd = len(S.coeff_shape(cm_dom, cm_rank))
                mm = xp.asarray(ms.reshape((-1,) + (1,) * (nd - 1)))
                vbar = xp.conj(1j * mm) * cot
            elif isinstance(x, E.Laplacian):
                # forward: contract(grad(coeff(grad(v)))): chain of the adjoints
                g2shape = (S.nc, S.nc) + cot.shape
                g2bar = self._contract_T(cot, g2shape)
                c1bar = self._grad_T(g2bar, cm_dom, cm_rank + 1)
                g1bar = self.T.mixed_to_coeff_T(c1bar, cm_dom, cm_rank + 1)
                vbar = self._grad_T(g1bar, cm_dom, cm_rank)
            else:
                if isinstance(x, E.Gradient):
                    gbar = cot
                elif isinstance(x, E.Divergence):
                    gbar = self._contract_T(cot, (S.nc, S.nc) + cot.shape)
                else:
                    if cm_rank != 1:
                        raise SpectralError("spherical adjoint: curl of a vector only")
                    gbar = self._curl_T(cot, (S.nc,) + cot.shape)
                vbar = self._grad_T(gbar, cm_dom, cm_rank)
            self._T(arg, self._coeff_T(am, vbar), ctx)
            return
        if isinstance(x, E.Add):
            for a in x.args:
                if self._active(a):
                    am = self._meta_of(a, data, params)
                    self._T(a, self._grid_T(am, unbroadcast(cot, am.gshape)), ctx)
            return
        if isinstance(x, E.Mul):
            act = [a for a in x.args if self._active(a)]
            if len(act) != 1:
                raise SpectralError(f"spherical adjoint: '{x}' is not linear in the direction")
            fac = 1.0
            for a in x.args:
                if a is not act[0]:
                    fac = fac * self._pgrid(a, ctx)
            am = self._meta_of(act[0], data, params)
            gbar = unbroadcast(xp.conj(xp.asarray(fac)) * cot, am.gshape)
            self._T(act[0], self._grid_T(am, gbar), ctx)
            return
        if isinstance(x, (E.Dot, E.Cross, E.Outer)):
            a, b = x.args
            if self._active(a) and self._active(b):
                raise SpectralError(f"spherical adjoint: '{x}' is not linear in the direction")
            first = self._active(a)
            act, pas = (a, b) if first else (b, a)
            am = self._meta_of(act, data, params)
            P = self._pgrid(pas, ctx)
            if isinstance(x, E.Cross):
                bar = xp.zeros((S.nc,) + tuple(np.broadcast_shapes(am.gshape[1:], P.shape[1:])),
                               dtype=complex)
                for (i, j), (k, cst) in cross_table().items():
                    if first:                           # out[k] += cst a[i] b[j], a active
                        bar[i] += np.conj(cst) * xp.conj(P[j]) * cot[k]
                    else:
                        bar[j] += np.conj(cst) * xp.conj(P[i]) * cot[k]
            elif isinstance(x, E.Outer):
                ra, rb = a.rank, b.rank
                if first:                               # out = a[I, 1.., g] * b[1.., J, g]
                    pr = P.reshape((1,) * ra + P.shape)
                    bar = (xp.conj(pr) * cot).sum(axis=tuple(range(ra, ra + rb)))
                else:
                    ga = P.reshape(P.shape[:ra] + (1,) * rb + P.shape[ra:])
                    bar = (xp.conj(ga) * cot).sum(axis=tuple(range(ra)))
            else:
                bar = self._dot_T(first, P, a.rank, b.rank, am.gshape, cot)
            self._T(act, self._grid_T(am, unbroadcast(bar, am.gshape)), ctx)
            return
        if isinstance(x, E.Component):
            am = self._meta_of(x.args[0], data, params)
            gbar = xp.zeros(am.gshape, dtype=complex)
            gbar[2] = unbroadcast(cot, am.gshape[1:])
            self._T(x.args[0], self._grid_T(am, gbar), ctx)
            return
        if isinstance(x, E.Conj):
            am = self._meta_of(x.args[0], data, params)
            out = cot
            for ax in range(am.rank):
                out = xp.take(out, xp.asarray([1, 0, 2][:S.nc]), axis=ax)
            self._T(x.args[0], self._grid_T(am, unbroadcast(xp.conj(out), am.gshape)), ctx)
            return
        raise SpectralError(f"spherical adjoint: node {type(x).__name__} ('{x}') of the linearized "
                            "right-hand side has no transpose (pointwise functions of the state, "
                            "integrals or interpolations in a volume RHS are not supported)")

    def _dot_T(self, first: bool, P, ra: int, rb: int, gshape, cot):
        """Adjoint of ``sphere_eval._dot`` for the active operand (a if ``first`` else b), the
        other one (``P``, grid) fixed. Forward: out = sum_(s,t) G[s, t] outer(A[s], b[t]),
        A = a with its last index moved first, outer over the remaining indices."""
        xp = array_module(cot)
        G = self.S.G
        ar, br = ra - 1, rb - 1
        nc = self.S.nc
        if first:                                       # cotangent of a: (nc,)*ra + grid
            bar_A = []
            for s in range(nc):
                acc = 0
                for t in range(nc):
                    if G[s, t]:
                        y = P[t]                        # b[t]: (nc,)*br + grid_b
                        yr = y.reshape((1,) * ar + y.shape)
                        acc = acc + np.conj(G[s, t]) * (xp.conj(yr) * cot).sum(
                            axis=tuple(range(ar, ar + br)))
                bar_A.append(acc)
            gfull = np.broadcast_shapes(*[np.shape(b) for b in bar_A if not np.isscalar(b)])
            A = xp.stack([xp.broadcast_to(xp.asarray(b, dtype=complex), gfull) for b in bar_A])
            return xp.moveaxis(A, 0, ra - 1)
        A = xp.moveaxis(P, ra - 1, 0)                   # a fixed
        bars = []
        for t in range(nc):
            acc = 0
            for s in range(nc):
                if G[s, t]:
                    xs = A[s]
                    xr = xs.reshape(xs.shape[:ar] + (1,) * br + xs.shape[ar:])
                    acc = acc + np.conj(G[s, t]) * (xp.conj(xr) * cot).sum(axis=tuple(range(ar)))
            bars.append(acc)
        gfull = np.broadcast_shapes(*[np.shape(b) for b in bars if not np.isscalar(b)])
        return xp.stack([xp.broadcast_to(xp.asarray(b, dtype=complex), gfull) for b in bars])

    # ------------------------------------------------------------------ public adjoint
    def _ctx(self, X, t, cache: dict | None = None) -> dict:
        """Passive data of the state X at time t. ``cache``: a dict owned by the caller for the
        duration of one adjoint step (FlowMap passes its per-step cache), where X is not modified;
        no cache across calls (an LRU keyed by the array's id and pointer
        returned the linearization of the previous contents when the caller updated X in place)."""
        key = ("sphere_ctx", id(X), t)
        if cache is not None and key in cache:
            return cache[key][1]
        ctx = {"data": self._state_data(X), "params": self._params(t), "vals": {}}
        if cache is not None:
            cache[key] = (X, ctx)                  # keep X alive: its id cannot be reused
        return ctx

    def transpose(self, X, W, t=None, cache: dict | None = None):
        """F'(X, t)† W for the real pairing <a, b> = Re(a^H b) (exact adjoint of ``apply``)."""
        if self._fused is not None:
            self._fused_passive.begin()
            try:
                # adjoint of the planned tangent F'(P X) P: P applied last (see __init__)
                return self._real_P.apply(self._transpose(X, W, t, cache))
            finally:
                self._fused_passive.begin()
        return self._transpose(X, W, t, cache)

    def _transpose(self, X, W, t, cache):
        xp = self.xp
        ctx0 = self._ctx(X, t, cache)
        ctx = dict(ctx0, acc={})
        data = dict(ctx0["data"])
        for v, d in self.dvars.items():                # zero directions: only shapes are needed
            data[d] = xp.zeros(self.host.var_shapes[self.variables.index(v)], dtype=complex)
        ctx["data"] = data
        S = self.S
        planned = self._planned.planned if self._planned is not None else frozenset()
        for i, (le, k, RH, shape) in enumerate(zip(self.lin, self.rpow, self._RH, self.host.eq_shapes)):
            if le is None or RH is None or i in planned:
                continue
            dom, rank = le.domain, le.rank             # evaluate() uses the expression's domain
            if k:
                dom = self.ev.rpow_domain(dom)         # ... with r added for r^k le (evaluate)
            cbar = (RH @ W).reshape(shape)
            cs = S.coeff_shape(dom, rank)
            if cs != tuple(shape):                     # adjoint of embed_coeffs: restriction
                cbar = cbar[tuple(slice(0, n_) for n_ in cs)]
            m = self._meta_of(le, data, ctx["params"])
            if k:
                # forward: forward(broadcast(grid(v)) * r^k) ( projection of r^k F)
                gfull = self.T.forward_T(cbar, dom, rank)
                gbar = gfull * xp.asarray(S.ops.grid(gfull.shape[-1]) ** k)
                self._T(le, self._grid_T(m, unbroadcast(gbar, m.gshape)), ctx)
            elif m.kind == "C":
                self._T(le, unbroadcast(cbar, m.shape), ctx)
            elif m.kind == "X":
                xfull = self.T.mixed_to_coeff_T(cbar, dom, rank)
                self._T(le, unbroadcast(xfull, m.shape), ctx)
            else:
                self._T(le, unbroadcast(self.T.forward_T(cbar, dom, rank), m.shape), ctx)
        if planned:                                    # the planned equations (sphere_planned)
            for d, a in self._planned.transpose(ctx0["data"], ctx0["params"], W).items():
                acc = ctx["acc"]
                acc[d] = a if d not in acc else acc[d] + a
        out = xp.zeros(self.host.n, dtype=self.host.dtype)
        for v, (sel, flat) in zip(self.variables, self._cols):
            a = ctx["acc"].get(self.dvars[v])
            if a is not None:
                out[sel] += a.reshape(-1)[flat]
        return out

    def explain(self) -> str:
        # which terms are explicit (in F, linearized here); everything on the left-hand side
        # (implicit: viscosity, pressure, an implicit Coriolis term) goes through M and L
        terms = [f"eq {i}: {str(e)[:90]}" for i, (e, le) in
                 enumerate(zip(self.solver._rhs_exprs, self.lin)) if le is not None]
        tan = ("the planned evaluator of f_backend='fused' (tangent of the fused F, which reads its "
               "inputs through real-field transforms: F'(P X) P V, adjoint P F'(P X)^dagger W with P "
               "the m = 0 real projection)" if self._fused is not None else "the spherical evaluator")
        return ("F' and F'† matrix-free on the shell: IR linearization evaluated by " + tan + " "
                "(tangent); explicit adjoints of the radial Chebyshev synthesis/projection "
                "(with the r^k F projection of the shell rows), spin-weighted Gauss-Legendre synthesis/projection, "
                "azimuthal FFTs with the conjugate modes -m, spin-basis grad/div/curl/lap and the "
                "products (adjoint); real pairing <a, b> = Re(a^H b)\n"
                "    adjoint path: " + self._planned_note + "\n"
                "    explicit terms (F, linearized): " + ("; ".join(terms) if terms else "none (F' = 0)")
                + "; implicit terms: left-hand sides (M, L)")
