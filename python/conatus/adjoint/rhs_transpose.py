"""Matrix-free linearized right-hand side F'(X) and its exact transpose F'(X)ᵀ (phase 2).

The explicit right-hand sides are expanded once into their polynomial form (the same expansion as
the fused evaluator, :func:`conatus.timestep.fused.expand_polynomial`)

    F_(eq, comp) = sum_m  c_m  C( prod_j  G(D^{a_mj} f_mj) ),

with G the backward transform to the dealiased grid, C the forward transform keeping the first N
modes (the dealiasing projection) and D^a products of 1D spectral derivatives. Then

    F'(X) V   = sum_m c_m sum_{j active} C( G(D^a V_j) * prod_{i != j} g_i ),
    F'(X)ᵀ W  = sum_m c_m sum_{j active} (D^a)ᵀ Gᵀ( Cᵀ(W) * prod_{i != j} g_i ),

where g_i = G(D^a f_i) are the passive factor grids at X. Gᵀ and Cᵀ are the **explicit transposes**
of the reference transforms (bilinear pairing, no conjugation, D8), written with FFT/DCT below
(the dense matrices of the bases of ``discretize.families`` are transposed as they are) and checked
against dense matrices in tests/adjoint; (D^a)ᵀ uses the 1D derivative matrix built
from the evaluator itself. The map V -> F'(X) V is the same linear map as the IR linearization
evaluated by :class:`SpectralEvaluator` up to rounding (products are formed on the same grid).

The equation rows go through ``GlobalSystem.gather_rhs`` (conversion to C^(lam), row selection)
whose transpose is written here too. Right-hand sides with a **derivative of a product** (divergence form, curl curl of
−∇·(uu): the poloidal–toroidal channel) use the *general form* of the fused evaluator instead
(:func:`conatus.timestep.fused.expand_general`, the same representation the forward F uses):

    F_(eq, comp) = sum_t c_t Post_t( C( prod_j G(A_tj X) ) ),

with A_tj linear forms (sums of c · D^a f_comp in coefficient space) and Post_t a post-operator
(integrals along Fourier axes = zero-mode slice × length, then spectral derivatives), applied to
the **projected** product exactly as the forward evaluation does (no Leibniz). Then

    F'(X) V  = sum_t c_t Post_t C( sum_j G(A_tj V) prod_{i != j} G(A_ti X) ),
    F'(X)ᵀ W = sum_t c_t sum_j A_tjᵀ Gᵀ( Cᵀ(Post_tᵀ W) prod_{i != j} G(A_ti X) ).

Declared in ``explain()`` (per equation: polynomial or general form). Other non-polynomial
nodes (interpolation, …) are **refused** (SpectralError): FlowMap then needs
``f_transpose="dense"`` explicitly.

**Time-dependent right-hand sides** (``IVPSolver(..., time_parameter="t")``): each RHS is
split once, symbolically, into (:func:`split_time`)

    F(X, t) = F_0(X) + sum_g a_g(t) F_g(X),

where a_g are field-free scalar expressions of t (e.g. cos(2 pi t) of a Stokes base flow
U(y, t) = cos(2 pi t) Uc(y) + sin(2 pi t) Us(y)) and F_g do not contain t. The split uses only
exact identities: a_g is constant in space, so it commutes with every linear spatial operator
(derivatives, gradient, divergence, components, integrals, lifts) and factors out of products.
Then F'(X, t) = F_0'(X) + sum_g a_g(t) F_g'(X), every F_g expanded once (polynomial or general
form as above), a_g evaluated at the stage time. Terms that do not depend on the state (a
time-dependent forcing or boundary value) drop out of the derivative. A node that mixes t and
the fields otherwise (e.g. cos(t u)) is refused.
"""
from __future__ import annotations


import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp

from ..ir import expr as E
from ..ir.bases import Chebyshev, ComplexFourier, RealFourier
from ..reference import transforms as T
from ..timestep.fused import _formkey, expand_general, expand_polynomial
from ..timestep.spectral import SpectralError
from ..errors import OptionError

__all__ = ["LinearizedRHS", "grid_T", "coeffs_T", "split_time", "eval_scalar"]


# ------------------------------------------------------------------ 1D transposes (reference conventions)
def _fft_mods(xp):
    """(fft module, dct function) for NumPy or CuPy (same scipy.fft semantics)."""
    if xp is np:
        return np.fft, sfft.dct
    import cupyx.scipy.fft as cfft
    return xp.fft, cfft.dct


def _family_T(b, which: str, v, xp):
    """Transposes of the dense transforms of the bases of ``discretize.families`` (Legendre,
    Jacobi, Gegenbauer, Hermite, Laguerre, compound Chebyshev): backward
    f = B c on the dealiased grid -> Bᵀ w, forward c = F f -> Fᵀ c. None if ``b`` has no family."""
    from ..discretize.families import family_of
    fam = family_of(b)
    if fam is None:
        return None
    B, F = fam.matrices(True)                       # (M x N), (N x M)
    A = B.T if which == "grid" else F.T
    sh = v.shape
    out = xp.asarray(A) @ xp.ascontiguousarray(v.reshape(sh[0], -1))
    return out.reshape((A.shape[0],) + sh[1:])


def grid_T(w, b, axis: int, xp=np):
    """Transpose of the backward transform to the dealiased grid (``T.backward`` /
    ``xp_transforms.backward``, same conventions): M grid values -> N coefficients."""
    N, M = b.size, b.grid_size
    fft, dct = _fft_mods(xp)
    w = xp.moveaxis(xp.asarray(w), axis, 0)
    if isinstance(b, Chebyshev):
        # backward: y = pad(c*s); y0 *= 2; f = idct2(M y) = dct3(y)/2  =>  cᵀ = (dct2(w)/2)[:N] * s
        y = dct(w, type=2, axis=0) / 2
        sgn = xp.asarray((-1.0) ** np.arange(N)).reshape((N,) + (1,) * (w.ndim - 1))
        c = y[:N] * sgn
    elif isinstance(b, RealFourier):
        if xp.iscomplexobj(w):
            raise SpectralError("RealFourier transpose takes real data")
        # backward: f_j = c0 + sum_k a_k cos(k x_j) - b_k sin(k x_j)
        R = fft.rfft(w, axis=0)
        K = N // 2
        c = xp.zeros((N,) + w.shape[1:])
        c[0] = R[0].real
        c[2:N:2] = R[1:K].real          # sum_j w_j cos(k x_j)
        c[3:N:2] = R[1:K].imag          # -sum_j w_j sin(k x_j)
    elif isinstance(b, ComplexFourier):
        # backward: f_j = sum_k F_k e^{+ikx_j}  =>  F_kᵀ = sum_j w_j e^{+ikx_j} = M ifft(w)_k
        F = fft.ifft(w, axis=0) * M
        K = N // 2
        c = xp.zeros((N,) + w.shape[1:], dtype=complex)
        c[:K] = F[:K]
        c[K + 1:] = F[M - K + 1:]
    else:
        c = _family_T(b, "grid", w, xp)
        if c is None:
            raise SpectralError(f"transpose of the {type(b).__name__} transform not implemented")
    return xp.moveaxis(c, 0, axis)


def coeffs_T(ct, b, axis: int, xp=np):
    """Transpose of the forward transform from the dealiased grid (first N modes kept):
    N coefficients -> M grid values."""
    N, M = b.size, b.grid_size
    fft, dct = _fft_mods(xp)
    ct = xp.moveaxis(xp.asarray(ct), axis, 0)
    shape = (M,) + ct.shape[1:]
    if isinstance(b, Chebyshev):
        # forward: y = dct2(f)/M; y0 /= 2; c = (y*s)[:N]  =>  fᵀ = dct3(pad(c*s))/M
        y = xp.zeros(shape, dtype=ct.dtype)
        y[:N] = ct * xp.asarray((-1.0) ** np.arange(N)).reshape((N,) + (1,) * (ct.ndim - 1))
        f = dct(y, type=3, axis=0) / M
    elif isinstance(b, RealFourier):
        if xp.iscomplexobj(ct):
            raise SpectralError("RealFourier transpose takes real data")
        # forward: c0 = mean f, a_k = (2/M) sum f cos, b_k = -(2/M) sum f sin
        # => fᵀ_j = (1/M)(c0ᵀ + 2 sum_k aᵀ_k cos - bᵀ_k sin) = backward of the scaled coefficients
        sc = np.full((N,) + (1,) * (ct.ndim - 1), 2.0 / M)
        sc[0] = 1.0 / M
        sc[1] = 0.0                     # msin_0 is not an unknown of forward
        if xp is np:
            f = T.backward(ct * sc, b, axis=0, dealias=True)
        else:
            from ..timestep import xp_transforms as XT
            f = XT.backward(xp, ct * xp.asarray(sc), b, axis=0)
    elif isinstance(b, ComplexFourier):
        # forward: F = fft(f)/M, c = F on the kept modes  =>  fᵀ = fft(pad(cᵀ))/M
        K = N // 2
        F = xp.zeros(shape, dtype=complex)
        F[:K] = ct[:K]
        F[M - K + 1:] = ct[K + 1:]
        f = fft.fft(F, axis=0) / M
    else:
        f = _family_T(b, "coeffs", ct, xp)
        if f is None:
            raise SpectralError(f"transpose of the {type(b).__name__} transform not implemented")
    return xp.moveaxis(f, 0, axis)


def _apply_axis(A, c, axis: int, xp=np):
    """out[.., n, ..] = sum_m A[n, m] c[.., m, ..] along ``axis`` (A: scipy sparse or DeviceCsr)."""
    cm = xp.moveaxis(c, axis, 0)
    sh = cm.shape
    out = A @ xp.ascontiguousarray(cm.reshape(sh[0], -1))
    return xp.moveaxis(xp.asarray(out).reshape((A.shape[0],) + sh[1:]), 0, axis)


def _reduce_to(a: np.ndarray, shape: tuple) -> np.ndarray:
    """Transpose of broadcasting ``shape`` to ``a.shape`` (sum over broadcast axes)."""
    axes = tuple(i for i, (sa, s) in enumerate(zip(a.shape, shape)) if s == 1 and sa != 1)
    return a.sum(axis=axes, keepdims=True) if axes else a


# ------------------------------------------------------------------ time-dependent RHS: exact split
# spatial linear operators with one operand: a scalar a(t), constant in space, commutes with them
_LINEAR_OPS = (E.Differentiate, E.Gradient, E.Divergence, E.Laplacian, E.Curl, E.Component,
               E.Integrate, E.Interpolate, E.Lift)


def split_time(e: E.Expr, tname: str) -> list[tuple]:
    """[(a_g, F_g)] with e = sum_g a_g F_g exactly; a_g is None (the constant 1) or a field-free
    scalar expression containing the parameter ``tname``, F_g does not contain ``tname``. Terms
    with the same a_g (structural equality) are summed. SpectralError if ``e`` mixes ``tname``
    and the fields in any other way (then no such split exists)."""

    def dep(x: E.Expr) -> bool:
        return any(p.name == tname for p in x.free_parameters())

    def amul(a, b):
        if a is None:
            return b
        return a if b is None else E.mul(a, b)

    def rec(x: E.Expr) -> list[tuple]:
        if not dep(x):
            return [(None, x)]
        if not x.free_fields():
            if x.rank:
                raise SpectralError(f"time-dependent RHS: tensor-valued function of {tname} "
                                    f"({x}) refused")
            return [(x, E.Constant(1))]
        if isinstance(x, E.Add):
            return [p for a in x.args for p in rec(a)]
        if isinstance(x, E.Mul):
            out: list = [(None, [])]
            for f in x.args:
                out = [(amul(a, b), ss + [s]) for a, ss in out for b, s in rec(f)]
            return [(a, E.mul(*ss)) for a, ss in out]
        if isinstance(x, (E.Dot, E.Cross, E.Outer)):
            A, B = rec(x.args[0]), rec(x.args[1])
            return [(amul(a1, a2), x.rebuild([s1, s2])) for a1, s1 in A for a2, s2 in B]
        if isinstance(x, _LINEAR_OPS) and len(x.args) == 1:
            return [(a, x.rebuild([s])) for a, s in rec(x.args[0])]
        raise SpectralError(f"time-dependent RHS: node {type(x).__name__} ({x}) mixes {tname} and "
                            f"the fields; not a sum of (function of {tname}) x (expression "
                            f"without {tname}): refused")

    groups: dict = {}
    for a, s in rec(e):
        if s.is_zero():
            continue
        groups[a] = s if a not in groups else E.add(groups[a], s)
    return [(a, s) for a, s in groups.items() if not s.is_zero()]


def _refuse_nondifferentiable(exprs, variables) -> None:
    """A typed refusal for abs/sign of the state: the tangent
    and adjoint maps of F do not exist where the argument changes sign. It used to fail later
    with "fused RHS: node UnaryFunction not supported", which did not say why."""
    from ..errors import UnsupportedError
    from ..ir.expr_algebra import _FUNCS, NONDIFFERENTIABLE, NONDIFFERENTIABLE_FIX
    state = frozenset(variables)
    for e in exprs:
        if e is None:
            continue
        for x in e.walk():
            if (isinstance(x, E.UnaryFunction) and x.args[0].free_fields() & state
                    and (x.func in NONDIFFERENTIABLE or _FUNCS[x.func][3] is None)):
                user = x.func not in NONDIFFERENTIABLE
                raise UnsupportedError(
                    f"tangent / adjoint of the right-hand side: {x} is not differentiable",
                    why=(f"{x.func}() was registered without derivative=" if user else
                         f"{x.func} has no derivative where its argument {x.args[0]} is 0 "
                         f"({'a corner' if x.func == 'abs' else 'a jump'}), and a trajectory "
                         "crosses that set: the tangent map (jvp) and the adjoint (vjp) of the "
                         "flow are undefined there; a value would depend on an arbitrary choice "
                         "at 0"),
                    fix=("pass derivative=f' to cn.pointwise(...)" if user
                         else NONDIFFERENTIABLE_FIX),
                    doc="docs/site/guide/dynamics/adjoint.md")


def eval_scalar(x: E.Expr, params: dict):
    """Value of a field-free scalar expression (Parameter, Constant, +, *, **, pointwise
    functions) with ``params`` {name: value}; SpectralError on any other node."""
    if isinstance(x, E.Parameter):
        v = params.get(x.name, x.value)
        if v is None:
            raise SpectralError(f"no value for parameter {x.name}")
        return v
    if isinstance(x, E.Constant):
        return x.value
    if isinstance(x, E.Add):
        return sum(eval_scalar(a, params) for a in x.args)
    if isinstance(x, E.Mul):
        out = 1.0
        for a in x.args:
            out = out * eval_scalar(a, params)
        return out
    if isinstance(x, E.Power):
        return eval_scalar(x.args[0], params) ** x.exponent
    if isinstance(x, E.UnaryFunction):
        return x.numpy_func(eval_scalar(x.args[0], params))
    raise SpectralError(f"time coefficient: node {type(x).__name__} not supported")


# ------------------------------------------------------------------ linearized RHS
class LinearizedRHS:
    """F'(X) V and F'(X)ᵀ W for the polynomial RHS of an :class:`IVPSolver` (NumPy on CPU, CuPy
    on the GPU backend: same formulas, arrays stay on the device)."""

    def __init__(self, solver) -> None:
        s = solver
        _refuse_nondifferentiable(s._rhs_exprs, s.system.variables)
        self.solver, self.system, self.ev = s, s.system, s.evaluator
        self.xp = s.xp
        self.gpu = self.xp is not np
        self.coords = list(s.plan.domain.coordsys.coords)
        self.bases = [s.plan.domain.basis(c) for c in self.coords]
        self.variables = list(self.system.variables)
        self._vidx = {v: k for k, v in enumerate(self.variables)}
        params = dict(s.plan.params)
        # time-dependent RHS: F = F_0 + sum_g a_g(t) F_g (split_time); group g = 0 is the
        # t-free part, self.tcoefs[g - 1] = a_g. Without time parameter: one group, g = 0.
        self.tname = getattr(s, "time_parameter", None)
        self.tcoefs: list = []
        parts_all: list = []
        for e in s._rhs_exprs:
            if e is None:
                parts_all.append([])
                continue
            parts = split_time(e, self.tname) if self.tname else [(None, e)]
            tagged = []
            for a, se in parts:
                if not set(se.free_fields()) & set(self.variables):
                    continue                    # part without the state (forcing): derivative 0
                if a is None:
                    g = 0
                else:
                    if a not in self.tcoefs:
                        self.tcoefs.append(a)
                    g = self.tcoefs.index(a) + 1
                tagged.append((g, se))
            parts_all.append(tagged)
        self.polys: list[dict | None] = []
        # general form ({comp: [(c, (dc, ig), forms, g)]}) for every equation when the polynomial
        # form refuses a derivative of a product: the same switch, for all equations at once, as
        # FusedRHS._compile (so the tangent differentiates the forward F term by term)
        self.generals: list[dict | None] = [None] * len(s._rhs_exprs)
        fourier_axes = [i for i, b in enumerate(self.bases) if not isinstance(b, Chebyshev)]
        general = False
        for parts in parts_all:
            for _, se in parts:
                try:
                    expand_polynomial(se, self.coords, params)
                except SpectralError as err:
                    if "derivative of a product" not in str(err):
                        raise                                    # other nodes: refused
                    general = True
                    break
            if general:
                break
        for k, parts in enumerate(parts_all):
            if not parts:
                self.polys.append(None)             # constant RHS (BCs, gauges): derivative 0
                continue
            if general:
                Gk: dict = {}
                for g, se in parts:
                    gen = expand_general(se, self.coords, params, fourier_axes)
                    for comp, terms in gen.items():
                        Gk.setdefault(comp, []).extend(
                            (c, post, fs, g) for c, post, fs in terms
                            if fs and any(a is not None and a[0] in self._vidx
                                          for f in fs for _, a in f))
                self.generals[k] = Gk
                self.polys.append(None)
                continue
            Pk: dict = {}
            for g, se in parts:
                P = expand_polynomial(se, self.coords, params)
                # keep monomials that contain at least one variable (the others do not depend on X)
                for comp, monos in P.items():
                    Pk.setdefault(comp, []).extend((c, fs, g) for c, fs in monos
                                                   if any(f[0] in self._vidx for f in fs))
            self.polys.append(Pk)
        self._dmatsT: dict = {}
        self._convT: dict = {}
        # No cache of state-dependent data across calls (the defect of the
        # spherical adjoint, here too): an LRU keyed by the array's id and pointer returned the
        # linearization about the previous contents when the caller updated X in place.
        host = self.system.host if self.gpu else self.system
        # (sel, flat) index maps of gather_rhs / scatter, on the array module of the state
        asx = self.xp.asarray
        if self.gpu:                      # int32 maps on the device when they fit
            from ..timestep.device import _index_map
            asx = lambda a: _index_map(self.xp, a)  # noqa: E731
        self._rows = {key: (asx(sel), asx(host.rows_flat[sel])) for key, sel in host._row_sel.items()}
        self._cols = [(asx(sel), asx(host.cols_flat[sel])) for sel in host._col_sel]

    # -------------------------------------------------------------- helpers
    def _sparse(self, A: sp.spmatrix):
        """Sparse matrix for :func:`_apply_axis` on this array module."""
        if not self.gpu:
            return A.tocsr()
        from ..timestep.device import DeviceCsr
        return DeviceCsr(A.tocsr(), np.result_type(A.dtype, self.system.dtype))

    # -------------------------------------------------------------- factor grids (passive)
    def _factor_coeffs(self, data: dict, f):
        """D^a f_comp in coefficient space, with the evaluator's own derivative (same arithmetic)."""
        field, comp, a = f
        c = self.xp.asarray(data[field])
        c = c[comp] if comp else c
        nd = len(self.coords)
        if c.ndim < nd:                              # constant along trailing coordinates
            c = c.reshape(c.shape + (1,) * (nd - c.ndim))
        for coord, n in zip(self.coords, a):
            for _ in range(n):
                c = self.ev._diff(c, coord, 0)
        return c

    def factor_grids(self, X, cache: dict | None = None) -> dict:
        """Grids G(D^a f) of every factor at the state X. ``cache``: a dict owned by the caller
        for the duration of one tangent/adjoint step (FlowMap passes its per-step cache), during
        which X is not modified; no cache across calls (an array updated in place keeps its id
        and pointer: same defect as)."""
        key = ("factor", id(X))
        if cache is not None and key in cache:
            return cache[key][1]
        data = dict(self.solver._ncc)
        data.update(zip(self.variables, self.system.scatter(X)))
        grids = {}
        for P in self.polys:
            if P is None:
                continue
            for monos in P.values():
                for _, fs, _g in monos:
                    for f in fs:
                        if f not in grids:
                            grids[f] = self.ev.to_grid(self._factor_coeffs(data, f), 0)
        if cache is not None:
            cache[key] = (X, grids)                # keep X alive: its id cannot be reused
        return grids

    # -------------------------------------------------------------- derivative transposes
    def _dmatT(self, coord, n: int):
        """(D^n)ᵀ along ``coord``, from the evaluator's own 1D derivative (built once)."""
        key = (coord, n)
        A = self._dmatsT.get(key)
        if A is None:
            from ..timestep.spectral import SpectralEvaluator
            b = self.ev._basis(coord)
            ref = SpectralEvaluator(self.solver.plan.domain)          # NumPy reference operator
            D1 = sp.csr_matrix(ref._host_matrix(lambda r, e: r._diff(e, coord, 0), b, coord))
            Dn = sp.identity(b.size, format="csr", dtype=D1.dtype)
            for _ in range(n):
                Dn = (D1 @ Dn).tocsr()
            A = self._dmatsT[key] = self._sparse(Dn.T)
        return A

    def _GT_batched(self, gct: dict) -> dict:
        """Gᵀ of every factor grid cotangent, stacked by (shape, dtype) so that each transposed
        transform runs once per axis on a batch (same numbers as factor by factor)."""
        xp = self.xp
        groups: dict = {}
        for f, g in gct.items():
            k = self._vidx[f[0]]
            vshape = self.system.var_shapes[k]
            gshape = tuple(vshape[len(vshape) - len(self.coords):])
            groups.setdefault((g.shape, g.dtype.str, gshape), []).append(f)
        out = {}
        k = self.transform_batch
        for (_shape, _, gshape), fs_all in groups.items():
            # at most transform_batch factors per stacked transform (the forward F's setting;
            # same per-factor transforms, smaller temporaries)
            step = k if k else len(fs_all)
            for i0 in range(0, len(fs_all), step):
                fs = fs_all[i0:i0 + step]
                B = xp.stack([gct.pop(f) for f in fs])
                for i, b in enumerate(self.bases):
                    if b is not None and gshape[i] == b.size and B.shape[i + 1] == b.grid_size:
                        B = grid_T(B, b, i + 1, xp)
                for j, f in enumerate(fs):
                    out[f] = B[j]
        return out

    @property
    def transform_batch(self) -> int | None:
        """Fields per batched transposed transform: the forward fused F's ``transform_batch``
        (None: all at once)."""
        fu = getattr(self.solver, "_fused", None)
        fu = getattr(fu, "inner", fu)
        return getattr(fu, "transform_batch", None)

    def _CT(self, ct, out_grid_shape: tuple):
        """Transpose of: grid (shape ``out_grid_shape``) -> coefficients -> embedding in the equation
        shape (size-1 grid axes are the zero mode)."""
        g = ct
        for i, b in enumerate(self.bases):
            if out_grid_shape[i] == 1:
                g = self.xp.take(g, self.xp.asarray([0]), axis=i)    # embedᵀ: zero mode only
            elif b is not None:
                g = coeffs_T(g, b, i, self.xp)
        return g

    # -------------------------------------------------------------- gather_rhs / scatter transposes
    def _gather_rhs_T(self, w) -> list:
        xp, sysm = self.xp, self.system
        zc = sysm.plan.layout.zcoord
        coords = sysm.plan.domain.coordsys.coords
        eqs = sysm.plan.problem.equations
        out: list = [None] * len(eqs)
        for (k, lam), (sel, flat_idx) in self._rows.items():
            if self.polys[k] is None and self.generals[k] is None:
                continue
            shape = sysm.eq_shapes[k]
            flat = xp.zeros(int(np.prod(shape)), dtype=np.result_type(w.dtype, float))
            if self.gpu:                      # transpose of a gather: scatter-add
                import cupyx
                cupyx.scatter_add(flat, flat_idx, w[sel])
            else:
                np.add.at(flat, flat_idx, w[sel])
            c = flat.reshape(shape)
            if lam > 0:
                ax = eqs[k].lhs.rank + coords.index(zc)
                if c.shape[ax] > 1:
                    ST = self._convT.get(lam)
                    if ST is None:
                        S = sysm.plan.layout.zconv(lam).tocsr()   # family-aware
                        ST = self._convT[lam] = self._sparse(S.T)
                    c = _apply_axis(ST, c, ax, xp)
            out[k] = c if out[k] is None else out[k] + c
        return out

    def _scatter_T(self, arrays: list):
        xp, sysm = self.xp, self.system
        v = xp.zeros(sysm.n, dtype=sysm.dtype)
        for (sel, flat_idx), a in zip(self._cols, arrays):
            if a is None:
                continue
            v[sel] += xp.asarray(a).reshape(-1)[flat_idx]
        return v

    # -------------------------------------------------------------- general form (products of forms)
    def _form_coeffs(self, data: dict, form, linear: bool):
        """Linear form sum c · D^a f_comp in the full coefficient shape (size-1 axes of a field
        embedded as its zero mode). ``linear``: drop the constant and the atoms that are not
        in ``data`` (NCC atoms: zero derivative)."""
        full = tuple(b.size for b in self.bases)
        out = None
        for cf, a in form:
            if a is None:
                if linear:
                    continue
                t = self.xp.zeros(full)
                t[(0,) * len(full)] = cf
            elif a[0] not in data:
                if linear:
                    continue
                raise KeyError(f"no data for {a[0].name}")
            else:
                t = cf * self.ev._embed(self._factor_coeffs(data, a), 0, full)
            out = t if out is None else out + t
        return out

    def general_grids(self, X, cache: dict | None = None) -> dict:
        """Grids G(A X) of every linear form of the general-form equations at X (``cache``: as
        :meth:`factor_grids`)."""
        key = ("general", id(X))
        if cache is not None and key in cache:
            return cache[key][1]
        data = dict(self.solver._ncc)
        data.update(zip(self.variables, self.system.scatter(X)))
        grids: dict = {}
        for Gk in self.generals:
            for terms in (Gk or {}).values():
                for _, _, fs, _g in terms:
                    for f in fs:
                        fk = _formkey(f)
                        if fk not in grids:
                            grids[fk] = self.ev.to_grid(self._form_coeffs(data, f, False), 0)
        if cache is not None:
            cache[key] = (X, grids)
        return grids

    def _general_apply(self, k: int, grids: dict, dV: dict, lin: dict, shape: tuple, tv=None):
        """F'_k(X) V for a general-form equation k (see module docstring); ``tv``: values of the
        time coefficients (index g), None without time dependence."""
        xp, ev = self.xp, self.ev
        r0 = self.system.plan.problem.equations[k].lhs.rank
        res = xp.zeros(shape, dtype=np.result_type(self.system.dtype, float))
        for comp, terms in self.generals[k].items():
            prodC: dict = {}
            for c, (dc, ig), fs, g in terms:
                c = c if g == 0 else c * tv[g]
                key = tuple(_formkey(f) for f in fs)
                if key not in prodC:
                    g = None
                    for j, f in enumerate(fs):
                        fk = _formkey(f)
                        if fk not in lin:
                            v = self._form_coeffs(dV, f, True)
                            lin[fk] = None if v is None else ev.to_grid(v, 0)
                        if lin[fk] is None:
                            continue
                        term = lin[fk]
                        for i, h in enumerate(fs):
                            if i != j:
                                term = term * grids[_formkey(h)]
                        g = term if g is None else g + term
                    prodC[key] = None if g is None else ev.to_coeffs(g, 0)
                r = prodC[key]
                if r is None:
                    continue
                for ax, flag in enumerate(ig):                 # integrals first (zero mode)
                    if flag:
                        r = xp.take(r, xp.asarray([0]), axis=ax) * self.bases[ax].length
                for ax, n in enumerate(dc):                    # then derivatives
                    for _ in range(n):
                        r = ev._diff(r, self.coords[ax], 0)
                res[comp] = res[comp] + c * ev._embed(r, 0, shape[r0:])
        return res

    def _general_transpose(self, k: int, grids: dict, wk_all, var_ct: list, tv=None) -> None:
        """Accumulate F'_k(X)ᵀ w into the variable cotangents (general-form equation k)."""
        xp = self.xp
        full = tuple(b.size for b in self.bases)
        for comp, terms in self.generals[k].items():
            wk = wk_all[comp]
            acc: dict = {}
            for c, (dc, ig), fs, g in terms:
                c = c if g == 0 else c * tv[g]
                r = wk
                post = tuple(1 if ig[i] else full[i] for i in range(len(full)))
                for i in range(len(full)):                     # embedᵀ (zero mode kept)
                    if post[i] == 1 and r.shape[i] > 1:
                        r = xp.take(r, xp.asarray([0]), axis=i)
                for ax, n in enumerate(dc):                    # derivativesᵀ
                    if n:
                        r = (_apply_axis(self._dmatT(self.coords[ax], n), r, ax, xp)
                             if r.shape[ax] > 1 else 0 * r)     # derivative of a constant: 0
                for ax, flag in enumerate(ig):                 # integralsᵀ: into the zero mode
                    if flag:
                        z = xp.zeros(r.shape[:ax] + (full[ax],) + r.shape[ax + 1:], dtype=r.dtype)
                        idx = [slice(None)] * r.ndim
                        idx[ax] = slice(0, 1)
                        z[tuple(idx)] = r * self.bases[ax].length
                        r = z
                key = tuple(_formkey(f) for f in fs)
                acc[key] = (fs, c * r if key not in acc else acc[key][1] + c * r)
            for fs, R in acc.values():
                Gt = R
                for i, b in enumerate(self.bases):             # Cᵀ: coefficients -> grid
                    Gt = coeffs_T(Gt, b, i, xp)
                for j, f in enumerate(fs):
                    g = Gt
                    for i, h in enumerate(fs):
                        if i != j:
                            g = g * grids[_formkey(h)]
                    for i, b in enumerate(self.bases):         # Gᵀ: grid -> coefficients
                        g = grid_T(g, b, i, xp)
                    for cf, a in f:
                        if a is None or a[0] not in self._vidx:
                            continue
                        kv = self._vidx[a[0]]
                        vshape = self.system.var_shapes[kv]
                        gshape = tuple(vshape[len(vshape) - len(self.coords):])
                        ct = g
                        for i in range(len(full)):             # embedᵀ to the field's shape
                            if gshape[i] == 1 and ct.shape[i] > 1:
                                ct = xp.take(ct, xp.asarray([0]), axis=i)
                        skip = False
                        for i, n in enumerate(a[2]):           # (D^a)ᵀ
                            if n:
                                if ct.shape[i] == 1:
                                    skip = True                # derivative of a constant: 0
                                    break
                                ct = _apply_axis(self._dmatT(self.coords[i], n), ct, i, xp)
                        if skip:
                            continue
                        if var_ct[kv] is None:
                            var_ct[kv] = xp.zeros(vshape, dtype=np.result_type(self.system.dtype, float))
                        var_ct[kv][a[1]] = var_ct[kv][a[1]] + cf * ct

    # -------------------------------------------------------------- apply / transpose
    def time_values(self, t) -> list | None:
        """[1, a_1(t), a_2(t), ...] for a time-dependent RHS (None otherwise); t is required then."""
        if not self.tcoefs:
            return None
        if t is None:
            raise OptionError(f"time-dependent RHS ({self.tname}): the stage time t is required",
                              fix="pass the stage time t= (F depends on the time parameter)")
        params = dict(self.solver.plan.params)
        params[self.tname] = t
        return [1.0] + [eval_scalar(a, params) for a in self.tcoefs]

    def apply(self, X, V, t=None, cache: dict | None = None):
        """F'(X, t) V (global vectors); ``t`` only for a time-dependent RHS; ``cache``: see
        :meth:`factor_grids` (None: everything recomputed from X)."""
        xp = self.xp
        tv = self.time_values(t)
        grids = self.factor_grids(X, cache)
        sysm, ev = self.system, self.ev
        dV = dict(zip(self.variables, sysm.scatter(V)))
        dgrid: dict = {}
        ggrid, glin = None, {}
        out = []
        for k, (P, eq, shape) in enumerate(zip(self.polys, sysm.plan.problem.equations,
                                                sysm.eq_shapes)):
            if P is None:
                if self.generals[k] is not None:
                    if ggrid is None:
                        ggrid = self.general_grids(X, cache)
                    out.append(self._general_apply(k, ggrid, dV, glin, shape, tv))
                else:
                    out.append(None)
                continue
            r = eq.lhs.rank
            res = xp.zeros(shape, dtype=np.result_type(sysm.dtype, float))
            for comp, monos in P.items():
                acc: dict = {}                              # output grid shape -> summed grid
                for cm, fs, g in monos:
                    if g:
                        cm = cm * tv[g]
                    for j, f in enumerate(fs):
                        if f[0] not in self._vidx:
                            continue
                        if f not in dgrid:
                            dgrid[f] = ev.to_grid(self._factor_coeffs(dV, f), 0)
                        term = cm * dgrid[f]
                        for i, g in enumerate(fs):
                            if i != j:
                                term = term * grids[g]
                        acc[term.shape] = term if term.shape not in acc else acc[term.shape] + term
                for g in acc.values():
                    c = ev.to_coeffs(g, 0)
                    res[comp] = res[comp] + ev._embed(c, 0, shape[r:])
            out.append(res)
        return sysm.gather_rhs(out)

    def transpose(self, X, W, t=None, cache: dict | None = None):
        """F'(X, t)ᵀ W (bilinear transpose of :meth:`apply`; ``cache``: see :meth:`factor_grids`)."""
        xp = self.xp
        tv = self.time_values(t)
        grids = self.factor_grids(X, cache)
        sysm = self.system
        eq_ct = self._gather_rhs_T(xp.asarray(W))
        gct: dict = {}                                     # factor -> grid cotangent
        for k, P in enumerate(self.polys):
            if P is None or eq_ct[k] is None:
                continue
            for comp, monos in P.items():
                ctc = eq_ct[k][comp]
                cache: dict = {}
                for cm, fs, g in monos:
                    if g:
                        cm = cm * tv[g]
                    oshape = np.broadcast_shapes(*[grids[f].shape for f in fs])
                    if oshape not in cache:
                        cache[oshape] = self._CT(ctc, oshape)
                    G = cache[oshape]
                    for j, f in enumerate(fs):
                        if f[0] not in self._vidx:
                            continue
                        term = cm * G
                        for i, g in enumerate(fs):
                            if i != j:
                                term = term * grids[g]
                        term = _reduce_to(term, grids[f].shape)
                        gct[f] = term if f not in gct else gct[f] + term
        var_ct: list = [None] * len(self.variables)
        if any(g is not None for g in self.generals):
            ggrid = self.general_grids(X, cache)
            for k, Gk in enumerate(self.generals):
                if Gk is not None and eq_ct[k] is not None:
                    self._general_transpose(k, ggrid, eq_ct[k], var_ct, tv)
        coef = self._GT_batched(gct)                      # one transposed transform per shape
        for f, c in coef.items():
            field, comp, a = f
            k = self._vidx[field]
            for i, (coord, n) in enumerate(zip(self.coords, a)):
                if n and c.shape[i] > 1:
                    c = _apply_axis(self._dmatT(coord, n), c, i, xp)
            if var_ct[k] is None:
                var_ct[k] = xp.zeros(sysm.var_shapes[k], dtype=np.result_type(sysm.dtype, float))
            var_ct[k][comp] = var_ct[k][comp] + c
        return self._scatter_T(var_ct)

    def explain(self) -> str:
        nm = sum(len(m) for P in self.polys if P for m in P.values())
        ng = sum(G is not None for G in self.generals)
        gen = (f", general form (products of linear forms, post-operators on the projected "
               f"products) of {ng} RHS(s)" if ng else "")
        return (f"LinearizedRHS ({'GPU, CuPy' if self.gpu else 'CPU, NumPy'}): polynomial form of "
                f"{sum(P is not None for P in self.polys)} RHS(s){gen}, {nm} monomial(s) depending on "
                "the state; F'ᵀ matrix-free with explicit transposes of the dealiased transforms "
                "(FFT/DCT) and of the spectral derivatives"
                + (f"; time-dependent ({self.tname}): exact split F_0 + sum_g a_g({self.tname}) F_g "
                   f"with a_g = {', '.join(str(a) for a in self.tcoefs)} evaluated at the stage "
                   "times" if self.tcoefs else ""))
