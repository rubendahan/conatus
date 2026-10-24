"""Fused evaluation of polynomial explicit right-hand sides (``f_backend="fused"``).

The IR right-hand sides are compiled **once** into a polynomial form

    F_(eq, comp) = sum_m  c_m  prod_j  D^{a_mj} f_mj ,

where each *factor* D^a f is a field component under a product of 1D spectral derivatives.
Evaluation then does, for all equations at once:

1. factor coefficients: Fourier derivatives are diagonal, Chebyshev derivatives use the
   backward recurrence (one CUDA thread per column; no dense derivative matrix);
2. **one** batched backward transform of the stacked factors to the dealiased grid;
3. **one** generated CUDA kernel computing every output component pointwise (products and
   sums fused);
4. **one** batched forward transform (first N modes kept: the dealiasing projection).

It gives the same numbers as :class:`SpectralEvaluator` (the oracle) up to rounding: the
evaluator also transforms factors to the dealiased grid, multiplies and projects; linear
combinations of products commute with the (linear) forward transform. Derivatives of
products (e.g. div(u u)) are **refused**, not expanded by Leibniz: on the truncated grid
Leibniz would change the numbers. Unsupported nodes raise :class:`SpectralError`; nothing
falls back silently to another evaluator.

**Symmetry subspace** (``reflection=R`` along a RealFourier axis): every factor and output has a
known parity under R, so along that axis only cos (even) or only -sin (odd) coefficients are
non-zero. The factors are then compacted to those K = N/2 coefficients (the Chebyshev
transforms run on half the columns), mapped to the **half grid** x_j, j = 0..M/2 (the other half
is the mirror image), multiplied there (half the points), and mirrored back before the forward
FFT. Same numbers as the full evaluation up to FFT rounding (tested); declared in ``explain()``.

Parameters are frozen at compile time; :meth:`FusedRHS.check_params` recompiles when a
value changes. A time-dependent parameter is refused. On the CPU (``xp=None``) the same
expansion runs with NumPy; on the GPU the kernels are CuPy RawKernels for now (capturable in a CUDA graph, no allocation-dependent host sync); moving them into
``core/`` is a packaging step, not a change of algorithm.

The symbolic expansion (polynomial and general forms) is in :mod:`conatus.timestep.fused_expand`;
this module re-exports its names.
"""
from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import numpy as np

from ..ir import expr as E
from ..ir.bases import Chebyshev, ComplexFourier, RealFourier
from . import xp_transforms as XT
from .spectral import SpectralError
from .fused_expand import (  # noqa: F401 (re-export: moved to fused_expand.py)
    Factor, Mono, _add_into, _akey, _comps, _cross_monos, _fkey, _form, _formkey, _mul_monos, _peel,
    expand_general, expand_polynomial,
)

__all__ = ["FusedRHS", "expand_polynomial", "expand_general"]

_DIFF_SRC = r"""
#include <cupy/complex.cuh>
// Chebyshev derivative, coefficient index = slow axis (stride ncol): coalesced across threads.
template <typename T>
__device__ void cheb_d(const T* c, T* o, long long ncol, int N, double s) {
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ncol) return;
    T p2 = T(0), p1 = T(0);                 // out[n+2], out[n+1] (unscaled)
    o[(long long)(N - 1) * ncol + col] = T(0);
    for (int n = N - 2; n >= 0; --n) {
        T v = p2 + T(2.0 * (n + 1)) * c[(long long)(n + 1) * ncol + col];
        p2 = p1; p1 = v;
        o[(long long)n * ncol + col] = (n == 0 ? T(0.5) * v : v) * T(s);
    }
}
extern "C" __global__ void cheb_d_d(const double* c, double* o, long long nc, int N, double s) { cheb_d<double>(c, o, nc, N, s); }
extern "C" __global__ void cheb_d_z(const complex<double>* c, complex<double>* o, long long nc, int N, double s) { cheb_d<complex<double>>(c, o, nc, N, s); }
"""
_diff_mod = None


_CHEB3_SRC = r"""
#include <cupy/complex.cuh>
// Chebyshev derivative along the middle axis of an (A, N, B) array, plus an optional addend:
// o = s * D c + add. One thread per (a, b) column; b fastest, so a warp reads contiguous memory.
template <typename T>
__device__ void cheb_d3(const T* c, const T* add, T* o, long long A, int N, long long B,
                        double s, int has_add) {
    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= A * B) return;
    long long a = col / B, b = col % B;
    const long long base = a * (long long)N * B + b;
    T p2 = T(0), p1 = T(0);
    o[base + (long long)(N - 1) * B] = has_add ? add[base + (long long)(N - 1) * B] : T(0);
    for (int n = N - 2; n >= 0; --n) {
        T v = p2 + T(2.0 * (n + 1)) * c[base + (long long)(n + 1) * B];
        p2 = p1; p1 = v;
        T r = (n == 0 ? T(0.5) * v : v) * T(s);
        o[base + (long long)n * B] = has_add ? r + add[base + (long long)n * B] : r;
    }
}
extern "C" __global__ void cheb_d3_d(const double* c, const double* add, double* o, long long A,
                                     int N, long long B, double s, int h) { cheb_d3<double>(c, add, o, A, N, B, s, h); }
extern "C" __global__ void cheb_d3_z(const complex<double>* c, const complex<double>* add,
                                     complex<double>* o, long long A, int N, long long B, double s,
                                     int h) { cheb_d3<complex<double>>(c, add, o, A, N, B, s, h); }
"""
_cheb3_mod = None


class _LinearCombo:
    """out = sum_t c_t D^{a_t} X[src_t]: spectral derivatives of stacked coefficient arrays X
    (inputs of one full shape), with at most one Chebyshev axis. Evaluated by Horner in the
    Chebyshev order, out = S_0 + D_y(S_1 + D_y(S_2 + ...)), where S_b gathers the terms with b
    Chebyshev derivatives and applies their Fourier derivatives as (paired, for RealFourier)
    multipliers: one generated kernel per S_b and one Chebyshev pass per level on the GPU,
    instead of one pass per derivative and per term. Same numbers as repeated ``_diff`` up to
    rounding (tested)."""

    def __init__(self, owner: FusedRHS, terms: Sequence[tuple], cplx: bool) -> None:
        self.o, self.xp = owner, owner.xp
        self.cy = next((i for i, b in enumerate(owner.bases) if isinstance(b, Chebyshev)), None)
        self.cplx = cplx
        levels: dict = {}
        for c, src, a in terms:
            b = a[self.cy] if self.cy is not None else 0
            fa = tuple(0 if i == self.cy else n for i, n in enumerate(a))
            levels.setdefault(b, []).append((c, src, fa))
        self.levels = levels
        self.top = max(levels) if levels else 0
        self._k = {}
        if owner._gpu and levels:
            self._build_kernels()

    # -- GPU ------------------------------------------------------------------------------
    def _tables(self):
        """Per Fourier axis: derivative weight tables (RealFourier: (amax+1, 2, N) partner
        weights; ComplexFourier: (amax+1, N) values of (ik)^a)."""
        xp, o = self.xp, self.o
        amax = {i: max((fa[i] for lv in self.levels.values() for _, _, fa in lv), default=0)
                for i, b in enumerate(o.bases) if i != self.cy}
        tabs = {}
        for i, am in amax.items():
            b = o.bases[i]
            N = b.size
            if isinstance(b, RealFourier):
                k = 2 * np.pi * (np.arange(N) // 2) / b.length
                par = np.arange(N) % 2
                T = np.zeros((am + 1, 2, N))
                for a in range(am + 1):
                    w = 1j ** a
                    re, im = w.real, w.imag
                    ka = k ** a
                    # out_cos = k^a (re c - im s), out_sin = k^a (im c + re s)
                    T[a, 0] = np.where(par == 0, re, im) * ka
                    T[a, 1] = np.where(par == 0, -im, re) * ka
                tabs[i] = ("real", am, xp.asarray(T.reshape(-1)))
            else:
                k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / b.length
                k[N // 2] = 0.0
                T = np.stack([(1j * k) ** a for a in range(am + 1)])
                tabs[i] = ("complex", am, xp.asarray(T.reshape(-1)))
        return tabs

    def _build_kernels(self):
        xp, o = self.xp, self.o
        shape = tuple(b.size for b in o.bases)
        nd = len(shape)
        self._tabs = self._tables()
        T = "complex<double>" if self.cplx else "double"
        tab_axes = sorted(self._tabs)
        args = [f"const {T}* X", f"{T}* out", "long long P"]
        for i in tab_axes:
            kind = self._tabs[i][0]
            args.append(("const double* " if kind == "real" else "const complex<double>* ") + f"t{i}")
        for b, terms in self.levels.items():
            lines = ["#include <cupy/complex.cuh>",
                     'extern "C" __global__ void lvl(' + ", ".join(args) + ") {",
                     "  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
                     "  if (i >= P) return;", "  long long r = i;"]
            for ax in range(nd - 1, -1, -1):
                lines.append(f"  const long long i{ax} = r % {shape[ax]}LL; r /= {shape[ax]}LL;")
            lines.append(f"  {T} acc = {T}(0);")

            def at(src, idx):
                flat = "0"
                for ax in range(nd):
                    flat = f"({flat}) * {shape[ax]}LL + ({idx[ax]})"
                return f"X[{src}LL * P + {flat}]"

            def expr(src, fa, ax, idx):
                if ax == nd:
                    return at(src, idx)
                a = fa[ax]
                if ax == self.cy or a == 0:
                    return expr(src, fa, ax + 1, idx)
                kind, am, _ = self._tabs[ax]
                N = shape[ax]
                if kind == "complex":
                    return f"t{ax}[{a}LL * {N}LL + i{ax}] * " + expr(src, fa, ax + 1, idx)
                parts = []
                for q in (0, 1):
                    idq = list(idx)
                    idq[ax] = f"((i{ax} & ~1LL) + {q})"
                    parts.append(f"t{ax}[({a}LL * 2 + {q}) * {N}LL + i{ax}] * ("
                                 + expr(src, fa, ax + 1, idq) + ")")
                return "(" + " + ".join(parts) + ")"
            for c, src, fa in terms:
                cs = (f"{T}({float(np.real(c))!r}, {float(np.imag(c))!r})" if self.cplx
                      else f"{float(np.real(c))!r}")
                lines.append(f"  acc += {cs} * " + expr(src, fa, 0, [f"i{ax}" for ax in range(nd)]) + ";")
            lines += ["  out[i] = acc;", "}"]
            k = xp.RawKernel("\n".join(lines), "lvl", options=("-std=c++17",))
            k.compile()
            self._k[b] = k
        global _cheb3_mod
        if _cheb3_mod is None:
            _cheb3_mod = xp.RawModule(code=_CHEB3_SRC, options=("-std=c++17",))
        self._cheb = _cheb3_mod.get_function("cheb_d3_z" if self.cplx else "cheb_d3_d")
        self._tab_args = [self._tabs[i][2] for i in tab_axes]

    def _level_gpu(self, b, X, out):
        P = int(np.prod(X.shape[1:]))
        self._k[b](((P + 255) // 256,), (256,), (X, out, np.int64(P), *self._tab_args))
        return out

    def _cheb_add_gpu(self, c, add, out):
        shape = c.shape
        A = int(np.prod(shape[:self.cy]))
        N = shape[self.cy]
        B = int(np.prod(shape[self.cy + 1:]))
        s = 2.0 / self.o.bases[self.cy].length
        ncol = A * B
        h = add is not None
        self._cheb(((ncol + 255) // 256,), (256,),
                   (c, add if h else c, out, np.int64(A), np.int32(N), np.int64(B), np.float64(s),
                    np.int32(1 if h else 0)))
        return out

    # -- evaluation -----------------------------------------------------------------------
    def __call__(self, X, out=None):
        """X: stacked inputs (n_in, *shape), contiguous. Returns the combination (shape)."""
        xp, o = self.xp, self.o
        shape = X.shape[1:]
        dt = complex if self.cplx else X.dtype
        if out is None:
            out = xp.empty(shape, dtype=dt)
        if not self.levels:
            out[...] = 0
            return out
        if o._gpu:
            acc = None
            for b in range(self.top, -1, -1):
                S = None
                if b in self.levels:
                    S = self._level_gpu(b, X, xp.empty(shape, dtype=dt) if acc is not None or b else out)
                if acc is None:
                    acc = S if S is not None else xp.zeros(shape, dtype=dt)
                    continue
                dst = out if b == 0 else xp.empty(shape, dtype=dt)
                acc = self._cheb_add_gpu(acc, S, dst)
            if acc is not out:
                out[...] = acc
            return out
        acc = None
        for b in range(self.top, -1, -1):
            S = None
            for c, src, fa in self.levels.get(b, []):
                v = X[src]
                for ax, n in enumerate(fa):
                    for _ in range(n):
                        v = o._diff(v, ax)
                S = c * v if S is None else S + c * v
            if acc is None:
                acc = S if S is not None else np.zeros(shape, dtype=dt)
            else:
                acc = o._diff(acc, self.cy)
                if S is not None:
                    acc = acc + S
        out[...] = acc
        return out


class FusedRHS:
    """Compiled right-hand sides of all equations (see module docstring)."""

    def __init__(self, domain, equations: Sequence, rhs_exprs: Sequence, params: Mapping[str, Any],
                 xp: Any = None, dtype=float, reflection=None, symmetry=None) -> None:
        self.xp = np if xp is None else xp
        self._gpu = self.xp is not np
        self.domain = domain
        self.coords = list(domain.coordsys.coords)
        self.bases = [domain.basis(c) for c in self.coords]
        for b in self.bases:
            if b is None or not isinstance(b, (Chebyshev, RealFourier, ComplexFourier)):
                raise SpectralError("fused RHS: every coordinate needs a Fourier/Chebyshev basis")
        self.equations, self.rhs_exprs = list(equations), list(rhs_exprs)
        self.dtype = np.dtype(dtype)
        # transform_batch (declared in explain()): at most this many fields per batched
        # transform of the polynomial form (the factor stack backward, the outputs forward), the
        # results copied into the full stack; None: one batch of all fields. Smaller batches cut
        # the transform temporaries (complex spectra, padded Chebyshev FFTs) in proportion; the
        # numbers are those of the one-batch path (same per-field transforms; equal to rounding
        # on the GPU, where cuFFT plans differ with the batch size)
        self.transform_batch: int | None = None
        # use_ffc (opt-in): batched 3D transforms in the polynomial form
        self._ffc = None
        # parity-reduced evaluation along one reflected RealFourier axis (None: full evaluation)
        self.reflection, self.symmetry, self.rax = None, None, None
        if reflection is not None:
            rax = self.coords.index(reflection.coord)
            b = self.bases[rax]
            if not isinstance(b, RealFourier):
                raise SpectralError("fused RHS: parity reduction needs a RealFourier axis")
            if b.grid_size % 2 or b.size % 2:
                raise SpectralError("fused RHS: parity reduction needs even N and grid size")
            self.reflection, self.symmetry, self.rax = reflection, symmetry, rax
        self._compile(dict(params))

    # ------------------------------------------------------------------ compile
    def _compile(self, params: dict) -> None:
        self.params = params
        self.mode = "polynomial"
        try:
            self._compile_polynomial(params)
        except SpectralError as err:
            # derivatives of products (divergence form, curl curl of -div(u u), ...): general
            # form (products of linear forms, post-operators on the projected products)
            if "derivative of a product" not in str(err):
                raise
            self.mode = "general"
            self._compile_general(params)
        self.cplx = self.dtype.kind == "c" or any(np.iscomplexobj(c) for *_, m in self.outputs
                                                   for c, _ in m)
        if self.mode == "general":
            self.cplx = self.cplx or any(np.iscomplexobj(c) for terms in self.post_plan.values()
                                         for c, _, _ in terms)
        if self.reflection is not None:
            if self.mode == "general":
                raise SpectralError("fused RHS: parity reduction not implemented for the general "
                                    "form (post-operators on products)")
            self._compile_parities()
        self._classify_low()
        self._kernel = None
        self._src = self._kernel_source()
        self._prepare_device()

    def _classify_low(self) -> None:
        """Low-dimensional factors: a factor whose field lives on a sub-domain (an NCC
        profile U(y), a time coefficient of StageTimeFused on the empty domain, a mean-flow unknown)
        is constant along the missing axes. It is transformed on its own axes only and read by the
        kernel through a broadcast index, instead of being embedded in a full 3D coefficient array
        and transformed in 3D (polynomial form without parity reduction; declared in explain()).
        ``self.low``: per factor, None (full) or (dependence mask, grid shape, offset in the
        buffer of low grids); ``self.full_index``: factor -> row of the stacked full grids."""
        nd = len(self.bases)
        self.low, self.full_index, self.low_size = [], [], 0
        self._low_on = self.mode == "polynomial" and self.reflection is None and nd >= 2
        for f, _comp, _a in (self.factor_list if self._low_on else []):
            dep = tuple(f.domain.has(c) for c in self.coords)
            if all(dep):
                self.low.append(None)
                self.full_index.append(sum(x is None for x in self.low) - 1)
                continue
            gshape = tuple(b.grid_size if d else 1 for b, d in zip(self.bases, dep, strict=True))
            self.low.append((dep, gshape, self.low_size))
            self.full_index.append(-1)
            self.low_size += int(np.prod(gshape))
        if not self._low_on or not any(x is not None for x in self.low):
            self._low_on = False
            self.low, self.full_index = [None] * len(self.factor_list), list(range(len(self.factor_list)))

    def use_ffc(self, on: bool = True) -> None:
        """        the channel): batched 3D transforms (``xp_transforms.FourierFourierChebyshev``, already
        used by the general form) instead of axis by axis, i.e. far fewer memory passes; the same
        transforms to rounding (tested). Grid layout (Chebyshev, Fourier, Fourier), internal.
        Opt-in, declared in explain(); refused where it does not apply."""
        if on:
            kinds = [type(b) for b in self.bases]
            if not (self.mode == "polynomial" and self.reflection is None and len(self.bases) == 3
                    and kinds.count(RealFourier) == 2 and kinds.count(Chebyshev) == 1
                    and self.dtype.kind != "c" and not self.cplx):
                raise SpectralError("use_ffc: needs the polynomial form of a real RHS on two "
                                    "RealFourier axes and one Chebyshev axis, without parity reduction")
            self._ffc = XT.FourierFourierChebyshev(self.xp, self.bases, kinds.index(Chebyshev))
        else:
            self._ffc = None
        self._src = self._kernel_source()
        self._prepare_device()

    def _gaxes(self) -> tuple:
        """Domain axes in the order of the grid layout of the polynomial evaluation."""
        if self._ffc is not None:
            return (self._ffc.cy, *self._ffc.fa)
        return tuple(range(len(self.bases)))

    def set_low_factors(self, on: bool) -> None:
        """Turn the low-dimensional factor path on (default where it applies) or off (every
        factor embedded in 3D, the reference) and rebuild the kernel."""
        if on:
            self._classify_low()
        else:
            self._low_on = False
            self.low = [None] * len(self.factor_list)
            self.full_index = list(range(len(self.factor_list)))
        self._src = self._kernel_source()
        self._prepare_device()

    def _compile_general(self, params: dict) -> None:
        """General form (see ``expand_general``): distinct linear forms (one backward transform
        each), grid outputs (either one per distinct product, or one per (output, post-operator)
        sum, whichever is fewer: forward transforms), post-operators per equation output."""
        fourier_axes = [i for i, b in enumerate(self.bases) if not isinstance(b, Chebyshev)]
        forms: dict = {}
        atoms: dict = {}
        per_out: dict = {}              # (k, comp) -> [(c, post, form indices)]
        self.const_terms: dict = {}     # (k, comp) -> constant (added to the zero mode)
        for k, (eq, e) in enumerate(zip(self.equations, self.rhs_exprs, strict=True)):
            if e is None or not e.free_fields():
                continue
            gen = expand_general(e, self.coords, params, fourier_axes)
            for comp in _comps(eq.lhs):
                lst = []
                for c, post, fs in gen.get(comp, []):
                    if not fs:
                        if any(post[0]) or any(post[1]):
                            raise SpectralError("fused RHS: post-operator on a constant")
                        self.const_terms[(k, comp)] = self.const_terms.get((k, comp), 0.0) + c
                        continue
                    ids = []
                    for f in fs:
                        key = _formkey(f)
                        if key not in forms:
                            forms[key] = f
                            for _, a in f:
                                if a is not None:
                                    atoms.setdefault(_akey(a), a)
                        ids.append(list(forms).index(key))
                    lst.append((c, post, tuple(sorted(ids))))
                per_out[(k, comp)] = lst
        self.forms = list(forms.values())
        self.factor_list = list(atoms.values())          # coefficient-space derivative atoms
        prods = sorted({fs for lst in per_out.values() for _, _, fs in lst})
        by_post = sorted({(k, comp, post) for (k, comp), lst in per_out.items()
                          for _, post, _ in lst}, key=str)
        self.grouping = "by product" if len(prods) < len(by_post) else "by output"
        self.post_plan = {}
        if self.grouping == "by product":
            pidx = {fs: i for i, fs in enumerate(prods)}
            self.outputs = [(-1, (), [(1.0, fs)]) for fs in prods]
            for key, lst in per_out.items():
                self.post_plan[key] = [(c, pidx[fs], post) for c, post, fs in lst]
        else:
            gidx = {kp: i for i, kp in enumerate(by_post)}
            monos: list = [[] for _ in by_post]
            for (k, comp), lst in per_out.items():
                for c, post, fs in lst:
                    monos[gidx[(k, comp, post)]].append((c, fs))
            self.outputs = [(k, comp, m) for (k, comp, _), m in zip(by_post, monos, strict=True)]
            for (k, comp, post), i in gidx.items():
                self.post_plan.setdefault((k, comp), []).append((1.0, i, post))
        for key in self.const_terms:
            self.post_plan.setdefault(key, [])
        # two RealFourier axes + one Chebyshev axis (the channel): batched 3D transforms with
        # contiguous FFTs (xp_transforms.FourierFourierChebyshev); otherwise axis by axis
        self._fft3 = None
        kinds = [type(b) for b in self.bases]
        if len(self.bases) == 3 and kinds.count(RealFourier) == 2 and kinds.count(Chebyshev) == 1 \
                and self.dtype.kind != "c" and not any(np.iscomplexobj(c) for lst in per_out.values()
                                                       for c, _, _ in lst):
            self._fft3 = XT.FourierFourierChebyshev(self.xp, self.bases, kinds.index(Chebyshev))
        # linear combinations evaluated by _LinearCombo (Horner in the Chebyshev order):
        # the forms (inputs: field components) and the non-integrated post-operators (inputs:
        # the projected grid outputs); integrated terms (mean-flow equations) stay per term
        self._form_inputs = sorted({(a[0].name, a[1]): (a[0], a[1]) for f in self.forms
                                    for _, a in f if a is not None}.items())
        fidx = {k: i for i, (k, _) in enumerate(self._form_inputs)}
        self._form_inputs = [v for _, v in self._form_inputs]
        cform = self.dtype.kind == "c" or any(isinstance(b, ComplexFourier) for b in self.bases)
        self._form_combos = [
            _LinearCombo(self, [(c, fidx[(a[0].name, a[1])], a[2]) for c, a in f if a is not None],
                         cform or any(np.iscomplexobj(c) for c, _ in f))
            for f in self.forms]
        self._post_combos = {}
        for key, terms in self.post_plan.items():
            full = [(c, g, dc) for c, g, (dc, ig) in terms if not any(ig)]
            if full:
                self._post_combos[key] = _LinearCombo(
                    self, full, cform or any(np.iscomplexobj(c) for c, _, _ in full))

    def _compile_polynomial(self, params: dict) -> None:
        factors: dict = {}
        outputs = []            # (eq index, comp, monomials)
        for k, (eq, e) in enumerate(zip(self.equations, self.rhs_exprs, strict=True)):
            if e is None or not e.free_fields():
                continue                  # zero or constant RHS: handled by the caller
            poly = expand_polynomial(e, self.coords, params)
            for comp in _comps(eq.lhs):
                monos = [(c, fs) for c, fs in poly.get(comp, []) if c != 0]
                for _, fs in monos:
                    for f in fs:
                        factors.setdefault(_fkey(f), f)
                outputs.append((k, comp, monos))
        self.factor_list = list(factors.values())
        idx = {_fkey(f): i for i, f in enumerate(self.factor_list)}
        self.outputs = [(k, comp, [(c, tuple(idx[_fkey(f)] for f in fs)) for c, fs in monos])
                        for k, comp, monos in outputs]

    def _compile_parities(self) -> None:
        """Parity under the reflection of every factor (field component parity times (-1)^(number
        of derivatives along the reflected axis)) and of every output (all monomials must agree,
        else the RHS breaks the declared symmetry)."""
        from ..ir.symmetry import SymmetryError, component_parity
        R, rax = self.reflection, self.rax
        self.factor_par = []
        for f, comp, a in self.factor_list:
            p = int(component_parity(self.symmetry.parity(f, R), f.tensorsig, comp, R))
            self.factor_par.append(p * (-1) ** a[rax])
        self.output_par = []
        for k, comp, monos in self.outputs:
            ps = {int(np.prod([self.factor_par[j] for j in fs])) for _, fs in monos}
            if len(ps) > 1:
                raise SymmetryError(f"fused RHS: output of equation {k}{list(comp) if comp else ''} "
                                    "mixes parities under the declared reflection")
            self.output_par.append(ps.pop() if ps else 1)
        if self.cplx:
            raise SpectralError("fused RHS: parity reduction implemented for real RHS only")
        # Device constants for the half-grid transforms (built here, never during a capture).
        xp, nd, rax = self.xp, len(self.bases), self.rax
        K = self.bases[rax].size // 2

        def along(v, lead):                     # (lead, K) -> broadcastable over the array axes
            shp = [lead] + [1] * nd
            shp[rax + 1] = K
            return xp.asarray(v.reshape(shp))

        fp = np.array(self.factor_par)
        k = np.arange(K)
        # backward spectrum: even F_k = a_k/2 (F_0 = a_0), odd F_k = i s_k/2 (F_0 = 0)
        w = np.where(fp[:, None] > 0, np.where(k == 0, 1.0, 0.5), np.where(k == 0, 0.0, 0.5j))
        M = self.bases[rax].grid_size          # FFT normalisations folded into the weights
        self._red_wb = along(w.astype(complex) * M, len(fp))
        op = np.array(self.output_par) if self.output_par else np.ones(0)
        # forward: even a_0 = Re F_0, a_k = 2 Re F_k ; odd s_0 = 0, s_k = 2 Im F_k
        we = np.where(op[:, None] > 0, np.where(k == 0, 1.0, 2.0), 0.0)
        wo = np.where(op[:, None] < 0, np.where(k == 0, 0.0, 2.0), 0.0)
        self._red_we, self._red_wo = along(we / M, len(op)), along(wo / M, len(op))
        sgn = np.ones([len(op)] + [1] * nd)
        sgn[:, ...] = op.reshape([len(op)] + [1] * nd)
        self._red_sgn = xp.asarray(sgn)
        kshape = [1] * nd
        kshape[rax] = K
        self._red_k = xp.asarray((2 * np.pi * k / self.bases[rax].length).reshape(kshape))

    def check_params(self, params: Mapping[str, Any]) -> None:
        """Recompile when the parameter values differ from those compiled in."""
        if dict(params) != self.params:
            self._compile(dict(params))

    def _kernel_source(self) -> str:
        T = "complex<double>" if self.cplx else "double"
        nf = len(self.forms) if self.mode == "general" else len(self.factor_list)
        lines = ["#include <cupy/complex.cuh>",
                 f'extern "C" __global__ void fused_rhs(const {T}* g, {T}* out, long long P) {{',
                 "  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
                 "  if (i >= P) return;"]
        if getattr(self, "_low_on", False):
            lines[1] = (f'extern "C" __global__ void fused_rhs(const {T}* g, const {T}* low, {T}* out, '
                        "long long P) {")
            ga = self._gaxes()                  # grid layout: these domain axes in this order
            M = [self.bases[a].grid_size for a in ga]
            for j in range(nf):
                lw = self.low[j]
                if lw is None:
                    lines.append(f"  const {T} f{j} = g[{self.full_index[j]}LL * P + i];")
                    continue
                dep0, gs0, off = lw
                dep, gs = [dep0[a] for a in ga], [gs0[a] for a in ga]
                terms = []
                for ax in range(len(M)):
                    if not dep[ax]:
                        continue
                    stride = int(np.prod(M[ax + 1:]))
                    lstride = int(np.prod(gs[ax + 1:]))
                    terms.append(f"((i / {stride}LL) % {M[ax]}LL) * {lstride}LL")
                idx = " + ".join(terms) if terms else "0LL"
                lines.append(f"  const {T} f{j} = low[{off}LL + {idx}];")
        else:
            for j in range(nf):
                lines.append(f"  const {T} f{j} = g[{j}LL * P + i];")
        for o, (_, _, monos) in enumerate(self.outputs):
            terms = []
            for c, fs in monos:
                cs = (f"{T}({float(np.real(c))!r}, {float(np.imag(c))!r})" if self.cplx
                      else f"{float(c)!r}")
                terms.append("*".join([cs] + [f"f{j}" for j in fs]))
            lines.append(f"  out[{o}LL * P + i] = " + (" + ".join(terms) if terms else f"{T}(0)") + ";")
        lines.append("}")
        return "\n".join(lines)

    # Tolerance of the real projection on RealFourier grids (same rule as spectral.py): a larger
    # imaginary part is an error, never dropped silently.
    IMAG_TOL = 1e-13

    def _low_grids(self, data, shape, cdt):
        """Grid values of the low-dimensional factors (see _classify_low), concatenated: each
        factor's coefficients on its own axes (zero-padded to the basis size, derivatives along
        them, zero for a derivative along a missing axis), transformed along those axes only; a
        missing axis holds the zero mode, whose basis function is 1 (cos 0, T_0)."""
        xp = self.xp
        nd = len(self.bases)
        out = xp.zeros(self.low_size, dtype=cdt)
        for (f, comp, a), lw in zip(self.factor_list, self.low, strict=True):
            if lw is None:
                continue
            dep, gshape, off = lw
            if any(n and not d for n, d in zip(a, dep, strict=True)):
                continue                       # derivative along an axis it does not vary on
            c = xp.asarray(data[f])
            if comp:
                c = c[comp]
            c = c.reshape(c.shape + (1,) * (nd - c.ndim))
            cs = tuple(b.size if d else 1 for b, d in zip(self.bases, dep, strict=True))
            if any(n > m for n, m in zip(c.shape, cs, strict=True)):
                raise SpectralError(f"fused RHS: factor {f.name} has coefficient shape {c.shape}, "
                                    f"larger than its axes {cs}")
            if c.shape != cs:
                full = xp.zeros(cs, dtype=np.result_type(c.dtype, float))
                full[tuple(slice(0, n) for n in c.shape)] = c
                c = full
            for ax in range(nd):
                for _ in range(a[ax]):
                    c = self._diff(c, ax)
            g = c[None]
            for ax in self._backward_order:
                if dep[ax]:
                    g = XT.backward(xp, g, self.bases[ax], axis=ax + 1)
            if isinstance(self.bases[0], RealFourier) and xp.iscomplexobj(g) and not self.cplx:
                g = self._real_projection(g, f"inverse transform of the low-dimensional factor {f.name}")
            ga = self._gaxes()
            if ga != tuple(range(nd)):           # into the grid layout of the evaluation
                g = xp.transpose(g, (0,) + tuple(1 + a for a in ga))
            n = int(np.prod(gshape))
            out[off:off + n] = g.reshape(-1)
        return out

    def _batched(self, A, fn):
        """fn(A) on the leading (field) axis in slices of at most ``transform_batch`` fields, the
        results written into one array (``transform_batch`` None or >= the number of fields: fn(A))."""
        k = self.transform_batch
        n = A.shape[0]
        if not k or n <= k:
            return fn(A)
        out = None
        for i0 in range(0, n, k):
            r = fn(A[i0:i0 + k])
            if out is None:
                out = self.xp.empty((n,) + r.shape[1:], dtype=r.dtype)
            out[i0:i0 + r.shape[0]] = r
            del r
        return out

    def _real_projection(self, A, where: str):
        """Real part of ``A``, checked: |Im A| <= IMAG_TOL * max(1, |Re A|). CPU: checked now.
        GPU: the ratio is accumulated on the device (no host sync, capturable in a CUDA graph) and
        checked by :meth:`check_real_projection`, which the integrator calls after every step or
        replayed block."""
        xp = self.xp
        if not self._gpu:
            im = float(np.max(np.abs(A.imag))) if A.size else 0.0
            re = float(np.max(np.abs(A.real))) if A.size else 0.0
            if im > self.IMAG_TOL * max(1.0, re):
                raise SpectralError(f"fused RHS: imaginary part {im:.1e} on a RealFourier grid "
                                    f"({where}); refusing to drop it")
            return A.real
        ratio = xp.max(xp.abs(A.imag)) / xp.maximum(1.0, xp.max(xp.abs(A.real)))
        buf = getattr(self, "_imag_ratio", None)
        if buf is None:
            buf = self._imag_ratio = xp.zeros((), dtype=np.float64)
        xp.maximum(buf, ratio, out=buf)
        return A.real

    def check_real_projection(self) -> None:
        """GPU: raise if any real projection since the last check exceeded IMAG_TOL (host sync)."""
        buf = getattr(self, "_imag_ratio", None)
        if buf is None:
            return
        r = float(buf)
        buf[...] = 0.0
        if r > self.IMAG_TOL:
            raise SpectralError(f"fused RHS: imaginary part {r:.1e} (relative) on a RealFourier "
                                "grid; refusing to drop it")

    def explain(self) -> str:
        """Factors, outputs, transforms and kernel of the fused right-hand side."""
        lines = [f"FusedRHS ({self.mode} form): {len(self.factor_list)} factors -> {len(self.outputs)} outputs, "
                 f"{'complex' if self.cplx else 'real'}, one batched transform each way, one fused kernel"]
        for f, comp, a in self.factor_list:
            d = "".join(f"d{c.name}^{n}" for c, n in zip(self.coords, a, strict=True) if n)
            lines.append(f"  factor {d or ''}{f.name}{list(comp) if comp else ''}")
        for k, comp, monos in self.outputs:
            what = "product" if k < 0 else f"eq {k}{list(comp) if comp else ''}"
            lines.append(f"  {what}: {len(monos)} monomials")
        if self.mode == "general":
            lines.append(f"  general form: {len(self.forms)} linear forms (one backward transform "
                         f"each) of {len(self.factor_list)} derivative atoms; {len(self.outputs)} "
                         f"grid outputs ({self.grouping}); post-operators (spectral derivatives, "
                         "Fourier integrals) applied to the projected products, as the evaluator "
                         "does (no Leibniz rewrite); linear combinations by Horner in the "
                         "Chebyshev order (one generated kernel per level)"
                         + ("; batched 3D transforms (Chebyshev axis last, then a 2D real FFT "
                            "of the two RealFourier axes)" if self._fft3 is not None else ""))
        if self.reflection is not None:
            lines.append(f"  parity-reduced along {self.reflection.coord.name} ({self.reflection!r}): "
                         "cos/-sin compaction, products on the half grid, mirrored forward FFT")
        if getattr(self, "_low_on", False):
            nl = sum(x is not None for x in self.low)
            lines.append(f"  {nl} low-dimensional factor(s) (fields on a sub-domain: profiles, time "
                         "coefficients) transformed on their own axes and broadcast in the kernel, "
                         f"not embedded in 3D; {len(self.factor_list) - nl} full factor(s) transformed")
        if self._ffc is not None:
            lines.append("  batched 3D transforms (Chebyshev axis last, then a 2D real FFT of the two "
                         "RealFourier axes; use_ffc, same transforms to rounding)")
        if self.transform_batch:
            lines.append(f"  transforms batched by at most {self.transform_batch} fields (polynomial form; "
                         "results copied into the full stack: smaller transform temporaries, same "
                         "per-field transforms)")
        lines.append(f"  real projection on RealFourier grids checked: imaginary part <= {self.IMAG_TOL:g} "
                     "relative, error otherwise (GPU: accumulated on the device, checked after each "
                     "step / replayed block)")
        lines.append("  derivatives of products are refused (no Leibniz rewrite); parameters frozen, "
                     "recompiled when they change")
        return "\n".join(lines)

    # ------------------------------------------------------------------ evaluate
    def _full(self, a, comp, shape):
        """Component ``comp`` of ``a`` embedded into the full coefficient shape (size-1 axes =
        zero mode, as in SpectralEvaluator._embed)."""
        xp = self.xp
        a = xp.asarray(a)
        if comp:
            a = a[comp]
        if a.shape == shape:
            return a
        a = a.reshape(a.shape + (1,) * (len(shape) - a.ndim))
        out = xp.zeros(shape, dtype=np.result_type(a.dtype, float))
        out[tuple(slice(0, s) for s in a.shape)] = a
        return out

    def _diff(self, c, ax: int):
        xp, b = self.xp, self.bases[ax]
        if isinstance(b, Chebyshev):
            # c'_n = c'_{n+2} + 2(n+1) c_{n+1} (halved at n = 0). GPU: one thread per column
            # with the Chebyshev index as the *slow* axis, so a warp reads contiguous memory
            # (the first version, index as fast axis, had uncoalesced reads; parallel scans cost
            # more launches than they saved at these sizes).
            cm = xp.ascontiguousarray(xp.moveaxis(c, ax, 0))
            N = cm.shape[0]
            s = 2.0 / b.length
            if self._gpu:
                out = xp.empty_like(cm)
                ncol = cm.size // N
                k = self._cheb_k["z" if cm.dtype.kind == "c" else "d"]
                k(((ncol + 255) // 256,), (256,), (cm, out, np.int64(ncol), np.int32(N), np.float64(s)))
            else:
                out = np.zeros_like(cm)
                if N > 1:
                    out[N - 2] = 2 * (N - 1) * cm[N - 1]
                    for n in range(N - 3, -1, -1):
                        out[n] = out[n + 2] + 2 * (n + 1) * cm[n + 1]
                    out[0] = out[0] / 2
                out = out * s
            return xp.moveaxis(out, 0, ax)
        mult = self._fourier_mult[ax]
        if isinstance(b, ComplexFourier):
            return mult * c
        cm = xp.moveaxis(c, ax, 0)
        sw = xp.empty_like(cm)
        sw[0::2] = -cm[1::2]
        sw[1::2] = cm[0::2]
        return mult * xp.moveaxis(sw, 0, ax)

    def _prepare_device(self) -> None:
        """Device constants and kernels built up front: nothing is uploaded or compiled during
        a CUDA-graph capture (H2D copies are forbidden while capturing)."""
        xp = self.xp
        nd = len(self.bases)
        self._backward_order = sorted(range(nd), key=lambda i: not isinstance(self.bases[i], Chebyshev))
        self._fourier_mult = {}
        for ax, b in enumerate(self.bases):
            shp = [1] * nd
            shp[ax] = b.size
            N = b.size
            if isinstance(b, ComplexFourier):
                k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / b.length
                k[N // 2] = 0.0
                self._fourier_mult[ax] = xp.asarray((1j * k).reshape(shp))
            elif isinstance(b, RealFourier):
                k = 2 * np.pi * np.arange(N // 2) / b.length
                self._fourier_mult[ax] = xp.asarray(np.repeat(k, 2).reshape(shp))
        if self._gpu:
            global _diff_mod
            if _diff_mod is None:
                _diff_mod = xp.RawModule(code=_DIFF_SRC, options=("-std=c++17",))
            self._cheb_k = {"d": _diff_mod.get_function("cheb_d_d"), "z": _diff_mod.get_function("cheb_d_z")}
            self._kernel = xp.RawKernel(self._src, "fused_rhs", options=("-std=c++17",))
            self._kernel.compile()
            for b in self.bases:          # twiddles of the xp transforms, cached before capture
                XT._twiddle(xp, b.grid_size, b.size, 1.0)
                XT._twiddle(xp, b.grid_size, b.size, -1.0)

    def _deriv_memo(self, data, shape):
        """Factor coefficients (full layout), derivatives peeled one at a time and shared."""
        memo: dict = {}

        def deriv(f, comp, a):
            base = (f.name, comp)
            z = (0,) * len(a)
            if base + (z,) not in memo:
                memo[base + (z,)] = self._full(data[f], comp, shape)
            for prev, cur, ax in _peel(a):      # peel one derivative off the last axis
                if base + (cur,) not in memo:
                    memo[base + (cur,)] = self._diff(memo[base + (prev,)], ax)
            return memo[base + (tuple(a),)]
        return deriv

    def _deriv_memo_reduced(self, data, shape):
        """Compact factor coefficients along the reflected axis (cos for even, -sin for odd,
        K = N/2 entries), derivatives taken in compact form: along the reflected axis
        d/dx maps even a_k -> odd k a_k and odd s_k -> even -k s_k (same numbers as the
        full-layout derivative); other axes as in the full path, on half the columns."""
        from ..ir.symmetry import component_parity
        R, r = self.reflection, self.rax
        memo: dict = {}

        def base(f, comp):
            p = int(component_parity(self.symmetry.parity(f, R), f.tensorsig, comp, R))
            full = self._full(data[f], comp, shape)
            return full[tuple([slice(None)] * r + [slice(0 if p > 0 else 1, None, 2)])], p

        def rec(f, comp, a):                 # -> compact coefficients (iterative, see _peel)
            key0 = (f.name, comp)
            if key0 + ((0,) * len(a),) not in memo:
                memo[key0 + ((0,) * len(a),)] = base(f, comp)
            for prev, cur, ax in _peel(a):
                if key0 + (cur,) not in memo:
                    c, p = memo[key0 + (prev,)]
                    memo[key0 + (cur,)] = (((1.0 if p > 0 else -1.0) * self._red_k * c, -p)
                                           if ax == r else (self._diff(c, ax), p))
            return memo[key0 + (tuple(a),)][0]
        return rec

    def _pointwise(self, G):
        xp = self.xp
        P = int(np.prod(G.shape[1:]))
        no = len(self.outputs)
        out_dtype = complex if self.cplx else G.dtype
        O = xp.empty((no,) + G.shape[1:], dtype=out_dtype)
        if self._gpu:
            Gk = G.astype(out_dtype, copy=False) if self.cplx else G
            self._kernel(((P + 255) // 256,), (256,), (Gk, O, np.int64(P)))
        else:
            flat = G.reshape(G.shape[0], -1)
            for o, (_, _, monos) in enumerate(self.outputs):
                acc = np.zeros(P, dtype=out_dtype)
                for c, fs in monos:
                    t = c
                    for j in fs:
                        t = t * flat[j]
                    acc = acc + t
                O[o] = acc.reshape(G.shape[1:])
        return O

    def _evaluate_reduced(self, data: Mapping[E.Field, Any]) -> list:
        """Parity-reduced evaluation (see module docstring): compact cos/-sin coefficients along
        the reflected axis, half-grid products, mirrored forward FFT."""
        xp, r = self.xp, self.rax
        shape = tuple(b.size for b in self.bases)
        b = self.bases[r]
        N, M = b.size, b.grid_size
        K, H = N // 2, M // 2
        rs = lambda sl: tuple([slice(None)] * (r + 1) + [sl])        # noqa: E731 (axis r of stacked arrays)
        deriv = self._deriv_memo_reduced(data, shape)
        nf = len(self.factor_list)
        G = xp.empty((nf,) + shape[:r] + (K,) + shape[r + 1:], dtype=float)
        for j, (f, comp, a) in enumerate(self.factor_list):
            G[j] = deriv(f, comp, a)
        for ax in self._backward_order:                  # other axes first (half the columns)
            if ax != r:
                G = XT.backward(xp, G, self.bases[ax], axis=ax + 1)
        S = xp.zeros(G.shape[:r + 1] + (H + 1,) + G.shape[r + 2:], dtype=complex)
        S[rs(slice(0, K))] = G * self._red_wb                     # (weights include the factor M)
        G = xp.ascontiguousarray(xp.fft.irfft(S, n=M, axis=r + 1)[rs(slice(0, H + 1))])
        O = self._pointwise(G)                           # half grid: j = 0..M/2
        ext = xp.concatenate([O, self._red_sgn * xp.flip(O[rs(slice(1, H))], axis=r + 1)], axis=r + 1)
        Fh = xp.fft.rfft(ext, axis=r + 1)[rs(slice(0, K))]
        C = Fh.real * self._red_we + Fh.imag * self._red_wo        # (weights include 1/M)
        for ax in self._backward_order[::-1]:
            if ax != r:
                C = XT.forward(xp, C, self.bases[ax], axis=ax + 1)
        res: list = [None] * len(self.equations)
        for o, (k, comp, _) in enumerate(self.outputs):
            eq = self.equations[k]
            if res[k] is None:
                cs = tuple(cs.dim for cs in eq.lhs.tensorsig)
                res[k] = xp.zeros(cs + shape, dtype=C.dtype)
            sl = tuple([slice(None)] * r + [slice(0 if self.output_par[o] > 0 else 1, None, 2)])
            res[k][comp][sl] = C[o]
        return res

    def _evaluate_general(self, data: Mapping[E.Field, Any]) -> list:
        """General form: linear forms in coefficient space -> one batched backward transform ->
        one fused kernel (grid outputs) -> one batched forward transform -> post-operators
        (spectral derivatives, integrals along Fourier axes) per equation output."""
        xp = self.xp
        shape = tuple(b.size for b in self.bases)
        nd = len(shape)
        cdt = complex if (self.cplx or any(isinstance(b, ComplexFourier) for b in self.bases)) else float
        X = xp.stack([self._full(data[f], comp, shape).astype(cdt, copy=False)
                      for f, comp in self._form_inputs]) if self._form_inputs else None
        G = xp.empty((len(self.forms),) + shape, dtype=cdt)
        zero = (0,) * nd
        for j, (form, combo) in enumerate(zip(self.forms, self._form_combos, strict=True)):
            combo(X, out=G[j])
            for c, a in form:
                if a is None:
                    G[(j,) + zero] += c                 # constant: the zero mode (T_0, k = 0)
        if self._fft3 is not None:            # real-to-complex FFTs: real grid by construction
            G = xp.ascontiguousarray(self._fft3.backward(G))
            O = xp.ascontiguousarray(self._fft3.forward(self._pointwise(G)))
        else:
            for ax in self._backward_order:
                G = XT.backward(xp, G, self.bases[ax], axis=ax + 1)
            if isinstance(self.bases[0], RealFourier) and xp.iscomplexobj(G) and not self.cplx:
                G = self._real_projection(G, "inverse transform of the linear forms")
            G = xp.ascontiguousarray(G)
            O = self._pointwise(G)
            for ax in self._backward_order[::-1]:
                b = self.bases[ax]
                if isinstance(b, RealFourier) and xp.iscomplexobj(O):
                    O = self._real_projection(O, "products before the forward RealFourier transform")
                O = XT.forward(xp, O, b, axis=ax + 1)
            O = xp.ascontiguousarray(O)
        memo: dict = {}

        def post(g: int, dc: tuple, ig: tuple):
            """Integrated terms: integrals first (a zero-mode slice), then the derivatives
            (they commute; the derivative along an integrated axis was dropped at compile).
            Iterative (no recursive closure)."""
            z = (0,) * nd
            if (g, z, ig) not in memo:
                r = O[g]
                for ax in [i for i, f in enumerate(ig) if f]:
                    r = r[tuple(slice(0, 1) if i == ax else slice(None) for i in range(nd))]
                    r = r * self.bases[ax].length
                memo[(g, z, ig)] = r
            for prev, cur, ax in _peel(dc):
                if (g, cur, ig) not in memo:
                    memo[(g, cur, ig)] = self._diff(memo[(g, prev, ig)], ax)
            return memo[(g, tuple(dc), ig)]

        res: list = [None] * len(self.equations)
        for (k, comp), terms in self.post_plan.items():
            eq = self.equations[k]
            eshape = tuple(b.size if eq.domain is not None and eq.domain.has(co) else 1
                           for co, b in zip(self.coords, self.bases, strict=True))
            if res[k] is None:
                cs = tuple(c.dim for c in eq.lhs.tensorsig)
                res[k] = xp.zeros(cs + eshape, dtype=O.dtype)
            acc = res[k][comp] if comp else res[k]
            combo = self._post_combos.get((k, comp))
            if combo is not None:
                if acc.shape != shape:
                    raise SpectralError(f"fused RHS: output of equation {k} has shape {shape}, "
                                        f"expected {acc.shape} (domain mismatch)")
                combo(O, out=acc)
            for c, g, (dc, ig) in terms:
                if not any(ig):
                    continue
                r = post(g, dc, ig)
                if r.shape != eshape:
                    raise SpectralError(f"fused RHS: output of equation {k} has shape {r.shape}, "
                                        f"expected {eshape} (domain mismatch)")
                acc += c * r
            cst = self.const_terms.get((k, comp))
            if cst:
                acc[zero] += cst
        return res

    def evaluate(self, data: Mapping[E.Field, Any]) -> list:
        """Coefficient arrays per equation with a field-dependent RHS (full shapes); ``None``
        for zero or constant right-hand sides (the caller evaluates those once)."""
        if self.mode == "general":
            return self._evaluate_general(data)
        if self.reflection is not None and self.factor_list:
            return self._evaluate_reduced(data)
        xp = self.xp
        shape = tuple(b.size for b in self.bases)
        nf = len(self.factor_list)
        cdt = complex if (self.cplx or any(isinstance(b, ComplexFourier) for b in self.bases)) else float
        lowbuf = None
        if nf and self._low_on:
            lowbuf = self._low_grids(data, shape, cdt)
            nfull = sum(x is None for x in self.low)
            G = xp.empty((max(nfull, 1),) + shape, dtype=cdt)
            deriv = self._deriv_memo(data, shape)
            for j, (f, comp, a) in enumerate(self.factor_list):
                if self.low[j] is None:
                    G[self.full_index[j]] = deriv(f, comp, a)
            if not nfull:
                G[0] = 0.0
            del deriv
        elif nf:
            G = xp.empty((nf,) + shape, dtype=cdt)
            # (field, comp, derivative counts) -> coefficients, shared prefixes; iterative (a
            # recursive closure here was a reference cycle keeping the arrays alive)
            deriv = self._deriv_memo(data, shape)
            for j, (f, comp, a) in enumerate(self.factor_list):
                G[j] = deriv(f, comp, a)
            del deriv
        if nf and self._ffc is not None:          # batched 3D transforms (use_ffc)
            G = xp.ascontiguousarray(self._batched(G, self._ffc.backward))
        elif nf:
            def bw(G):
                # backward: Chebyshev axes first (their 2M FFTs then run on N, not M, Fourier points)
                for ax in self._backward_order:
                    G = XT.backward(xp, G, self.bases[ax], axis=ax + 1)
                if isinstance(self.bases[0], RealFourier) and xp.iscomplexobj(G) and not self.cplx:
                    G = self._real_projection(G, "inverse transform of the factors")
                return G
            G = xp.ascontiguousarray(self._batched(G, bw))
        else:
            gshape = tuple(self.bases[a].grid_size for a in self._gaxes())
            G = xp.zeros((1,) + gshape)
        P = int(np.prod(G.shape[1:]))
        no = len(self.outputs)
        out_dtype = complex if self.cplx else G.dtype
        O = xp.empty((no,) + G.shape[1:], dtype=out_dtype)
        if self._gpu:
            Gk = G.astype(out_dtype, copy=False) if self.cplx else G
            if lowbuf is not None:
                Lk = lowbuf.astype(out_dtype, copy=False) if self.cplx else lowbuf
                self._kernel(((P + 255) // 256,), (256,), (Gk, Lk, O, np.int64(P)))
            else:
                self._kernel(((P + 255) // 256,), (256,), (Gk, O, np.int64(P)))
        else:
            gshape = G.shape[1:]
            if lowbuf is not None:          # full rows of G, low grids broadcast (same products)
                ga = self._gaxes()
                vals = [G[self.full_index[j]] if lw is None else
                        lowbuf[lw[2]:lw[2] + int(np.prod(lw[1]))].reshape([lw[1][a] for a in ga])
                        for j, lw in enumerate(self.low)]
            else:
                vals = list(G)
            for o, (_, _, monos) in enumerate(self.outputs):
                acc = np.zeros(gshape, dtype=out_dtype)
                for c, fs in monos:
                    t = c
                    for j in fs:
                        t = t * vals[j]
                    acc = acc + t
                O[o] = acc

        def fw(O):
            if self._ffc is not None:
                return self._ffc.forward(O)
            for ax in self._backward_order[::-1]:    # forward: Fourier first (M -> N), then Chebyshev
                b = self.bases[ax]
                if isinstance(b, RealFourier) and xp.iscomplexobj(O):
                    O = self._real_projection(O, "products before the forward RealFourier transform")
                O = XT.forward(xp, O, b, axis=ax + 1)
            return O
        O = self._batched(O, fw)
        res: list = [None] * len(self.equations)
        for o, (k, comp, _) in enumerate(self.outputs):
            eq = self.equations[k]
            if res[k] is None:
                cs = tuple(cs.dim for cs in eq.lhs.tensorsig)
                res[k] = xp.zeros(cs + shape, dtype=O.dtype)
            res[k][comp] = O[o]
        return res
