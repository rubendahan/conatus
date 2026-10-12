"""Planned ("fused") right-hand side of the spherical IVPs: ``f_backend="fused"``.

The reference :class:`~conatus.discretize.sphere_eval.SphereEvaluator` walks each right-hand side
expression as written: a sub-expression used twice is transformed twice, a coefficient field
(NCC) is transformed to the grid at every call, ``grad(cos(f t) U_c + sin(f t) U_s)`` goes grid ->
coefficients -> grid at every stage, and the transforms treat the two spin components of a real
vector as independent complex fields. :class:`SphereFusedRHS` evaluates the **same expressions
with the same operators** (it is a subclass of the reference evaluator: products, dot, cross,
derivatives and frame terms are its code) and plans the work once:

1. **Common sub-expressions** (structural identity of the IR) are evaluated once per call of F,
   across all equations (``u``, ``grad(u)`` of ``u.grad u + U.grad u``).
2. **Constants** -- nodes that depend on neither the state nor a parameter (NCC fields, their
   gradients, ...) -- are evaluated once (first call) and kept on the device, on the grid and
   in coefficients.
3. **Linear derivatives of time combinations of constants** are distributed:
   ``grad(sum_g a_g(t) C_g) -> sum_g a_g(t) grad(C_g)`` with ``grad(C_g)`` a cached constant
   (exact algebra; it removes one forward and one backward transform per stage and is more
   accurate: no grid round trip of the sum). Scalar factors a_g(t) are evaluated on the host.
4. **Transforms of real fields** (RealFourier): the two spin components s, -s of a real tensor are
   complex conjugates on the grid, so one inverse FFT gives both (the m = 0 entry of the pair is
   symmetrized, (V_s + conj V_-s) / 2: the projection on real fields); the purely radial
   components (u_r, (grad u)_rr, scalars) are real: half-size real FFTs (irfft / rfft). The
   angular and radial transform matrices are real: products by them are real GEMMs on the
   interleaved (re, im) data (half the flops of the complex GEMM of the reference, no cast), and
   the radial syntheses of a gradient (d/dr, 1/r) are one GEMM.
5. **Products on the GPU**: each product tree (sums, products by scalars, dot, cross, outer,
   components, scalar functions; the r^k factor of the shell rows included) is compiled to
   straight-line code per canonical output component and run as **one generated elementwise
   kernel** that reads the canonical components of its leaves (layout phi-last, contiguous) once
   and writes the canonical components of the result once (``_ProductPlan``; its array
   execution is the oracle of the kernel in the tests). On the CPU the reference array
   operations are faster and are used.
6. On the GPU everything stays on the device (no host synchronization inside F); time factors
   a_g(t) are kernel arguments.
7. **Angular GEMMs on the existing degrees**: Phi_l^{m,s} = 0 exactly for l < max(|m|, |s|),
   so the stored wavenumbers are cut into m-chunks and chunk [m0, m1) multiplies only the
   degrees l >= max(|m0|, |s|) (``ang_chunks``; about 40 % fewer flops at nm >= 64).
8. **CUDA graph replay** of a time-dependent step (``IVPSolver.run_graph``): the GEMMs call
   cuBLAS directly (``discretize._cublas``: CuPy refuses cuBLAS inside a capture) and, during
   the capture, the product kernels read a_g(t) from device slots (:class:`GraphScalars`)
   refreshed by the host before each replay with the values of the plain step.

Same numbers as the reference up to rounding: the summation order of the GEMMs (real instead of
complex), the FFT normalization (``norm="forward"``), the distributed derivatives, the
symmetrized m = 0 pair, the dropped rounding-level imaginary parts of real components and the
per-point evaluation order of the kernels (FMA contraction by NVRTC) change the last bits (tested: tests/spherical/test_sphere_fused.py,
1e-13 per call and over 30 RK443 steps). Real fields only: the expressions are checked at
construction (no complex constant, no complex field); parameter values are checked at each call.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..discretize._xp import array_module
from ..discretize.compiler import DiscretizationError
from ..discretize.sphere_eval import _C, _G, _R2, _X, SphereEvaluator, _flip, comps, embed_coeffs
from ..discretize.sphere_ops import spin_of
from ..errors import UnsupportedError
from ..ir import expr as E

__all__ = ["SphereFusedRHS", "GraphScalars", "ang_chunks_rule", "distribute_derivatives"]

_DERIV = (E.Gradient, E.Divergence, E.Curl, E.Laplacian)
_REAL_NODES = (E.Field, E.Constant, E.Parameter, E.Gradient, E.Divergence, E.Curl, E.Laplacian,
               E.Differentiate, E.Add, E.Mul, E.Dot, E.Outer, E.Cross, E.Power, E.UnaryFunction,
               E.Component, E.Conj)


# ----------------------------------------------------------------------------------------------
# expression planning
# ----------------------------------------------------------------------------------------------
def _linear_terms(a: E.Expr):
    """a = sum_g s_g C_g with s_g a field-free scalar expression (None: 1) and C_g free of
    parameters, or None when a is not of that form."""
    if not a.free_parameters():
        return [(None, a)]
    if not a.free_fields():
        return None
    if isinstance(a, E.Add):
        out = []
        for b in a.args:
            t = _linear_terms(b)
            if t is None:
                return None
            out.extend(t)
        return out
    if isinstance(a, E.Mul):
        fa = [b for b in a.args if b.free_fields()]
        sc = [b for b in a.args if not b.free_fields()]
        if len(fa) != 1 or any(b.rank for b in sc):
            return None
        sub = _linear_terms(fa[0])
        if sub is None:
            return None
        k = sc[0]
        for b in sc[1:]:
            k = k * b
        return [(k if s is None else k * s, C) for s, C in sub]
    return None


def distribute_derivatives(e: E.Expr, state: frozenset) -> tuple[E.Expr, list[str]]:
    """Rewrite grad/div/curl/lap of a time combination of constants (no state field, some
    parameter) as the combination of the derivatives of the constants (exact algebra). Returns
    the new expression and the list of rewrites (for explain())."""
    notes: list[str] = []

    def walk(x: E.Expr) -> E.Expr:
        if not x.args:
            return x
        args = [walk(a) for a in x.args]
        y = x if all(a is b for a, b in zip(args, x.args)) else x.rebuild(args)
        if isinstance(y, _DERIV):
            a = y.args[0]
            if a.free_parameters() and a.free_fields() and not (a.free_fields() & state):
                terms = _linear_terms(a)
                if terms is not None:
                    out = None
                    for s, C in terms:
                        d = y.rebuild([C])
                        t = d if s is None else s * d
                        out = t if out is None else out + t
                    notes.append(f"{y} -> {out}")
                    return out
        return y
    return walk(e), notes


def _check_real(e: E.Expr, where: str) -> None:
    """The fused transforms use the reality of the fields (RealFourier): refuse what could make
    a right-hand side complex (complex constant, complex field, unknown node)."""
    for x in e.walk():
        if not isinstance(x, _REAL_NODES):
            raise UnsupportedError(f"{where}: node {type(x).__name__} ('{x}') in a spherical "
                                   "right-hand side is not supported by f_backend='fused'",
                                   fix="f_backend='evaluator' (the reference evaluator)")
        if isinstance(x, E.Constant) and np.iscomplexobj(x.value) and np.any(np.imag(x.value)):
            raise UnsupportedError(f"{where}: complex constant {x.value} in a right-hand side of "
                                   "real fields", fix="f_backend='evaluator'")
        if isinstance(x, E.Field) and np.dtype(x.dtype).kind == "c":
            raise UnsupportedError(f"{where}: complex field {x.name}: f_backend='fused' uses the "
                                   "reality of the fields", fix="f_backend='evaluator'")


# ----------------------------------------------------------------------------------------------
# real-matrix products on complex data (interleaved re/im, real GEMM)
# ----------------------------------------------------------------------------------------------
def _direct_cublas() -> bool:
    from ..discretize._cublas import available
    return available()


def _gemm(A, B, xp):
    """A @ B for float64 arrays, (P, Q) @ (Q, R) or batched (n, P, Q) @ (n, Q, R) (or (P, Q)
    shared by the batch). GPU: the direct cuBLAS call of ``discretize._cublas`` (usable inside
    a CUDA graph capture, where CuPy refuses cuBLAS) or, if that library cannot be loaded,
    ``cupy.matmul`` (declared by ``explain()``; no graph replay then); CPU: NumPy."""
    from ..discretize._cublas import available, gemm_rm
    if xp is np or not available():
        return xp.matmul(A, B)
    A, B = xp.ascontiguousarray(A, dtype=np.float64), xp.ascontiguousarray(B, dtype=np.float64)
    P, Q = A.shape[-2:]
    R = B.shape[-1]
    n = B.shape[0] if B.ndim == 3 else 1
    if (A.ndim == 3 and A.shape[0] != n) or B.shape[-2] != Q or A.ndim > 3 or B.ndim > 3:
        raise ValueError(f"_gemm: shapes {A.shape} @ {B.shape}")
    C = xp.empty((n, P, R) if B.ndim == 3 else (P, R), dtype=np.float64)
    gemm_rm(xp, P, R, Q, A.data.ptr, Q, P * Q if A.ndim == 3 else 0, B.data.ptr, R, Q * R,
            C.data.ptr, R, P * R, n)
    return C


def _left_real(B, y, xp):
    """B @ y for a real (batched) matrix B (..., P, Q) and complex y (..., Q, R): one real GEMM
    on the interleaved data (..., Q, 2R) (y viewed as float64)."""
    y = xp.ascontiguousarray(y)
    z = _gemm(B, y.view(np.float64), xp)
    return z.view(np.complex128)


def _right_real(y, M, xp):
    """y @ M for complex y (..., Q) and a real matrix M (Q, P): one real GEMM on (2 K, Q) rows
    (real and imaginary parts as separate rows)."""
    sh = y.shape
    K, Q = int(np.prod(sh[:-1])), sh[-1]
    yr = xp.ascontiguousarray(xp.ascontiguousarray(y).view(np.float64).reshape(K, Q, 2).transpose(0, 2, 1))
    z = _gemm(yr.reshape(2 * K, Q), M, xp)                       # (2K, P)
    z = xp.ascontiguousarray(z.reshape(K, 2, M.shape[1]).transpose(0, 2, 1))
    return z.view(np.complex128).reshape(sh[:-1] + (M.shape[1],))


def default_ang_chunks(space, xp) -> int:
    """Number of m-chunks of the angular GEMMs (``ang_chunks``); the rule and its reason are
    :func:`ang_chunks_rule` (declared in explain())."""
    return ang_chunks_rule(space, xp)[0]


# at most 4 chunks: 8 was never faster on A100 (both B2 grids) nor in the L40S micro-benchmark;
# on L40S at 256x127x64, 8 chunks beat 1 by 5 %, 4 unmeasured
# there: cap kept at 4, to be confirmed by a 1/2/4/8 measurement.
_MAX_CHUNKS = 4


def ang_chunks_rule(space, xp) -> tuple:
    """(number of m-chunks, reason) for the angular GEMMs. Chunks drop the exactly-zero terms
    (about 40 % of the flops at nm >= 64 with 4 chunks) for more, smaller GEMMs: worth it where
    FP64 GEMMs bound F -- the CPU and GPUs with reduced FP64 throughput (compute capability
    other than 6.0, 7.0, 8.0, 9.0, 10.0: L40S) -- and, on full-FP64 GPUs (A100, H100), only for
    large GEMMs (>= 1 GFLOP per call; at 128x63x48 on A100 one chunk is faster). Measured on
    A100; the reduced-FP64 branch (L40S, graph step): 1/2/4/8 chunks within 7 % at 128x63x48 (26.0/25.5/26.5/27.3 ms);
    at 256x127x64 4 chunks are the fastest, 9 % faster than 1 and 4 % faster than 8
   , hence the cap of 4. Same numbers either way up to
    rounding (the dropped terms are exact zeros; the summation blocking of the GEMM changes)."""
    nm = len(space.ms)
    if nm < 16 or not space.real:
        return 1, f"nm = {nm} < 16 or complex fields: one GEMM"
    many = min(_MAX_CHUNKS, nm // 16)
    if xp is None or xp is np:
        return many, f"CPU: min({_MAX_CHUNKS}, nm // 16) with nm = {nm}"
    try:
        cc = int(xp.cuda.Device().compute_capability)
    except Exception:                                  # noqa: BLE001 -- no device query: default
        return many, f"GPU of unknown compute capability: min({_MAX_CHUNKS}, nm // 16)"
    if cc not in (60, 70, 80, 90, 100):              # reduced FP64 throughput
        return many, (f"GPU sm_{cc} with reduced FP64 throughput: min({_MAX_CHUNKS}, nm // 16) "
                      "(L40S: 4 chunks fastest at 256x127x64, 9 % faster than 1)")
    Mt = space.sizes(True)[1]
    Mr = space.sizes(True)[2] if not space.s2 else 1
    flops = 2.0 * nm * Mt * (space.L + 1) * 2 * Mr
    if flops >= 1e9:
        return many, f"full-FP64 GPU sm_{cc}, {flops / 1e9:.2f} GFLOP per angular GEMM >= 1 (measured on A100)"
    return 1, f"full-FP64 GPU sm_{cc}, {flops / 1e9:.2f} GFLOP per angular GEMM < 1: one GEMM (measured on A100)"


def ang_chunks(ms: np.ndarray, s: int, L: int, nchunk: int) -> list:
    """Chunks (m0, m1, l0) of the stored wavenumbers for the angular GEMMs restricted to the
    degrees that exist: Phi_l^{m,s} = 0 exactly for l < max(|m|, |s|) (sphere_ops.values: zero
    columns), so chunk [m0, m1) of increasing |m| only needs l >= l0 = max(|m_m0|, |s|).
    Same products, the exactly-zero terms dropped (about half the flops for many chunks)."""
    nm = len(ms)
    if nchunk <= 1 or nm < 2 or np.any(np.diff(np.abs(ms)) < 0):
        return [(0, nm, 0)]
    edges = sorted(set(np.linspace(0, nm, min(nchunk, nm) + 1).astype(int).tolist()))
    out = []
    for m0, m1 in zip(edges[:-1], edges[1:]):
        out.append((m0, m1, min(max(int(abs(ms[m0])), abs(s)), L + 1)))
    return out


def _ang_real(which: str, A, y, chunks: list, xp):
    """Angular transform with the real (nm, P, Q) stack A on complex y (nm, Q, R), as
    ``_left_real`` but chunk by chunk over m on the existing degrees (``ang_chunks``):
    'bwd' (synthesis, Q = L+1): out[m] = A[m][:, l0:] @ y[m][l0:]; 'fwd' (projection, P = L+1):
    out[m][l0:] = A[m][l0:] @ y[m], rows l < l0 exactly zero (as the full product gives).
    GPU: cuBLAS strided-batched GEMMs on the views (no copy); CPU (or no direct cuBLAS):
    matmul per chunk."""
    from ..discretize._cublas import available, gemm_rm
    y = xp.ascontiguousarray(y)
    nm, Q, R = y.shape
    P = A.shape[1]
    yf = y.view(np.float64)                                   # (nm, Q, 2R)
    W = 2 * R
    if len(chunks) == 1:
        return _gemm(A, yf, xp).view(np.complex128)
    out = (xp.empty if which == "bwd" else xp.zeros)((nm, P, W), dtype=np.float64)
    if xp is np or not available():
        for m0, m1, l0 in chunks:
            if which == "bwd":
                out[m0:m1] = xp.matmul(A[m0:m1, :, l0:], yf[m0:m1, l0:, :])
            else:
                out[m0:m1, l0:, :] = xp.matmul(A[m0:m1, l0:, :], yf[m0:m1])
        return out.view(np.complex128)
    A = xp.ascontiguousarray(A, dtype=np.float64)
    a0, y0, c0 = A.data.ptr, yf.data.ptr, out.data.ptr
    for m0, m1, l0 in chunks:
        k = (Q if which == "fwd" else Q - l0)
        rows = P - l0 if which == "fwd" else P
        if k <= 0 or rows <= 0:
            if which == "bwd":
                out[m0:m1] = 0
            continue
        # row-major C' (rows x W) = A' (rows x k) @ Y' (k x W) on views of the full arrays
        if which == "bwd":
            pa, pc = a0 + 8 * (m0 * P * Q + l0), c0 + 8 * (m0 * P * W)
            py = y0 + 8 * (m0 * Q * W + l0 * W)
        else:
            pa, pc = a0 + 8 * (m0 * P * Q + l0 * Q), c0 + 8 * (m0 * P * W + l0 * W)
            py = y0 + 8 * (m0 * Q * W)
        gemm_rm(xp, rows, W, k, pa, Q, P * Q, py, W, Q * W, pc, W, P * W, m1 - m0)
    return out.view(np.complex128)


def _dot_fast(a, ra: int, b, rb: int, G):
    """sphere_eval._dot (last index of a with the first of b, spin metric G) with the same
    products and the same order of the sums, without the multiplications by G = 1 and with the
    sums accumulated in place (same values; signed zeros aside)."""
    xp = array_module(a, b)
    A = xp.moveaxis(a, ra - 1, 0)
    ar, br = ra - 1, rb - 1
    out = None
    for s in range(G.shape[0]):
        for t in range(G.shape[1]):
            if not G[s, t]:
                continue
            x, y = A[s], b[t]
            x = x.reshape(x.shape[:ar] + (1,) * br + x.shape[ar:])
            y = y.reshape((1,) * ar + y.shape)
            term = x * y if G[s, t] == 1 else G[s, t] * (x * y)
            if out is None:
                out = term
            else:
                out += term
    return out


_SPEC_SRC = {
    # spin pair: spectrum of comp c (phi last) from its angular synthesis V and the partner's W,
    # both (nm, Mt, Mr) contiguous: m = 0 symmetrized, -m by conjugation, the rest zero
    "pair_in": ("raw complex128 V, raw complex128 W, int64 nm, int64 Mp, int64 TR",
                "complex128 S",
                "const long long k = i % Mp, q = i / Mp;"
                "if (k == 0) S = 0.5 * (V[q] + conj(W[q]));"
                "else if (k < nm) S = V[k * TR + q];"
                "else if (k > Mp - nm) S = conj(W[(Mp - k) * TR + q]);"
                "else S = complex<double>(0.0, 0.0);"),
    # radial / scalar component: half spectrum for the real inverse FFT
    "real_in": ("raw complex128 V, int64 nm, int64 Mh, int64 TR", "complex128 H",
                "const long long k = i % Mh, q = i / Mh;"
                "if (k == 0) H = complex<double>(V[q].real(), 0.0);"
                "else if (k < nm) H = V[k * TR + q];"
                "else H = complex<double>(0.0, 0.0);"),
    # forward: stored modes of comp c and of its partner (conjugates of the modes -m), (nm, Mt, Mr)
    "pair_out": ("raw complex128 F, int64 Mp, int64 TR", "complex128 A, complex128 B",
                 "const long long m = i / TR, q = i % TR;"
                 "A = F[q * Mp + m];"
                 "B = conj(F[q * Mp + (m == 0 ? 0 : Mp - m)]);"),
    "real_out": ("raw complex128 F, int64 Mh, int64 TR", "complex128 A",
                 "const long long m = i / TR, q = i % TR;"
                 "A = F[q * Mh + m];"),
}
_SPEC_K: dict = {}


def _spec_kernel(xp, name: str):
    """Generated elementwise kernel of the spectrum assembly / extraction (GPU path of
    ``mixed_to_parts`` / ``parts_to_mixed``; the array code is its CPU oracle)."""
    k = _SPEC_K.get(name)
    if k is None:
        ins, outs, body = _SPEC_SRC[name]
        k = _SPEC_K[name] = xp.ElementwiseKernel(ins, outs, body, "cn_sphere_spec_" + name)
    return k


class _FastSpace:
    """SphereSpace with faster transforms for real fields in the full space (no fold, spin
    components on the grid). Everything else is delegated to the space (the reference)."""

    def __init__(self, S, ang_chunks: int = 1) -> None:
        self._S = S
        self._cache: dict = {}
        self.nchunk = int(ang_chunks)                # m-chunks of the angular GEMMs (ang_chunks)

    def _ang(self, which: str, ms, s: int, Mt: int, y, xp):
        """Angular synthesis ('bwd') / projection ('fwd') of spin s on complex y, restricted to
        the existing degrees l >= max(|m|, |s|) per m-chunk (exactly-zero terms dropped)."""
        key = ("chunks", tuple(ms), s)
        ch = self._cache.get(key)
        if ch is None:
            ch = self._cache[key] = ang_chunks(np.asarray(ms), s, self._S.L, self.nchunk)
        return _ang_real(which, self._S._ang_stack(which, ms, s, Mt, xp), y, ch, xp)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._S, name)

    def _fast(self, dealias, physical, fold) -> bool:
        return self._S.real and dealias is True and not physical and fold is None

    # -------------------------------------------------------------- radial (shell)
    def _rad(self, whats: tuple, Mr: int, xp):
        key = ("rad", whats, Mr, xp is np)
        if key not in self._cache:
            M = np.concatenate([self._S._shell_mat("bwd", w, Mr) for w in whats], axis=1)
            self._cache[key] = xp.asarray(np.ascontiguousarray(M))
        return self._cache[key]

    def coeff_to_mixed_many(self, c, dom, rank: int, whats: tuple) -> list:
        """coeff_to_mixed for several ``what`` at once (one real GEMM, shell, dealiased grid)."""
        S = self._S
        xp = array_module(c)
        if not (S.shell and S.has(dom)[2]):
            return [S.coeff_to_mixed(c, dom, rank, True, w) for w in whats]
        Mr = S.sizes(True)[2]
        z = _right_real(xp.asarray(c, dtype=complex), self._rad(whats, Mr, xp), xp)
        return [z[..., i * Mr:(i + 1) * Mr] for i in range(len(whats))]

    def coeff_to_mixed(self, c, dom, rank: int, dealias, what: str = "val", fold=None):
        S = self._S
        if fold is not None or dealias is not True or not (S.shell and S.has(dom)[2]):
            return S.coeff_to_mixed(c, dom, rank, dealias, what, fold=fold)
        return self.coeff_to_mixed_many(c, dom, rank, (what,))[0]

    def mixed_to_coeff(self, x, dom, rank: int, dealias, fold=None):
        S = self._S
        if fold is not None or dealias is not True or not (S.shell and S.has(dom)[2]):
            return S.mixed_to_coeff(x, dom, rank, dealias, fold=fold)
        xp = array_module(x)
        Mr = x.shape[-1]
        key = ("fwd", Mr, xp is np)
        if key not in self._cache:
            self._cache[key] = xp.asarray(np.ascontiguousarray(S._shell_mat("fwd", "val", Mr)))
        return _right_real(xp.asarray(x, dtype=complex), self._cache[key], xp)

    # -------------------------------------------------------------- angular + azimuthal
    def mixed_to_grid(self, x, dom, rank: int, dealias, physical: bool = True, fold=None):
        S = self._S
        if not self._fast(dealias, physical, fold):
            return S.mixed_to_grid(x, dom, rank, dealias, physical, fold)
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        xp = array_module(x)
        x = xp.asarray(x)
        ms = S._m_of(dom)
        spec = {}
        for c in comps(rank, S.nc):
            y = x[c] if c else x
            y = y if not S.s2 else y[..., None]                     # (nm, L+1|1, Mr|1)
            if ht:
                spec[c] = self._ang("bwd", ms, spin_of(c), Mt, y, xp)
            else:                          # no theta: the single entry is the l = 0 coefficient
                spec[c] = y * _R2 if spin_of(c) == 0 else 0 * y
        out = xp.empty(S.grid_shape(dom, rank, True), dtype=complex)
        done: set = set()
        for c in comps(rank, S.nc):
            if c in done:
                continue
            V, f = spec[c], _flip(c)
            if not hp:                     # no phi: m = 0 only, the values themselves
                g = V
            elif f == c:                   # radial / scalar component: real on the grid
                H = xp.zeros((Mp // 2 + 1,) + V.shape[1:], dtype=complex)
                H[S.ms] = V
                H[0] = H[0].real
                g = xp.fft.irfft(H, n=Mp, axis=0, norm="forward")
            else:                          # spin pair (s, -s): one FFT, the partner by conjugation
                Sx = xp.zeros((Mp,) + V.shape[1:], dtype=complex)
                W = spec[f]
                Sx[S.ms[1:]] = V[1:]
                Sx[0] = 0.5 * (V[0] + xp.conj(W[0]))         # m = 0 of a real field: symmetrized
                Sx[(-S.ms[1:]) % Mp] = xp.conj(W[1:])
                g = xp.fft.ifft(Sx, axis=0, norm="forward")
                xp.conjugate(g[..., 0] if S.s2 else g, out=out[f])
                done.add(f)
            g = g[..., 0] if S.s2 else g
            if c:
                out[c] = g
            else:
                out[...] = g
        return out

    def grid_to_mixed(self, g, dom, rank: int, dealias, physical: bool = True, fold=None):
        S = self._S
        if not self._fast(dealias, physical, fold):
            return S.grid_to_mixed(g, dom, rank, dealias, physical, fold)
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        xp = array_module(g)
        g = xp.asarray(g)
        ms = S._m_of(dom)
        Mr = (S.sizes(True)[2] if hr else 1) if not S.s2 else 1
        out = xp.empty((S.nc,) * rank + (len(ms), S.L + 1 if ht else 1, Mr), dtype=complex)
        az: dict = {}
        for c in comps(rank, S.nc):
            if c in az:
                continue
            y = g[c] if c else g
            y = y if not S.s2 else y[..., None]
            f = _flip(c)
            if not hp:
                az[c] = xp.asarray(y, dtype=complex)
            elif f == c:                   # real component: half-size real FFT
                az[c] = xp.fft.rfft(y.real, axis=0, norm="forward")[S.ms]
            else:                          # spin pair: one FFT, the partner's modes by conjugation
                Fy = xp.fft.fft(y, axis=0, norm="forward")
                az[c] = Fy[S.ms]
                az[f] = xp.conj(Fy[(-S.ms) % Mp])
        for c in comps(rank, S.nc):
            s = spin_of(c)
            if ht:
                y = self._ang("fwd", ms, s, Mt, az[c], xp)
            else:
                y = az[c] / _R2 if s == 0 else 0 * az[c]
            if c:
                out[c] = y
            else:
                out[...] = y
        return out[..., 0] if S.s2 else out

    def mixed_to_parts(self, x, dom, rank: int) -> dict:
        """Mixed space -> canonical grid components of a real tensor (``_PG`` parts, layout
        ``_plast``: phi last, contiguous): one inverse FFT per spin pair, real inverse FFTs for
        radial / scalar components, no conjugate copies."""
        S = self._S
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        xp = array_module(x)
        x = xp.asarray(x)
        ms = S._m_of(dom)
        nm = len(S.ms)
        out = {}
        for c in comps(rank, S.nc):
            if _canon(c) != c:
                continue
            f = _flip(c)
            spec = []
            for cc in ((c,) if f == c else (c, f)):
                y = x[cc] if cc else x
                y = y if not S.s2 else y[..., None]                  # (nm, L+1|1, Mr|1)
                if ht:
                    y = self._ang("bwd", ms, spin_of(cc), Mt, y, xp)
                else:
                    y = y * _R2 if spin_of(cc) == 0 else 0 * y
                spec.append(xp.moveaxis(y, 0, -1))                  # (Mt|1, Mr|1, nm)
            V = spec[0]
            if hp and xp is not np:          # GPU: assembly in one kernel from (nm, Mt|1, Mr|1)
                Vc = xp.ascontiguousarray(xp.moveaxis(V, -1, 0))
                TR = int(np.prod(Vc.shape[1:]))
                if f == c:
                    H = xp.empty(V.shape[:-1] + (Mp // 2 + 1,), dtype=complex)
                    _spec_kernel(xp, "real_in")(Vc, nm, Mp // 2 + 1, TR, H)
                    g = xp.fft.irfft(H, n=Mp, axis=-1, norm="forward")
                else:
                    Wc = xp.ascontiguousarray(xp.moveaxis(spec[1], -1, 0))
                    Sx = xp.empty(V.shape[:-1] + (Mp,), dtype=complex)
                    _spec_kernel(xp, "pair_in")(Vc, Wc, nm, Mp, TR, Sx)
                    g = xp.fft.ifft(Sx, axis=-1, norm="forward")
            elif not hp:                     # no phi: m = 0 only, the values themselves
                g = xp.ascontiguousarray(V.real) if f == c else xp.ascontiguousarray(V)
            elif f == c:                     # radial / scalar component: real on the grid
                H = xp.empty(V.shape[:-1] + (Mp // 2 + 1,), dtype=complex)
                H[..., :nm] = V                              # stored m = 0 .. nm - 1 (RealFourier)
                H[..., nm:] = 0
                H[..., 0] = H[..., 0].real
                g = xp.fft.irfft(H, n=Mp, axis=-1, norm="forward")
            else:                            # spin pair: one FFT; m = 0 symmetrized (real field)
                W = spec[1]
                Sx = xp.empty(V.shape[:-1] + (Mp,), dtype=complex)
                Sx[..., 1:nm] = V[..., 1:]
                Sx[..., 0] = 0.5 * (V[..., 0] + xp.conj(W[..., 0]))
                Sx[..., nm:Mp - nm + 1] = 0
                Sx[..., Mp - 1:Mp - nm:-1] = xp.conj(W[..., 1:])       # modes -1 .. -(nm - 1)
                g = xp.fft.ifft(Sx, axis=-1, norm="forward")
            out[c] = g[:, 0] if S.s2 else g
        return out

    def parts_to_mixed(self, parts: dict, dom, rank: int):
        """Canonical grid components (layout ``_plast``) -> mixed space: one forward FFT per
        spin pair (the partner's modes by conjugation), real FFTs for radial / scalar ones."""
        S = self._S
        hp, ht, hr = S.has(dom)
        Mp, Mt = S.sizes(True)[:2]
        ms = S._m_of(dom)
        Mr = (S.sizes(True)[2] if hr else 1) if not S.s2 else 1
        first = next(iter(parts.values()))
        xp = array_module(first)
        out = xp.empty((S.nc,) * rank + (len(ms), S.L + 1 if ht else 1, Mr), dtype=complex)
        az: dict = {}
        for c, y in parts.items():
            y = y[:, None, :] if S.s2 else y                         # (Mt|1, Mr|1, Mp|1)
            f = _flip(c)
            if not hp:
                az[c] = xp.moveaxis(xp.asarray(y, dtype=complex), -1, 0)
                if f != c:
                    az[f] = xp.conj(az[c])
            elif xp is not np:               # GPU: extraction in one kernel -> (nm, Mt|1, Mr|1)
                TR = int(np.prod(y.shape[:-1]))
                shp = (len(S.ms),) + y.shape[:-1]
                if f == c:
                    Fh = xp.fft.rfft(y, axis=-1, norm="forward")
                    az[c] = xp.empty(shp, dtype=complex)
                    _spec_kernel(xp, "real_out")(Fh, Mp // 2 + 1, TR, az[c])
                else:
                    Fy = xp.fft.fft(y, axis=-1, norm="forward")
                    az[c], az[f] = xp.empty(shp, dtype=complex), xp.empty(shp, dtype=complex)
                    _spec_kernel(xp, "pair_out")(Fy, Mp, TR, az[c], az[f])
            elif f == c:
                az[c] = xp.moveaxis(xp.fft.rfft(y, axis=-1, norm="forward")[..., S.ms], -1, 0)
            else:
                Fy = xp.fft.fft(y, axis=-1, norm="forward")
                az[c] = xp.moveaxis(Fy[..., S.ms], -1, 0)
                az[f] = xp.moveaxis(xp.conj(Fy[..., (-S.ms) % Mp]), -1, 0)
        for c in comps(rank, S.nc):
            s = spin_of(c)
            if ht:
                y = self._ang("fwd", ms, s, Mt, az[c], xp)
            else:
                y = az[c] / _R2 if s == 0 else 0 * az[c]
            if c:
                out[c] = y
            else:
                out[...] = y
        return out[..., 0] if S.s2 else out

    def backward(self, c, dom, rank: int, dealias, physical: bool = True, fold=None):
        return self.mixed_to_grid(self.coeff_to_mixed(c, dom, rank, dealias, fold=fold), dom, rank,
                                  dealias, physical, fold)

    def forward(self, g, dom, rank: int, dealias, physical: bool = True, fold=None):
        return self.mixed_to_coeff(self.grid_to_mixed(g, dom, rank, dealias, physical, fold), dom,
                                   rank, dealias, fold=fold)


# ----------------------------------------------------------------------------------------------
# products on the grid: canonical spin components, one generated kernel per product tree
# ----------------------------------------------------------------------------------------------
_PRODUCT = (E.Add, E.Mul, E.Dot, E.Cross, E.Outer, E.Component, E.Conj, E.Power, E.UnaryFunction)
_FUNCS = ("exp", "log", "sin", "cos", "tanh", "sqrt", "abs", "sign")


def _plast(shape: tuple) -> tuple:
    """Layout of the canonical grid components: the grid shape (Mphi, Mtheta[, Mr]) with phi
    moved last, (Mtheta, Mr, Mphi) / (Mtheta, Mphi): the FFTs run on the contiguous last axis
    and the product kernels read contiguous arrays."""
    return tuple(shape[1:]) + (shape[0],)


def _canon(c: tuple) -> tuple:
    """Canonical member of the spin pair {c, flip(c)} of a real tensor (its own if radial)."""
    f = _flip(c)
    return c if c <= f else f


def _real_comp(c: tuple) -> bool:
    """Component real on the grid for a real tensor: every index radial (or a scalar)."""
    return all(i == 2 for i in c)


class _PG(_G):
    """Grid value of a real tensor held by its canonical components only (``parts``: comp ->
    array; complex for a spin pair, real float64 for a radial / scalar component). The other
    member of a pair is the complex conjugate (real fields). ``g`` builds the full array of spin
    components on demand (consumers outside the planned path)."""
    __slots__ = ("parts", "nc", "_gc")

    def __init__(self, parts: dict, dom, rank: int, nc: int):
        self.parts, self.nc, self._gc = parts, nc, None
        self.dom, self.rank, self.f = dom, rank, None

    @property
    def g(self):
        if self._gc is None:
            first = next(iter(self.parts.values()))
            xp = array_module(first)
            shape = np.broadcast_shapes(*[a.shape for a in self.parts.values()])
            shape = shape[-1:] + shape[:-1]                   # phi first again
            out = xp.empty((self.nc,) * self.rank + shape, dtype=complex)
            for c in comps(self.rank, self.nc):
                k = _canon(c)
                v = xp.moveaxis(self.parts[k], -1, 0)
                v = v if k == c else xp.conj(v)
                if c:
                    out[c] = v
                else:
                    out[...] = v
            self._gc = out
        return self._gc


class _ProductPlan:
    """A product tree (sums, products by scalars, dot, cross, outer, components, scalar
    functions) compiled to straight-line code per canonical output component: SSA over loads
    of the canonical components of its leaves (values of non-product nodes: fields, derivatives,
    constants) and host scalars. Executed by array operations (NumPy / CuPy: the CPU path and
    the oracle) or, on the GPU, by one generated elementwise kernel (``cupy.ElementwiseKernel``)
    that reads each leaf once and writes the result once."""

    def __init__(self, x: E.Expr, nc: int, G: np.ndarray, rk: bool) -> None:
        self.nc, self.G, self.rk = nc, G, rk
        self.leaves: list = []                         # leaf nodes
        self.scalars: list = []                        # field-free nodes (host numbers per call)
        self.nodes: list = []                          # SSA: (op, args, real)
        self._index: dict = {}
        val = self._tensor(x)
        self.rank = x.rank
        self.out = [(c, val.get(c)) for c in comps(x.rank, nc) if _canon(c) == c]
        self._kernel = self._kernel_ptr = None

    # ---------------------------------------------------------------- SSA construction
    def _node(self, op: str, args: tuple, real: bool) -> int:
        key = (op, args)
        i = self._index.get(key)
        if i is None:
            i = self._index[key] = len(self.nodes)
            self.nodes.append((op, args, real))
        return i

    def _num(self, v) -> int:
        v = complex(v)
        return self._node("num", (v.real, v.imag), v.imag == 0)

    def _real(self, i) -> bool:
        return self.nodes[i][2]

    def _mul(self, a, b):
        if a is None or b is None:
            return None
        na, nb = self.nodes[a], self.nodes[b]
        if na[0] == "num" and nb[0] == "num":
            return self._num(complex(*na[1]) * complex(*nb[1]))
        if na[0] == "num" and na[1] == (1.0, 0.0):
            return b
        if nb[0] == "num" and nb[1] == (1.0, 0.0):
            return a
        if nb[0] == "num":                              # numbers first (one canonical order)
            a, b = b, a
        return self._node("mul", (a, b), self._real(a) and self._real(b))

    def _add(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        return self._node("add", (a, b), self._real(a) and self._real(b))

    def _tensor(self, x: E.Expr) -> dict:
        """comp -> SSA index (None: zero) of the value of x."""
        if not x.free_fields():
            if isinstance(x, E.Constant):
                return {(): self._num(x.value)}
            if x not in self.scalars:
                self.scalars.append(x)
            return {(): self._node("sc", (self.scalars.index(x),), True)}
        if isinstance(x, E.Add):
            out: dict = {}
            for a in x.args:
                for c, v in self._tensor(a).items():
                    out[c] = self._add(out.get(c), v)
            return out
        if isinstance(x, E.Mul):
            vals = [self._tensor(a) for a in x.args]
            tens = [v for v, a in zip(vals, x.args) if a.rank]
            if len(tens) > 1:
                raise DiscretizationError(f"'{x}': product of two tensors, use dot() or cross()")
            k = self._num(1.0)
            for v, a in zip(vals, x.args):
                if not a.rank:
                    k = self._mul(k, v.get(()))              # None (zero) stays None
            base = tens[0] if tens else {(): self._num(1.0)}
            return {c: self._mul(k, v) for c, v in base.items()}
        if isinstance(x, E.Dot):
            A, B = self._tensor(x.args[0]), self._tensor(x.args[1])
            ra, rb = x.args[0].rank, x.args[1].rank
            if ra == 0 or rb == 0:
                raise DiscretizationError("dot of a scalar")
            out = {}
            for ca in comps(ra - 1, self.nc):
                for cb in comps(rb - 1, self.nc):
                    acc = None
                    for s_ in range(self.nc):
                        for t_ in range(self.nc):
                            if self.G[s_, t_]:
                                term = self._mul(A.get(ca + (s_,)), B.get((t_,) + cb))
                                if term is not None and self.G[s_, t_] != 1:
                                    term = self._mul(self._num(self.G[s_, t_]), term)
                                acc = self._add(acc, term)
                    out[ca + cb] = acc
            return out
        if isinstance(x, E.Cross):
            from ..discretize.sphere_ops import cross_table
            A, B = self._tensor(x.args[0]), self._tensor(x.args[1])
            out = {(k,): None for k in range(self.nc)}
            for (i, j), (k, cst) in cross_table().items():
                out[(k,)] = self._add(out[(k,)], self._mul(self._num(cst),
                                                           self._mul(A.get((i,)), B.get((j,)))))
            return out
        if isinstance(x, E.Outer):
            A, B = self._tensor(x.args[0]), self._tensor(x.args[1])
            return {ca + cb: self._mul(A.get(ca), B.get(cb))
                    for ca in comps(x.args[0].rank, self.nc) for cb in comps(x.args[1].rank, self.nc)}
        if isinstance(x, E.Component):
            if x.index != 2 or self.nc != 3:
                raise DiscretizationError(f"'{x}': only the radial component (index 2) in 3D")
            A = self._tensor(x.args[0])
            return {c: A.get((2,) + c) for c in comps(x.rank, self.nc)}
        if isinstance(x, E.Conj):                       # real fields: conj(u) = u (spins flip back)
            return self._tensor(x.args[0])
        if isinstance(x, (E.Power, E.UnaryFunction)):
            a = self._tensor(x.args[0]).get(())
            if a is None:
                a = self._num(0.0)
            if not self._real(a):                       # real scalar of complex-typed spin products
                a = self._node("re", (a,), True)
            if isinstance(x, E.Power):
                return {(): self._node("pow", (a, float(x.exponent)), True)}
            if x.func not in _FUNCS:
                raise KeyError(x.func)
            return {(): self._node("fn", (x.func, a), True)}
        # leaf: a non-product node, read through its canonical components
        if x not in self.leaves:
            self.leaves.append(x)
        li = self.leaves.index(x)
        out = {}
        for c in comps(x.rank, self.nc):
            k = _canon(c)
            ld = self._node("ld", (li, k), _real_comp(k))
            out[c] = ld if k == c else self._node("conj", (ld,), _real_comp(k))
        return out

    # ---------------------------------------------------------------- execution
    def _live(self) -> list:
        live = set()
        stack = [i for _, i in self.out if i is not None]
        while stack:
            i = stack.pop()
            if i in live:
                continue
            live.add(i)
            op, args, _ = self.nodes[i]
            if op in ("mul", "add"):
                stack.extend(args)
            elif op in ("conj", "pow", "re"):
                stack.append(args[0])
            elif op == "fn":
                stack.append(args[1])
        return sorted(live)

    def run(self, leaf_parts: list, scalars: list, shape: tuple, xp, rk=None,
            mode: str = "kernel", sptr=None) -> dict:
        """Canonical components of the result (dict comp -> array of ``shape``): ``mode``
        'kernel' (generated CUDA kernel, CuPy only) or 'arrays' (array operations, the oracle).
        ``sptr`` = (device array, offset): the scalars are read on the device (graph replay)."""
        if mode == "kernel" and xp is not np:
            return self._run_kernel(leaf_parts, scalars, shape, xp, rk, sptr)
        return self._run_arrays(leaf_parts, scalars, shape, xp, rk)

    def _run_arrays(self, leaf_parts, scalars, shape, xp, rk):
        v: dict = {}
        for i in self._live():
            op, args, real = self.nodes[i]
            if op == "ld":
                v[i] = leaf_parts[args[0]][args[1]]
            elif op == "sc":
                v[i] = scalars[args[0]]
            elif op == "num":
                v[i] = args[0] if args[1] == 0 else complex(*args)
            elif op == "mul":
                v[i] = v[args[0]] * v[args[1]]
            elif op == "add":
                v[i] = v[args[0]] + v[args[1]]
            elif op == "conj":
                v[i] = v[args[0]] if self._real(args[0]) else xp.conj(v[args[0]])
            elif op == "pow":
                v[i] = v[args[0]] ** args[1]
            elif op == "re":
                v[i] = v[args[0]].real
            else:
                f = getattr(xp, args[0])
                v[i] = f(v[args[1]])
        out = {}
        for c, i in self.out:
            real = _real_comp(c)
            a = xp.zeros(shape, dtype=float if real else complex) if i is None else xp.asarray(v[i])
            if real and xp.iscomplexobj(a):
                a = a.real
            if rk is not None:
                a = a * rk
            out[c] = xp.ascontiguousarray(xp.broadcast_to(a, shape)).astype(float if real else complex, copy=False)
        return out

    def _kernel_src(self, leaf_parts, ptr: bool = False) -> tuple:
        """(parameters, outputs, body) of the generated elementwise kernel. ``ptr``: the host
        scalars are read from a device array (``sv[so + k]``) instead of kernel arguments (CUDA
        graph replay with time-dependent factors, :class:`GraphScalars`); same values."""
        live = self._live()
        loads = sorted({self.nodes[i][1] for i in live if self.nodes[i][0] == "ld"})
        lname = {ld: f"L{ld[0]}_{''.join(map(str, ld[1])) or 's'}" for ld in loads}
        ins = [f"{'float64' if _real_comp(ld[1]) else 'complex128'} {lname[ld]}" for ld in loads]
        if ptr:
            ins += ["raw float64 sv", "int64 so"] if self.scalars else []
        else:
            ins += [f"float64 s{k}" for k in range(len(self.scalars))]
        if self.rk:
            ins.append("float64 rk")
        outs = [f"{'float64' if _real_comp(c) else 'complex128'} o{j}" for j, (c, _) in enumerate(self.out)]
        body = []
        T = {True: "double", False: "complex<double>"}
        for i in live:
            op, args, real = self.nodes[i]
            if op == "ld":
                e = lname[args]
            elif op == "sc":
                e = f"sv[so + {args[0]}]" if ptr else f"s{args[0]}"
            elif op == "num":
                e = repr(float(args[0])) if args[1] == 0 else f"complex<double>({args[0]!r}, {args[1]!r})"
            elif op == "mul":
                e = f"v{args[0]} * v{args[1]}"
            elif op == "add":
                e = f"v{args[0]} + v{args[1]}"
            elif op == "conj":
                e = f"v{args[0]}" if self._real(args[0]) else f"conj(v{args[0]})"
            elif op == "pow":
                e = f"pow(v{args[0]}, {args[1]!r})"
            elif op == "re":
                e = f"v{args[0]}.real()"
            else:
                fn, a = args
                e = {"abs": f"fabs(v{a})", "sign": f"(double)((v{a} > 0) - (v{a} < 0))"}.get(fn, f"{fn}(v{a})")
            body.append(f"const {T[real]} v{i} = {e};")
        for j, (c, i) in enumerate(self.out):
            real = _real_comp(c)
            if i is None:
                e = "0.0" if real else "complex<double>(0.0, 0.0)"
            else:
                e = f"v{i}" if (self._real(i) or not real) else f"v{i}.real()"
            if self.rk:
                e = f"({e}) * rk"
            body.append(f"o{j} = {e};")
        return loads, ", ".join(ins), ", ".join(outs), "\n".join(body)

    def _run_kernel(self, leaf_parts, scalars, shape, xp, rk, sptr=None):
        ptr = sptr is not None
        kern = self._kernel_ptr if ptr else self._kernel
        if kern is None:
            loads, ins, outs, body = self._kernel_src(leaf_parts, ptr)
            key = (ins, outs, body)
            kern = _KERNELS.get(key)
            if kern is None:
                import hashlib
                name = "cn_sphere_prod_" + hashlib.sha1(repr(key).encode()).hexdigest()[:12]
                kern = _KERNELS[key] = xp.ElementwiseKernel(ins, outs, body, name)
            if ptr:
                self._kernel_ptr = kern
            else:
                self._kernel = kern
            self._loads = loads
        args = [leaf_parts[li][c] for li, c in self._loads]
        if ptr:
            args += [sptr[0], int(sptr[1])] if self.scalars else []
        else:
            args += [float(v) for v in scalars]
        if self.rk:
            args.append(rk)
        outs = [xp.empty(shape, dtype=float if _real_comp(c) else complex) for c, _ in self.out]
        kern(*args, *outs)
        return {c: o for (c, _), o in zip(self.out, outs)}


_KERNELS: dict = {}


class _M(_G):
    """Memoized value of a node in coefficients or mixed space (``orig``: _C / _X) whose grid
    values are computed once, on first use (``g``): consumers that need the grid share one
    backward transform, consumers that need coefficients use ``orig`` (no grid round trip)."""
    __slots__ = ("orig", "_gc", "_ev", "_pc")

    def __init__(self, orig, ev):
        self.orig, self._gc, self._ev, self._pc = orig, None, ev, None
        self.dom, self.rank, self.f = orig.dom, orig.rank, None

    @property
    def g(self):
        if self._gc is None:
            self._gc = SphereEvaluator._grid(self._ev, self.orig).g
        return self._gc


class _FusedEvaluator(SphereEvaluator):
    """SphereEvaluator with a per-call memo (common sub-expressions), cached constants, host
    scalars and the fast transforms of ``_FastSpace``."""

    def __init__(self, space, xp, state: frozenset) -> None:
        super().__init__(space, xp=xp)
        nch, self.chunk_rule = ang_chunks_rule(space, xp)
        self.S = _FastSpace(space, ang_chunks=nch)
        self._host = SphereEvaluator(space)                # field-free scalars on the host
        self.state = state
        self._memo: dict = {}
        self._const: dict = {}
        self._kind: dict = {}                              # node -> (has fields, constant)
        self._plans: dict = {}                             # (product node, r^k) -> _ProductPlan
        # products on the grid: one generated kernel per product tree on the GPU ('kernel');
        # on the CPU the reference array operations of the evaluator ('evaluator': faster there
        # than the per-component straight-line code); 'arrays' runs the plans by array
        # operations (tests: the oracle of the generated kernels)
        self.products = "evaluator" if xp is np or xp is None else "kernel"
        self.hits = 0
        self.graph: GraphScalars | None = None             # CUDA graph replay (time factors)
        self._in_scalars = False

    def host_scalar(self, x: E.Expr, params) -> float:
        """Value of a field-free scalar node on the host (the planned F's time factors a_g(t))."""
        v = self._host._ev_raw(x, {}, params)
        v = np.asarray(v.g if isinstance(v, _G) else v)
        if v.ndim:
            raise DiscretizationError(f"'{x}': a field-free right-hand-side term must be a scalar")
        v = v.item()
        if isinstance(v, complex):
            if v.imag:
                raise DiscretizationError(
                    f"'{x}' = {v}: complex value in a right-hand side of real fields "
                    "(f_backend='fused')")
            v = v.real
        return v

    def begin(self) -> None:
        self._memo = {}

    def _ev(self, x: E.Expr, coeffs, params):
        v = self._const.get(x)
        if v is not None:
            return v
        if self.graph is not None and not self._in_scalars:
            self.graph.check_node(x)
        v = self._memo.get(x)
        if v is not None:
            self.hits += 1
            return v
        kind = self._kind.get(x)
        if kind is None:
            ff = x.free_fields()
            kind = self._kind[x] = (bool(ff), not (ff & self.state) and not x.free_parameters())
        if not kind[0]:                                    # scalar of parameters: host number
            v = self.host_scalar(x, params)
            self._memo[x] = v
            return v
        v = self._ev_raw(x, coeffs, params)
        if isinstance(v, (_C, _X)):                        # grid computed once, on first use
            v = _M(v, self)
        if kind[1]:
            self._const[x] = v                             # constant: kept across calls
        else:
            self._memo[x] = v
        return v

    def _coeff(self, v) -> _C:
        if isinstance(v, _M):
            return self._coeff(v.orig)
        if isinstance(v, _PG):
            S = self.S
            return _C(S.mixed_to_coeff(S.parts_to_mixed(v.parts, v.dom, v.rank), v.dom, v.rank, True),
                      v.dom, v.rank)
        return super()._coeff(v)

    def _ev_raw(self, x: E.Expr, coeffs, params):
        if isinstance(x, _PRODUCT) and self.products != "evaluator":
            return self._product(x, coeffs, params)
        if isinstance(x, E.Dot):                           # reference _dot without the products
            a = self._grid(self._ev(x.args[0], coeffs, params))   # by G = 1 and with in-place sums
            b = self._grid(self._ev(x.args[1], coeffs, params))
            return _G(_dot_fast(a.g, a.rank, b.g, b.rank, self.S.G), x.domain, x.rank)
        return super()._ev_raw(x, coeffs, params)

    def _parts(self, v) -> dict:
        """Canonical grid components of a leaf value."""
        S = self.S
        if isinstance(v, _PG):
            return v.parts
        if isinstance(v, _M):
            if v._pc is None:
                o = v.orig
                x = o.x if isinstance(o, _X) else S.coeff_to_mixed(o.a, o.dom, o.rank, True)
                v._pc = S.mixed_to_parts(x, o.dom, o.rank)
            return v._pc
        if isinstance(v, _G):
            xp = array_module(v.g)
            out = {}
            for c in comps(v.rank, S.nc):
                if _canon(c) == c:
                    a = xp.moveaxis(v.g[c] if c else v.g, 0, -1)
                    out[c] = xp.ascontiguousarray(a.real if _real_comp(c) else a)
            return out
        raise DiscretizationError(f"product leaf of type {type(v).__name__}")

    def _product(self, x: E.Expr, coeffs, params, rk=None, dom=None) -> _PG:
        """Value of a product tree on the grid (canonical components), one plan per node."""
        S = self.S
        key = (x, rk is not None)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._plans[key] = _ProductPlan(x, S.nc, S.G, rk is not None)
        leaves = [self._parts(self._ev(leaf, coeffs, params)) for leaf in plan.leaves]
        self._in_scalars = True
        try:
            scal = [self._ev(sn, coeffs, params) for sn in plan.scalars]
        finally:
            self._in_scalars = False
        sptr = None if self.graph is None or self.products != "kernel" else self.graph.place(plan.scalars, scal)
        dom = x.domain if dom is None else dom
        shape = _plast(S.grid_shape(dom, 0, True))
        shape = np.broadcast_shapes(shape, *[a.shape for p in leaves for a in p.values()])
        return _PG(plan.run(leaves, scal, tuple(shape), self.xp, rk, self.products, sptr), dom, x.rank, S.nc)

    def evaluate(self, e: E.Expr, coeffs, params, rpow: int = 0):
        """Reference ``evaluate`` (coefficients of r^rpow e, level 0) on the memoized values."""
        S = self.S
        if isinstance(e, _PRODUCT) and e.free_fields() and self.products != "evaluator":
            # products: canonical components, r^k inside the kernel
            dom = self.rpow_domain(e.domain) if rpow else e.domain
            rk = None
            if rpow:
                Mr = S.sizes(True)[2]
                key = ("rpow_plast", Mr, rpow)
                if key not in S._cache:                     # (Mr, 1): r axis of _plast
                    S._cache[key] = self.xp.asarray(np.ascontiguousarray((S.ops.grid(Mr) ** rpow)[:, None]))
                rk = S._cache[key]
            v = self._ev(e, coeffs, params) if rk is None else self._product(e, coeffs, params, rk, dom)
            return S.mixed_to_coeff(S.parts_to_mixed(v.parts, dom, e.rank), dom, e.rank, True)
        v = self._ev(e, coeffs, params)
        if isinstance(v, _M) and not rpow:
            v = v.orig
        if rpow:
            g = self._grid(v)
            xp = array_module(g.g)
            dom = self.rpow_domain(e.domain)
            full = xp.broadcast_to(g.g, self.S.grid_shape(dom, e.rank, True))
            key = ("rpow", full.shape[-1], rpow)
            if key not in self.S._cache:
                self.S._cache[key] = xp.asarray(self.S.ops.grid(full.shape[-1]) ** rpow)
            return self.S.forward(full * self.S._cache[key], dom, e.rank, True, physical=False)
        if isinstance(v, _C):
            return array_module(v.a).broadcast_to(v.a, self.S.coeff_shape(e.domain, e.rank)).copy()
        if isinstance(v, _X):
            return self.S.mixed_to_coeff(self._xfull(v, e.domain, e.rank), e.domain, e.rank, True)
        g = self._grid(v)
        full = array_module(g.g).broadcast_to(g.g, self.S.grid_shape(e.domain, e.rank, True))
        return self.S.forward(full, e.domain, e.rank, True, physical=False)

    def _grad(self, v: _C) -> _X:
        """Reference _grad with the two radial syntheses it uses (1/r, d/dr) in one GEMM."""
        S = self.S
        hp, ht, hr = S.has(v.dom)
        if S.s2 or not hr:
            return super()._grad(v)
        A, D = S.coeff_to_mixed_many(v.a, v.dom, v.rank, ("over_r", "dr"))   # (values unused)
        ms = S._m_of(v.dom)
        nc = S.nc
        xp = array_module(A)
        if xp is not np and self.products == "kernel":
            return _X(self._grad_kernel(A, D, v.rank, ms, xp), v.dom, v.rank + 1, None)
        out = xp.zeros((nc,) + A.shape, dtype=complex)
        for c in comps(v.rank, nc):
            s = spin_of(c)
            a = A[c] if c else A
            for d, sigma in ((0, -1), (1, 1)):
                G = S.Gstack(sigma, ms, s, xp)[:, :a.shape[1]]
                out[(d,) + c] += G[..., None] * a
            out[(2,) + c] += D[c] if c else D
            for i, t in enumerate(c):                          # frame (connection) terms
                rep = lambda j, c=c, i=i: c[:i] + (j,) + c[i + 1:]  # noqa: E731
                if t == 0:
                    out[(1,) + c] += A[rep(2)]
                elif t == 1:
                    out[(0,) + c] += A[rep(2)]
                else:
                    out[(1,) + c] -= A[rep(1)]
                    out[(0,) + c] -= A[rep(0)]
        return _X(out, v.dom, v.rank + 1, None)

    def _grad_kernel(self, A, D, rank: int, ms, xp):
        """GPU: the spin-basis gradient of ``_grad`` (G_-/G_+ in l, d/dr, frame terms) as one
        generated elementwise kernel writing every output component once, the terms of each
        component summed in the order of the array code (``_grad_terms``)."""
        S = self.S
        nc = S.nc
        key = ("gradk", rank, nc)                        # the source depends on both
        k = _KERNELS.get(key)
        cs = comps(rank, nc)
        terms = _grad_terms(rank, nc)
        if k is None:
            ins = [f"complex128 A{_cn(c)}" for c in cs] + [f"complex128 D{_cn(c)}" for c in cs]
            ins += [f"float64 Gm{_cn(c)}, float64 Gp{_cn(c)}" for c in cs]
            outs = [f"complex128 o{_cn(oc)}" for oc in terms]
            body = []
            for oc, ts in terms.items():
                e = None
                for sign, kind, c in ts:
                    t = {"Gm": f"Gm{_cn(c)} * A{_cn(c)}", "Gp": f"Gp{_cn(c)} * A{_cn(c)}",
                         "D": f"D{_cn(c)}", "A": f"A{_cn(c)}"}[kind]
                    e = (t if sign > 0 else f"-({t})") if e is None else f"({e}) {'+' if sign > 0 else '-'} {t}"
                body.append(f"o{_cn(oc)} = {e if e is not None else 'complex<double>(0.0, 0.0)'};")
            k = _KERNELS[key] = xp.ElementwiseKernel(", ".join(ins), ", ".join(outs), "\n".join(body),
                                                     f"cn_sphere_grad_r{rank}_c{nc}")
        out = xp.empty((nc,) + A.shape, dtype=complex)
        args = [A[c] if c else A for c in cs] + [D[c] if c else D for c in cs]
        L1 = A.shape[-2]
        for c in cs:
            s = spin_of(c)
            args += [S.Gstack(-1, ms, s, xp)[:, :L1, None], S.Gstack(1, ms, s, xp)[:, :L1, None]]
        k(*args, *[out[oc] for oc in terms])
        return out


def _cn(c: tuple) -> str:
    return "".join(map(str, c)) or "s"


def _grad_terms(rank: int, nc: int) -> dict:
    """Ordered terms (sign, kind, comp) of each output component of the 3D spin-basis gradient,
    in the order in which ``_FusedEvaluator._grad`` (and the reference) accumulates them."""
    out = {oc: [] for oc in comps(rank + 1, nc)}
    for c in comps(rank, nc):
        out[(0,) + c].append((1, "Gm", c))
        out[(1,) + c].append((1, "Gp", c))
        out[(2,) + c].append((1, "D", c))
        for i, t in enumerate(c):
            def rep(j, c=c, i=i):
                return c[:i] + (j,) + c[i + 1:]
            if t == 0:
                out[(1,) + c].append((1, "A", rep(2)))
            elif t == 1:
                out[(0,) + c].append((1, "A", rep(2)))
            else:
                out[(1,) + c].append((-1, "A", rep(1)))
                out[(0,) + c].append((-1, "A", rep(0)))
    return out


# ----------------------------------------------------------------------------------------------
# CUDA graph replay with time-dependent factors
# ----------------------------------------------------------------------------------------------
class GraphScalars:
    """Time factors a_g(t) of the planned F during a captured Runge--Kutta step (``run_graph`` of a
    time-dependent spherical IVP). While a step is captured, every product kernel reads its host
    scalars from a fixed device array ``dev`` (slots placed in call order) instead of kernel
    arguments; each slot remembers its expression and the stage abscissa c_j of the call. Before
    each replay the host evaluates the same expressions at ``t_n + c_j dt`` (the stage times of
    ``rk_stages``, same formula, same host code as the plain step) and the row is copied into
    ``dev``: the replayed step uses the very numbers of :meth:`IVPSolver.run`. A time-dependent
    scalar used anywhere else in F (outside a product kernel) would be frozen by the capture: refused."""

    CAP = 4096                                             # device slots (doubles) per step

    def __init__(self, solver, ev: _FusedEvaluator) -> None:
        self.solver, self.ev, self.xp = solver, ev, solver.xp
        self.tname = solver.time_parameter
        self.dev = self.xp.zeros(self.CAP, dtype=np.float64)
        self.records: list = []                            # (offset, expression, stage j)
        self.stage: int | None = None
        self.capturing = False
        self.c: list = []
        self.dt = 0.0

    def begin(self, capturing: bool) -> None:
        """Start a (warm-up or captured) step: slots placed from 0, the evaluator reads them."""
        self.records, self.capturing, self.stage = [], capturing, None
        self.ev.graph = self

    def end(self) -> None:
        self.ev.graph, self.stage, self.capturing = None, None, False

    def stage_F(self, t: float, dt: float, c):
        """F for rk_stages(X0, t, dt): each call is tagged with its stage abscissa c_j (the
        first j with t + c_j dt equal to the call's time; equal times give equal factors)."""
        self.c, self.dt = list(c), dt                     # exactly the values rk_stages uses
        times = [t + cj * dt for cj in c]

        def F(X, tt):
            js = [j for j, v in enumerate(times) if v == tt]
            if not js:
                raise UnsupportedError(f"run_graph: F called at t = {tt!r}, not a stage time of the step")
            if len({c[j] for j in js}) > 1:
                # distinct abscissae rounded to the same time (|t| >> dt at the capture): the tag
                # would be ambiguous and later replays would use the wrong stage's factors
                raise UnsupportedError(
                    f"run_graph: at t = {t!r} the stage times t + c_j dt (dt = {dt!r}) of distinct "
                    "abscissae c_j round to the same number; the time factors of the captured step "
                    "cannot be attributed to their stages",
                    fix="run(nsteps) (one Python call per step), or a time origin closer to t "
                        "(time_parameter shifted)")
            self.stage = js[0]
            try:
                return self.solver.F(X, tt)
            finally:
                self.stage = None
        return F

    def check_node(self, x: E.Expr) -> None:
        """Refuse a time-dependent scalar outside the product kernels (frozen by the capture)."""
        if not x.free_fields() and any(p.name == self.tname for p in x.free_parameters()):
            raise UnsupportedError(
                f"run_graph: the time factor '{x}' of the right-hand side is used outside a "
                "product kernel of the planned F; a captured step would freeze its value",
                fix="run(nsteps) (one Python call per step, t set at every stage)")

    def place(self, exprs: list, values: list):
        """Slots of the scalars of one product kernel: (dev, offset). Outside the capture (warm-up)
        the current values are written at once; during the capture only the slots are recorded."""
        if not exprs:
            return (self.dev, 0)
        if self.stage is None:
            raise UnsupportedError("run_graph: planned F called outside a Runge--Kutta stage")
        off = len(self.records)
        if off + len(exprs) > self.CAP:
            raise UnsupportedError(f"run_graph: more than {self.CAP} time factors per step")
        self.records.extend((off + i, e, self.stage) for i, e in enumerate(exprs))
        if not self.capturing:
            self.dev[off:off + len(exprs)] = self.xp.asarray(np.asarray(values, dtype=np.float64))
        return (self.dev, off)

    def table(self, t_steps: list, params: Mapping):
        """Device rows (len(t_steps), n_slots) of the slot values at each step start t_n."""
        n = len(self.records)
        host = np.empty((len(t_steps), max(n, 1)))
        p = dict(params)
        for i, tn in enumerate(t_steps):
            for off, e, j in self.records:
                p[self.tname] = tn + self.c[j] * self.dt    # the stage time of rk_stages
                host[i, off] = self.ev.host_scalar(e, p)
        return self.xp.asarray(host[:, :n])


# ----------------------------------------------------------------------------------------------
# the fused right-hand side of a spherical solver
# ----------------------------------------------------------------------------------------------
class SphereFusedRHS:
    """All right-hand sides of a spherical IVPSolver, planned (module doc). ``evaluate`` returns
    the per-equation level-0 coefficient arrays (r^k projected on the shell, embedded in the
    equation's layout), ready for ``system.gather_rhs``."""

    def __init__(self, solver) -> None:
        S = solver.plan.space
        # symmetry subspace (reduced system): the planned F runs on the full grid (all m, both
        # hemispheres) from the scattered state, in which every member of a kept orbit is written
        # (+- its representative): F(E x) of the subspace state, of which the system keeps the
        # representative rows. State, solves and histories stay in the subspace (the gain); only
        # the folded evaluation of F (hemisphere, sector) is the reference evaluator's (declared)
        self.reduced = bool(getattr(solver.system, "reduced", False))
        if not S.real:
            raise UnsupportedError("f_backend='fused' needs a RealFourier basis in phi (real "
                                   "fields: the transforms use the conjugate symmetry)",
                                   fix="f_backend='evaluator' (ComplexFourier)")
        state = frozenset(solver.system.variables)
        self.exprs, self.notes = [], []
        for i, e in enumerate(solver._rhs_exprs):
            if e is None:
                self.exprs.append(None)
                continue
            _check_real(e, f"right-hand side of equation {i}")
            e2, notes = distribute_derivatives(e, state)
            self.exprs.append(e2)
            self.notes.extend(notes)
        self.rpow = list(solver.plan.rhs_rpow)
        self.shapes = list(solver.system.eq_shapes)
        self.ev = _FusedEvaluator(S, solver.xp, state)
        self.n_calls = 0

    def evaluate(self, data: Mapping, params: Mapping) -> list:
        """Level-0 coefficients of every right-hand side (None where the equation has none)."""
        self.ev.begin()
        self.n_calls += 1
        out = [None if e is None else
               embed_coeffs(self.ev.evaluate(e, data, params, k), shape,
                            f"right-hand side of equation {i}")
               for i, (e, k, shape) in enumerate(zip(self.exprs, self.rpow, self.shapes))]
        self.ev.begin()                                    # release the per-call values
        return out

    def explain(self) -> str:
        S = self.ev.S
        lines = ["  F: planned spherical pseudospectral evaluation (f_backend='fused'): the "
                 "reference evaluator's operators, common sub-expressions once per call, constants "
                 f"(NCC values and derivatives) cached on the {'GPU' if self.ev.xp is not np else 'host'}, "
                 "real GEMMs for the (real) transform matrices, one FFT per spin pair of a real "
                 "tensor (m = 0 symmetrized), real FFTs for radial/scalar components, angular "
                 f"GEMMs in {S.nchunk} m-chunk(s) on the existing degrees l >= max(|m|, |s|) "
                 f"(exactly-zero terms dropped; rule: {self.ev.chunk_rule}), GEMMs by "
                 f"{'direct cuBLAS (capturable)' if self.ev.xp is not np and _direct_cublas() else 'matmul'}; "
                 "numbers = reference up to rounding (summation order)"]
        if self.notes:
            lines.append("    distributed derivatives (exact algebra): " + "; ".join(self.notes))
        if self.ev._const:
            lines.append(f"    {len(self.ev._const)} constant node(s) cached")
        if self.n_calls:
            lines.append(f"    common sub-expressions reused {self.ev.hits / self.n_calls:.0f} time(s) per call")
        lines.append(f"    dealiased grid {' x '.join(str(v) for v in S.sizes(True))}")
        if self.reduced:
            lines.append("    symmetry subspace: F planned on the full grid from the embedded state E x "
                         "(not folded: the folded F is f_backend='evaluator'); state, solves and "
                         "histories in the subspace, representative rows of F kept")
        return "\n".join(lines)
