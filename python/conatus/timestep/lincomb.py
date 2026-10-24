"""IMEX right-hand sides as one elementwise kernel.

The integrators accumulate ``rhs = MX0 + sum_k (dt a_ik) F_k - (dt h_ik) L X_k`` (RK) or
``rhs = sum_j (c_j F_j - (a_j / dt) M X_j)`` (SBDF) one array operation at a time: two kernels
and one temporary per term. On the GPU the same sequence of roundings is evaluated in one
kernel: every product and sum is an explicitly rounded IEEE operation (``__dmul_rn``,
``__dadd_rn``, ``__dsub_rn``: no fused multiply-add contraction), in the order of the array
code, so the result is bitwise the result of the array operations (and of NumPy on the CPU).
Real float64 vectors only; anything else takes the array operations.

Operands may be :class:`CsrApply` (``A @ x`` of a device CSR matrix, e.g. ``M X_0`` or ``L X_k``):
on the GPU the row dot product is evaluated inside the same kernel, with the loop of
``DeviceCsr`` (same order, same contraction), so no separate matvec kernel or temporary is needed.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["rk_rhs", "sbdf_rhs", "CsrApply"]

_kernels: dict = {}


class CsrApply:
    """Deferred ``A @ x`` (A: DeviceCsr or scipy/numpy matrix): computed where it is used."""

    __slots__ = ("A", "x", "_v")

    def __init__(self, A, x) -> None:
        self.A, self.x, self._v = A, x, None

    def value(self):
        """A @ x, computed on first use and kept."""
        if self._v is None:
            self._v = self.A @ self.x
        return self._v


def _val(v):
    return v.value() if isinstance(v, CsrApply) else v


def _matrix(A):
    """The matrix behind a profiling proxy (``runtime.phases._MatProxy.probed_target``): a probed
    solver must run the same program as an unprobed one. The proxy failed the ``DeviceCsr`` test
    below, so every probed RK step took the array path (MX0, L X_k and one temporary per term:
    other kernels and other memory than the run being profiled, gpu4)."""
    return getattr(A, "probed_target", A)


def _fusable(xp, v) -> bool:
    if isinstance(v, CsrApply):
        A = _matrix(v.A)
        return (hasattr(A, "indptr") and hasattr(A, "data") and getattr(A, "dtype", None) == np.float64
                and v.x.dtype == np.float64 and v.x.ndim == 1 and type(A).__name__ == "DeviceCsr")
    return v.dtype == np.float64


def _kernel(cp, key: tuple, body: str, nvec: int, nsc: int):
    k = _kernels.get(key)
    if k is None:
        params = ", ".join([f"float64 v{i}" for i in range(nvec)] + [f"float64 c{i}" for i in range(nsc)])
        k = _kernels[key] = cp.ElementwiseKernel(params, "float64 r", body, "cn_lincomb_" + str(abs(hash(key))))
    return k


def _gpu_ok(xp, vecs) -> bool:
    return xp is not np and all(v.dtype == np.float64 for v in vecs)


def rk_rhs(xp, base, terms: Sequence[tuple]):
    """``base + sum`` over ``terms`` = [(sign, coef, vec)], sign '+' or '-', applied in order as
    ``rhs += coef * vec`` / ``rhs -= coef * vec`` (a new array; ``base`` is not modified).
    ``base`` and the vectors may be :class:`CsrApply`."""
    ops_ = [base] + [v for _, _, v in terms]
    if xp is np or not all(_fusable(xp, v) for v in ops_):
        rhs = _val(base).copy()
        for sg, c, v in terms:
            if sg == "+":
                rhs += c * _val(v)
            else:
                rhs -= c * _val(v)
        return rhs
    if any(isinstance(v, CsrApply) for v in ops_):
        return _rk_rhs_csr(xp, base, terms)
    ops = "".join(sg for sg, _, _ in terms)
    lines = ["double a = v0;"]
    for i, (sg, _, _) in enumerate(terms):
        f = "__dadd_rn" if sg == "+" else "__dsub_rn"
        lines.append(f"a = {f}(a, __dmul_rn(c{i}, v{i + 1}));")
    lines.append("r = a;")
    k = _kernel(xp, ("rk", ops), "\n".join(lines), len(terms) + 1, len(terms))
    return k(base, *[v for _, _, v in terms], *[np.float64(c) for _, c, _ in terms])


def _rk_rhs_csr(xp, base, terms):
    """One raw kernel: operands are plain vectors or in-kernel CSR row products."""
    ops_ = [base] + [v for _, _, v in terms]
    kinds = tuple("c" if isinstance(v, CsrApply) else "v" for v in ops_)
    signs = "".join(sg for sg, _, _ in terms)
    key = ("rkcsr", kinds, signs)
    k = _kernels.get(key)
    if k is None:
        params, exprs = ["long long n", "double* out"], []
        for j, kd in enumerate(kinds):
            if kd == "v":
                params.append(f"const double* v{j}")
                exprs.append(f"v{j}[i]")
            else:
                params += [f"const int* p{j}", f"const int* j{j}", f"const double* d{j}",
                           f"const double* x{j}"]
                exprs.append(f"csr{j}")
        params += [f"double c{t}" for t in range(len(terms))]
        body = ["long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;", "if (i >= n) return;"]
        for j, kd in enumerate(kinds):
            if kd == "c":             # as DeviceCsr's csr_mm (same loop, same contraction)
                body.append(f"double csr{j} = 0.0; for (int q = p{j}[i]; q < p{j}[i + 1]; ++q) "
                            f"csr{j} += d{j}[q] * x{j}[(long long)j{j}[q]];")
        body.append(f"double a = {exprs[0]};")
        for t, sg in enumerate(signs):
            f = "__dadd_rn" if sg == "+" else "__dsub_rn"
            body.append(f"a = {f}(a, __dmul_rn(c{t}, {exprs[t + 1]}));")
        body.append("out[i] = a;")
        src = ('extern "C" __global__ void rk_rhs_csr(' + ", ".join(params) + ") {\n  "
               + "\n  ".join(body) + "\n}\n")
        k = _kernels[key] = xp.RawKernel(src, "rk_rhs_csr", options=("-std=c++17",))
    n = int(_val_len(base))
    out = xp.empty(n, dtype=np.float64)
    args = [np.int64(n), out]
    for v in ops_:
        if isinstance(v, CsrApply):
            args += [v.A.indptr, v.A.indices, v.A.data, xp.ascontiguousarray(v.x)]
        else:
            args.append(xp.ascontiguousarray(v))     # raw pointer: a strided view would be misread
    args += [np.float64(c) for _, c, _ in terms]
    k(((n + 255) // 256,), (256,), tuple(args))
    return out


def _val_len(v):
    return v.A.shape[0] if isinstance(v, CsrApply) else v.shape[0]


def sbdf_rhs(xp, n: int, dtype, pairs: Sequence[tuple]):
    """``sum_j (c_j F_j - d_j MX_j)`` over ``pairs`` = [(c_j, F_j, d_j, MX_j)], accumulated from
    zero in order, each term formed as ``c F - d MX`` (as ``rhs += c * F - d * MX``)."""
    if not _gpu_ok(xp, [v for _, F, _, M in pairs for v in (F, M)]):
        rhs = xp.zeros(n, dtype=dtype)
        for c, F, d, M in pairs:
            rhs += c * F - d * M
        return rhs
    lines = ["double a = 0.0;"]
    for i in range(len(pairs)):
        lines.append(f"a = __dadd_rn(a, __dsub_rn(__dmul_rn(c{2 * i}, v{2 * i}), "
                     f"__dmul_rn(c{2 * i + 1}, v{2 * i + 1})));")
    lines.append("r = a;")
    k = _kernel(xp, ("sbdf", len(pairs)), "\n".join(lines), 2 * len(pairs), 2 * len(pairs))
    vecs = [v for _, F, _, M in pairs for v in (F, M)]
    cs = [np.float64(x) for c, _, d, _ in pairs for x in (c, d)]
    return k(*vecs, *cs)
