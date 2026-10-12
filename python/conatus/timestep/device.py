"""GPU-resident stacked system for ``IVPSolver(..., backend="cuda")`` ( → performance target).

Mirror of :class:`GlobalSystem` whose state, matrices and gather/scatter maps live on the
device (CuPy arrays). Linear solves: the batched bordered banded GPU solver of
``conatus.runtime.cuda`` (one warp per mode), called in place on CuPy memory through
``solve_ptr`` (same default stream). The host GlobalSystem stays the single source of the
structure (ordering, permutations, buckets): nothing is re-derived here.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..runtime.banded import solve_operand
from .shared_factors import SharedFactor
from .system import GlobalSystem
from ..errors import OptionError

__all__ = ["DeviceSystem", "DeviceCsr"]

_CSR_SRC = r"""
#include <cupy/complex.cuh>
template <typename T>
__device__ void csr_mm(const int* indptr, const int* indices, const T* data, const T* B, T* C,
                       long long nrows, long long ncols) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nrows * ncols) return;
    long long row = i / ncols, col = i % ncols;
    T acc = T(0);
    for (int k = indptr[row]; k < indptr[row + 1]; ++k) acc += data[k] * B[(long long)indices[k] * ncols + col];
    C[i] = acc;
}
extern "C" __global__ void csr_mm_d(const int* p, const int* j, const double* a, const double* B, double* C,
                                    long long n, long long m) { csr_mm<double>(p, j, a, B, C, n, m); }
extern "C" __global__ void csr_mm_z(const int* p, const int* j, const complex<double>* a,
                                    const complex<double>* B, complex<double>* C, long long n, long long m) {
    csr_mm<complex<double>>(p, j, a, B, C, n, m); }
"""
_csr_module = None

# Complex-mode buckets (two RealFourier directions, system.py): the gather of the complex right-hand
# side Z = d * sum_p w_p rhs[rows_p] and the scatter-add x[cols_p] += Re(w_p d op(Z)) in one pass
# each (op = identity or conjugation), instead of ~30 CuPy operations and their temporaries per
# bucket. Same arithmetic per entry, in the same order as the CuPy expressions they replace (the
# weights are products of +-1, +-i and 1/2: every product is exact).
_ZB_SRC = r"""
#include <cupy/complex.cuh>
typedef complex<double> cd;
extern "C" __global__ void zb_gather(const double* rhs, const int* idx, const cd* w, int P,
                                     long long N, const double* d, long long dmod, int conj_out,
                                     cd* Z) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= N) return;
    cd acc = w[0] * rhs[idx[q]];
    for (int p = 1; p < P; ++p) acc = acc + w[p] * rhs[idx[(long long)p * N + q]];
    if (conj_out) acc = conj(acc);
    if (d) acc = acc * d[q % dmod];
    Z[q] = acc;
}
extern "C" __global__ void zb_scatter(const cd* Z, const int* idx, const cd* w, int P,
                                      long long N, const double* d, long long dmod, int conj_in,
                                      double* x) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= N) return;
    cd z = Z[q];
    if (conj_in) z = conj(z);
    if (d) z = z * d[q % dmod];
    for (int p = 0; p < P; ++p) {
        long long i = idx[(long long)p * N + q];   // int32 map (n < 2^31, _index_map)
        x[i] = x[i] + (w[p] * z).real();
    }
}
"""
_zb_module = None


class DeviceCsr:
    """CSR matrix on the device with a plain CUDA kernel for A @ B (vector or row-major matrix).
    Replaces cuSPARSE here because CuPy refuses cuSPARSE calls during CUDA-graph capture."""

    def __init__(self, A, dtype) -> None:
        import cupy as cp
        global _csr_module
        if _csr_module is None:
            _csr_module = cp.RawModule(code=_CSR_SRC, options=("-std=c++17",))
        A = A.tocsr().astype(dtype)
        A.sort_indices()
        self.shape, self.dtype = A.shape, np.dtype(dtype)
        self.indptr = cp.asarray(A.indptr.astype(np.int32))
        self.indices = cp.asarray(A.indices.astype(np.int32))
        self.data = cp.asarray(A.data)
        self._k = _csr_module.get_function("csr_mm_z" if self.dtype.kind == "c" else "csr_mm_d")

    def __matmul__(self, B):
        import cupy as cp
        vec = B.ndim == 1
        B2 = cp.ascontiguousarray(B.reshape(B.shape[0], -1).astype(self.dtype, copy=False))
        n, m = self.shape[0], B2.shape[1]
        C = cp.empty((n, m), dtype=self.dtype)
        tot = n * m
        if tot:
            self._k(((tot + 255) // 256,), (256,),
                    (self.indptr, self.indices, self.data, B2, C, np.int64(n), np.int64(m)))
        return C.reshape(n) if vec else C



def _index_map(cp, a):
    """Integer index map on the device as int32 when every index fits (half the bytes of int64: ~1 GiB of maps at the this case's (640, 64, 192)), else unchanged."""
    a = np.asarray(a)
    if a.dtype.kind in "iu" and (a.size == 0 or (a.min() >= 0 and a.max() < 2 ** 31 - 1)):
        a = a.astype(np.int32)
    return cp.asarray(a)


def _core_has_compact(cuda_mod) -> bool:
    """The compiled core has the transposed level schedule and the band release."""
    cu = getattr(getattr(cuda_mod, "_core", None), "cuda", None)
    return cu is not None and hasattr(getattr(cu, "BandedBatchD", None), "build_transpose_schedule")


def _cupy():
    import cupy as cp
    import cupyx.scipy.sparse as csp
    return cp, csp


class DeviceSystem:
    """Device copy of a GlobalSystem (same ordering, buckets and dtype)."""

    def __init__(self, host: GlobalSystem, pivot: bool = True, level_schedule: bool = True,
                 level_merge: bool = True) -> None:
        from ..runtime import cuda
        if not cuda.available():
            from ..errors import BackendError
            raise BackendError(
                "backend='cuda' requested but no CUDA device is available",
                where="IVPSolver(..., backend='cuda')",
                why=cuda.unavailable_reason(),
                fix="run in a job on a GPU node of your cluster, build the core with CUDA "
                    "(docs/site/install.md#gpu-cuda-install: CONATUS_CUDA=ON, CuPy from "
                    "conda-forge or the gpu extra of a wheel), or use backend='cpu'")
        cp, csp = _cupy()
        self.cp, self.csp, self._cuda = cp, csp, cuda
        self.host = host
        self.pivot = bool(pivot)
        # level-scheduled solves: same LU factors and per-row arithmetic as the band kernel,
        # factors streamed without their structural zeros (core/src/cuda/banded.cu)
        self.level_schedule = bool(level_schedule)
        # merged levels: several dependency levels per warp step (rounding-level differences)
        self.level_merge = bool(level_merge)
        for name in ("plan", "reduced", "ordering", "variables", "var_shapes", "eq_shapes", "n",
                     "complex", "dtype", "slots"):
            setattr(self, name, getattr(host, name))
        self.M = DeviceCsr(host.M, host.dtype)
        self.L = DeviceCsr(host.L, host.dtype)
        self.solve_device = None                 # use_solve_device: factors on another GPU
        self._solve_maps()
        self._col_sel = [(_index_map(cp, sel), _index_map(cp, host.cols_flat[sel]))
                         for sel in host._col_sel]
        self._row_sel = {key: (_index_map(cp, sel), _index_map(cp, host.rows_flat[sel]))
                         for key, sel in host._row_sel.items()}
        self._factors: dict = {}
        self._conv: dict = {}
        # compact_factors (opt-in, declared in explain()): after each factorization,
        # build the transposed level schedule and release the band LU, keeping the factors only
        # as level streams (both directions). Transposed solves then run on the streams: equal
        # to the band kernel to rounding (summation order). Set before the first solve.
        self._compact = False
        self._compacted = 0

    @property
    def compact_factors(self) -> bool:
        """Keep the factors only as level streams (opt-in; see __init__)."""
        return self._compact

    @compact_factors.setter
    def compact_factors(self, on: bool) -> None:
        # refused, never ignored ( step 3): a core without the transposed level
        # schedule, or solves without level schedule, used to leave every batch uncompacted with
        # only a count in explain() -- the memory plan of the caller then silently did not hold
        on = bool(on)
        if on:
            from ..errors import BackendError, OptionError
            if not _core_has_compact(self._cuda):
                raise BackendError(
                    "compact_factors=True: the compiled core has no transposed level-scheduled "
                    "solves (BandedBatch.build_transpose_schedule)",
                    where="DeviceSystem.compact_factors",
                    why="the core was built before",
                    fix="build the core of this tree (bash core/build.sh, or CN_CORE_BUILD=<dir> and "
                        "CONATUS_CORE_PATH=<dir>/core/core), or keep compact_factors=False")
            if not self.level_schedule:
                raise OptionError("compact_factors=True needs level_schedule=True (the factors are "
                                  "kept as level streams)", where="DeviceSystem.compact_factors",
                                  fix="leave level_schedule on, or keep compact_factors=False")
        self._compact = on

    def _solve_maps(self) -> None:
        """Gather/scatter maps of the solves, on the current device."""
        cp, host = self.cp, self.host
        self._buckets = [(idx, cp.ascontiguousarray(cp.asarray(rows, dtype=cp.int64)),
                          cp.ascontiguousarray(cp.asarray(cols, dtype=cp.int64)))
                         for idx, rows, cols in host._buckets]
        # complexified buckets: complex solves of half size, results split into cos/sin
        self._cbuckets = [(idx, *(cp.ascontiguousarray(cp.asarray(a, dtype=cp.int64)) for a in arrs))
                          for idx, *arrs in host._cbuckets]
        # >= 2 RealFourier directions: complex modes exp(i(kx x +- kz z)) (see system.py)
        self._zbuckets = [(keys, _index_map(cp, rows), _index_map(cp, cols),
                           [complex(w) for w in wr], [complex(w) for w in wc])
                          for keys, rows, cols, wr, wc in host._zbuckets]
        # fused gather/scatter of the complex-mode buckets (one kernel each way; False: the CuPy
        # expressions, kept as the reference); weights as device arrays for the kernels
        self.fused_zmaps = (bool(self._zbuckets) and host.dtype == np.float64
                            and all(r.dtype == np.int32 and c.dtype == np.int32
                                    for _k, r, c, *_w in self._zbuckets))
        self._zw = [(cp.asarray(np.asarray(wr, dtype=np.complex128)),
                     cp.asarray(np.asarray(wc, dtype=np.complex128)))
                    for _k, _r, _c, wr, wc in host._zbuckets]

    def use_solve_device(self, device: int) -> None:
        """Keep the factors and run the batched solves on another GPU of the node (opt-in, declared in explain()): right-hand sides and solutions cross by peer copies
        (cudaMemcpyPeer, ordered with the work of both devices); the state, the operators and F
        stay on the current device. Same kernels and factors: the solutions are unchanged. To be
        called before the first factorization."""
        cp = self.cp
        if self._factors:
            raise RuntimeError("use_solve_device: call it before the first factorization")
        dev, home = int(device), cp.cuda.Device().id
        if not 0 <= dev < cp.cuda.runtime.getDeviceCount():
            raise OptionError(f"use_solve_device: no device {dev} "
                              f"({cp.cuda.runtime.getDeviceCount()} visible)",
                              fix="a device index from 0 to the number of visible GPUs - 1")
        if dev == home:
            self.solve_device = None
            return
        with cp.cuda.Device(dev):
            self._solve_maps()
        self.solve_device, self._home = dev, home
        self._side = None

    def _peer(self, src):
        """Copy of ``src`` on the current device (cudaMemcpyPeer from src's device)."""
        cp = self.cp
        dst = cp.empty(src.shape, src.dtype)
        cp.cuda.runtime.memcpyPeer(dst.data.ptr, dst.device.id, src.data.ptr, src.device.id,
                                   src.nbytes)
        return dst

    # ------------------------------------------------------------------ solves
    def factor(self, a: float, b: float):
        if self.solve_device is not None:
            with self.cp.cuda.Device(self.solve_device):
                return self._factor(a, b)
        return self._factor(a, b)

    def _factor(self, a: float, b: float):
        key = (float(a), float(b))
        f = self._factors.get(key)
        if f is None:
            f = []
            for idx, _, _ in self.host._buckets:
                mats = []
                for i in idx:
                    s = self.slots[i]
                    mats.append((a * s.M + b * s.L).tocsr()[s.rperm][:, s.cperm].astype(self.dtype))
                f.append(self._batch(mats, list(range(self.slots[idx[0]].n_border)),
                                     pivot=self.pivot, dtype=self.dtype,
                                     level_schedule=self.level_schedule,
                                     level_merge=self.level_merge))
            for idx, *_ in self.host._cbuckets:
                czs = [self.slots[i].cz for i in idx]
                mats = [(a * cz.M + b * cz.L).tocsr()[cz.rperm][:, cz.cperm].astype(np.complex128)
                        for cz in czs]
                f.append(self._batch(mats, list(range(czs[0].n_border)),
                                     pivot=self.pivot, dtype=np.complex128,
                                     level_schedule=self.level_schedule,
                                     level_merge=self.level_merge))
            # complex modes: buckets equal up to signs share the factors of their partner
            # (host._zbucket_matrices checks it exactly; timestep/shared_factors.py)
            base = len(f)
            for mats, nb, shared in self.host._zbucket_matrices(a, b):
                if shared is not None:
                    f.append(SharedFactor(shared.primary, self.cp.asarray(shared.dr),
                                          self.cp.asarray(shared.dc)))
                    continue
                f.append(self._batch(mats, list(range(nb)),
                                     pivot=self.pivot, dtype=np.complex128,
                                     level_schedule=self.level_schedule,
                                     level_merge=self.level_merge))
            self.host._share_log(key, f[base:])
            self._factors[key] = f
        return f

    def _batch(self, *args, **kw):
        """cuda.BandedBatch (compacted at once with compact_factors: its band LU released before
        the next batch is factored, so the band storage of only one batch is alive at a time, as
        the memory planner models it step 3)."""
        bb = self._batch_alloc(*args, **kw)
        if self._compact:
            # a batch whose schedule does not apply keeps its band LU (level_note / level_t_note
            # say why; counted in explain())
            if bb.compact():
                self._compacted += 1
            self._cuda.pool_trim()                  # the band storage back to the driver
        return bb

    def _batch_alloc(self, *args, **kw):
        """cuda.BandedBatch, retried once after returning CuPy's cached free blocks to the driver
        when the core pool's cudaMalloc runs out of memory (CuPy retries its own allocations that
        way; the core's did not, so blocks cached by CuPy could make a factorization fail)."""
        try:
            return self._cuda.BandedBatch(*args, **kw)
        except (RuntimeError, MemoryError) as err:
            if "out of memory" not in str(err).lower():
                raise
            pool = self.cp.get_default_memory_pool()
            freed = pool.free_bytes()
            pool.free_all_blocks()
            try:
                return self._cuda.BandedBatch(*args, **kw)
            except (RuntimeError, MemoryError) as err2:
                if "out of memory" not in str(err2).lower():
                    raise
                # a real shortage: say so with the numbers ( step 3)
                free, total = self.cp.cuda.runtime.memGetInfo()
                st = self._cuda.pool_stats()
                raise MemoryError(
                    f"DeviceSystem: out of device memory factoring a batch, also after returning "
                    f"{freed / 2**30:.2f} GiB of CuPy's cached blocks to the driver (device: "
                    f"{free / 2**30:.2f} of {total / 2**30:.2f} GiB free; CuPy pool "
                    f"{pool.used_bytes() / 2**30:.2f} GiB in use; core pool "
                    f"{st['live_bytes'] / 2**30:.2f} GiB live). Plan the memory first "
                    "(conatus.runtime.memory.memory_report) and use the declared levers "
                    "(complexify, compact_factors, use_solve_device).") from err2

    def solve(self, a: float, b: float, rhs):
        return self._remote(self._solve_here, a, b, rhs, "DeviceSystem.solve")

    def solve_transpose(self, a: float, b: float, rhs):
        """Device version of GlobalSystem.solve_transpose (same factors, batched transposed
        solves on the GPU; see the host docstring for the index and complex conventions)."""
        return self._remote(self._solve_transpose_here, a, b, rhs, "DeviceSystem.solve_transpose")

    def _remote(self, fn, a, b, rhs, where):
        if self.solve_device is None:
            return fn(a, b, rhs)
        cp = self.cp
        if cp.cuda.get_current_stream().is_capturing():
            from ..errors import OptionError
            raise OptionError("use_solve_device: solves on another GPU cannot be captured in a CUDA "
                              "graph (peer copies on the legacy stream)", where=where,
                              fix="advance with run() instead of run_graph(), or keep the factors "
                                  "on the state's GPU")
        rhs = solve_operand(rhs, self.dtype, where, cp)
        with cp.cuda.Device(self.solve_device):
            x = fn(a, b, self._peer(rhs))
        return self._peer(x)

    def _solve_here(self, a: float, b: float, rhs):
        cp = self.cp
        f = self._factor(a, b)
        x = (cp.zeros if self._zbuckets else cp.empty)(self.n, dtype=self.dtype)
        stream = int(cp.cuda.get_current_stream().ptr)
        nr, nc = len(self._buckets), len(self._cbuckets)
        # the kernels read raw pointers typed by the factors
        rhs = solve_operand(rhs, self.dtype, "DeviceSystem.solve", cp)
        rp, xp_ = int(rhs.data.ptr), int(x.data.ptr)
        launches = []                           # (batch size, launch(stream)) of the fused kernels
        for (_idx, rows, cols), batch in zip(self._buckets, f[:nr], strict=False):
            # gather rhs[rows] and scatter x[cols] inside the solve kernel (no staging copies)
            launches.append((batch.batch, lambda st, b=batch, r=rows, c=cols: b._s.solve_mapped_ptr(
                rp, int(r.data.ptr), xp_, int(c.data.ptr), st)))
        for (_idx, rc, rs, cc, cs), batch in zip(self._cbuckets, f[nr:nr + nc], strict=False):
            if rhs.dtype == np.float64:         # z = rhs[rc] + i rhs[rs], split back, in the kernel
                launches.append((batch.batch, lambda st, b=batch, a=(rc, rs, cc, cs): b._s.solve_split_ptr(
                    rp, *(int(v.data.ptr) for v in a[:2]), xp_, *(int(v.data.ptr) for v in a[2:]), st)))
                continue
            Z = rhs[rc] + 1j * rhs[rs]          # complex128, contiguous
            batch._s.solve_ptr(int(Z.data.ptr), int(Z.shape[0]), int(Z.shape[1]), stream)
            x[cc] = Z.real
            x[cs] = Z.imag
        self._concurrent(launches)
        if self.fused_zmaps:
            self._zsolve(f, f[nr + nc:], rhs, x, stream, transpose=False)
            return x
        for (_keys, rows, cols, wr, wc), batch in zip(self._zbuckets, f[nr + nc:], strict=False):
            Z = sum(w * rhs[r] for w, r in zip(wr, rows, strict=False)).astype(np.complex128)
            sh = batch if isinstance(batch, SharedFactor) else None
            if sh is not None:                  # B = D_r A D_c: x = D_c A^{-1} D_r b (exact)
                Z, batch = Z * sh.dr, f[sh.primary]
            Z = cp.ascontiguousarray(Z)
            batch._s.solve_ptr(int(Z.data.ptr), int(Z.shape[0]), int(Z.shape[1]), stream)
            if sh is not None:
                Z = Z * sh.dc
            for w, c in zip(wc, cols, strict=False):
                x[c] += (w * Z).real
        return x

    def _zsolve(self, f, zf, rhs, x, stream: int, transpose: bool) -> None:
        """Complex-mode buckets with the fused gather/scatter kernels (see _ZB_SRC). Forward:
        Z = D_r sum_p wr_p rhs[rows_p], A Z' = Z, x[cols_p] += Re(wc_p D_c Z'). Transposed (the
        roles of rows and columns swap): Z = D_c conj(sum_p wr_p rhs[cols_p]), A^T Z' = Z,
        x[rows_p] += Re(wc_p D_r conj(Z')). D = 1 for a bucket with its own factors."""
        cp = self.cp
        global _zb_module
        if _zb_module is None:
            # no FMA contraction: acc + w * rhs rounded as the separate CuPy operations it replaces
            # (contracted, the gather differed from the CuPy path in the last bit; review)
            _zb_module = cp.RawModule(code=_ZB_SRC, options=("-std=c++17", "--fmad=false"))
        kg, ks = _zb_module.get_function("zb_gather"), _zb_module.get_function("zb_scatter")
        for (_keys, rows, cols, _wr, _wc), batch, (wr, wc) in zip(self._zbuckets, zf, self._zw, strict=False):
            sh = batch if isinstance(batch, SharedFactor) else None
            bb = f[sh.primary] if sh is not None else batch
            P, B, m = rows.shape
            N = B * m
            Z = cp.empty((B, m), dtype=np.complex128)
            src, dst = (cols, rows) if transpose else (rows, cols)
            d_in = (sh.dc if transpose else sh.dr) if sh is not None else None
            d_out = (sh.dr if transpose else sh.dc) if sh is not None else None
            grid = ((N + 255) // 256,)

            def dptr(d):
                return (0, 1) if d is None else (int(d.data.ptr), int(d.size))
            pi, mi = dptr(d_in)
            kg(grid, (256,), (rhs, src, wr, np.int32(P), np.int64(N), np.uint64(pi), np.int64(mi),
                              np.int32(transpose), Z))
            if transpose:
                bb._s.solve_transpose_ptr(int(Z.data.ptr), int(B), int(m), stream)
            else:
                bb._s.solve_ptr(int(Z.data.ptr), int(B), int(m), stream)
            po, mo = dptr(d_out)
            ks(grid, (256,), (Z, dst, wc, np.int32(P), np.int64(N), np.uint64(po), np.int64(mo),
                              np.int32(transpose), x))

    def _concurrent(self, launches) -> None:
        """Independent bucket solves (disjoint rows of x) run concurrently: the largest batch on
        the current stream, the others on side streams forked and joined by events (also inside
        a CUDA-graph capture, where they become parallel branches). A bucket of one mode (k = 0)
        is a single block: serialised, it would add its whole dependency chain to the step."""
        cp = self.cp
        main = cp.cuda.get_current_stream()
        launches = sorted(launches, key=lambda t: -t[0])
        if len(launches) <= 1 or not getattr(self, "concurrent", True):
            for _, go in launches:
                go(int(main.ptr))
            return
        side = getattr(self, "_side", None)
        if side is None or len(side) < len(launches) - 1:
            side = self._side = [(cp.cuda.Stream(non_blocking=True), cp.cuda.Event(block=False,
                                  disable_timing=True)) for _ in range(len(launches) - 1)]
            self._fork = cp.cuda.Event(block=False, disable_timing=True)
        self._fork.record(main)
        for (st, ev), (_, go) in zip(side, launches[1:], strict=False):
            st.wait_event(self._fork)
            go(int(st.ptr))
            ev.record(st)
        launches[0][1](int(main.ptr))
        for _st, ev in side[:len(launches) - 1]:
            main.wait_event(ev)

    def _solve_transpose_here(self, a: float, b: float, rhs):
        cp = self.cp
        rhs = solve_operand(rhs, self.dtype, "DeviceSystem.solve_transpose", cp)
        f = self._factor(a, b)
        x = (cp.zeros if self._zbuckets else cp.empty)(self.n, dtype=self.dtype)
        stream = int(cp.cuda.get_current_stream().ptr)
        nr, nc = len(self._buckets), len(self._cbuckets)
        for (_idx, rows, cols), batch in zip(self._buckets, f[:nr], strict=False):
            B = cp.ascontiguousarray(rhs[cols])
            batch._s.solve_transpose_ptr(int(B.data.ptr), int(B.shape[0]), int(B.shape[1]), stream)
            x[rows] = B
        zbs = self._zbuckets if not self.fused_zmaps else []
        if self.fused_zmaps:
            self._zsolve(f, f[nr + nc:], rhs, x, stream, transpose=True)
        for (_keys, rows, cols, wr, wc), batch in zip(zbs, f[nr + nc:], strict=False):
            Z = cp.conj(sum(w * rhs[c] for w, c in zip(wr, cols, strict=False)))
            sh = batch if isinstance(batch, SharedFactor) else None
            if sh is not None:                  # B^T = D_c A^T D_r: x = D_r A^{-T} D_c b (exact)
                Z, batch = Z * sh.dc, f[sh.primary]
            Z = cp.ascontiguousarray(Z)
            batch._s.solve_transpose_ptr(int(Z.data.ptr), int(Z.shape[0]), int(Z.shape[1]), stream)
            Z = cp.conj(Z)
            if sh is not None:
                Z = Z * sh.dr
            for w, r in zip(wc, rows, strict=False):
                x[r] += (w * Z).real
        for (_idx, rc, rs, cc, cs), batch in zip(self._cbuckets, f[nr:nr + nc], strict=False):
            Z = cp.ascontiguousarray(cp.conj(rhs[cc] + 1j * rhs[cs]))
            batch._s.solve_transpose_ptr(int(Z.data.ptr), int(Z.shape[0]), int(Z.shape[1]), stream)
            Z = cp.conj(Z)
            x[rc] = Z.real
            x[rs] = Z.imag
        return x

    @property
    def n_factorizations(self) -> int:
        return len(self._factors)

    # ------------------------------------------------------------------ gather / scatter
    def scatter(self, X, only: frozenset | None = None) -> list:
        """Device version of GlobalSystem.scatter (``only``: variables needed, others None)."""
        cp = self.cp
        out = []
        for v, (sel, flat), shape in zip(self.variables, self._col_sel, self.var_shapes, strict=False):
            if only is not None and v.name not in only:
                out.append(None)
                continue
            a = cp.zeros(int(np.prod(shape)), dtype=self.dtype)
            a[flat] = X[sel]
            a = a.reshape(shape)
            out.append(a if self.complex else a.real)
        return out

    def gather_state(self, coeffs: Sequence, check_subspace: bool = True):
        host = [self.cp.asnumpy(c) if hasattr(c, "device") else np.asarray(c) for c in coeffs]
        return self.cp.asarray(self.host.gather_state(host, check_subspace))

    def gather_rhs(self, eq_coeffs: Sequence):
        cp = self.cp
        b = cp.zeros(self.n, dtype=self.dtype)
        zc = self.plan.layout.zcoord
        coords = self.plan.domain.coordsys.coords
        for (k, lam), (sel, flat) in self._row_sel.items():
            c = eq_coeffs[k]
            if c is None:
                continue
            c = cp.asarray(c)
            if lam > 0:
                eq = self.plan.problem.equations[k]
                ax = eq.lhs.rank + coords.index(zc)
                if c.shape[ax] > 1:
                    S = self._conv.get(lam)
                    if S is None:
                        # the layout's conversion (family-aware: Hermite identity, Laguerre
                        # levels, compound per piece), as the host gather_rhs; the Chebyshev
                        # chain here made the GPU wrong for those bases
                        S = self._conv[lam] = DeviceCsr(self.plan.layout.zconv(lam).tocsr(),
                                                        self.dtype)
                    cm = cp.moveaxis(c, ax, 0)
                    c = cp.moveaxis((S @ cm.reshape(cm.shape[0], -1)).reshape(cm.shape), 0, ax)
            b[sel] = c.reshape(-1)[flat]
        return b

    def explain(self) -> str:
        return (self.host.explain().replace("batched C++ bordered banded solves",
                                            "batched GPU bordered banded solves, one warp per mode")
                + f"\n  state resident on the device (CuPy), pivot={'partial' if self.pivot else 'none (declared)'}"
                + (f"\n  factors and batched solves on GPU {self.solve_device} (peer copies of the "
                   "right-hand side and the solution; state, operators and F on GPU "
                   f"{getattr(self, '_home', 0)})" if self.solve_device is not None else "")
                + "\n  solves: gather/scatter fused in the solve kernel, independent buckets on "
                + ("concurrent streams, " if getattr(self, "concurrent", True) else "one stream, ")
                + self._explain_solve_kernels())

    def _explain_solve_kernels(self) -> str:
        """Which solve kernel the factored batches actually use. A batch may fall back to the band
        kernel although the level schedule was requested (BandedBatch.level_note says why): the
        request alone is not reported as the path taken."""
        if not self.level_schedule:
            return "band kernel (level schedule off)"
        if self.compact_factors:
            nb = sum(1 for fs in self._factors.values() for bb in fs if not isinstance(bb, SharedFactor))
            tail = (f"; compact factors: {self._compacted} of {nb} batches keep only level streams "
                    "(transposed solves on streams of U^T and L'^T, band LU released)")
        else:
            tail = ""
        return self._explain_levels() + tail

    def _explain_levels(self) -> str:
        same = ("same factors as the band kernel; consecutive levels merged into warp steps of "
                "<= 32 rows (a wider level stays a step of its own), equal to rounding"
                if self.level_merge else
                "same factors and per-row arithmetic as the band kernel")
        batches = [bb for fs in self._factors.values() for bb in fs
                   if not isinstance(bb, SharedFactor)]
        if not batches:
            return f"level-scheduled sparse LU factors requested ({same}; built at the first solve)"
        lvl = [bb for bb in batches if bb.level_scheduled]
        if len(lvl) == len(batches):
            return f"level-scheduled sparse LU factors ({same})"
        notes = sorted({bb.level_note for bb in batches if not bb.level_scheduled})
        return (f"level-scheduled sparse LU factors for {len(lvl)} of {len(batches)} batches ({same}); "
                f"band kernel for the others: {'; '.join(notes)}")
