"""Direct cuBLAS DGEMM (strided batched) usable inside a CUDA graph capture.

CuPy refuses every cuBLAS call while a stream is being captured ("calling cuBLAS API during
stream capture is currently unsupported"), although cuBLAS itself supports capture when the
handle owns a fixed workspace. The planned spherical F (``timestep/sphere_fused``) multiplies by
real transform matrices only, so one primitive suffices: a row-major strided-batched DGEMM
C_b = A_b @ B_b, called through ``ctypes`` on the process's ``libcublas`` (the library CuPy
loads) with our own handle per device, the current CuPy stream and a 32 MiB workspace allocated
once (no allocation during a capture). Pointer mode host (alpha = 1, beta = 0, read at the call).

If the library cannot be loaded (``available()`` False), the planned F falls back to
``cupy.matmul`` (same cuBLAS DGEMM through CuPy, declared by its ``explain()``) and a captured
time-dependent step is refused (``IVPSolver.run_graph``): CuPy's cuBLAS cannot be captured.
The handle and its workspace live as long as the process (one per device; never destroyed, so
no ordering problem with the CUDA context at exit). Not thread-safe: one handle per device is
shared by the calls of the process (they are issued from one thread here).
"""
from __future__ import annotations

import ctypes
import ctypes.util

__all__ = ["available", "gemm_rm"]

_LIB = None
_FAILED: str | None = None
_HANDLES: dict = {}
_WORKSPACE = 32 << 20
_ONE, _ZERO = ctypes.c_double(1.0), ctypes.c_double(0.0)
_OP_N = 0


def _lib():
    """The process's libcublas with the prototypes used here (loaded once; a failure is
    remembered and raised again as OSError, so the fallback is decided once)."""
    global _LIB, _FAILED
    if _FAILED is not None:
        raise OSError(_FAILED)
    if _LIB is None:
        import os
        import sys
        last = None
        # absolute candidates only (an empty CUDA_HOME gave the relative 'lib/libcublas.so.12',
        # looked up from the current directory)
        here = [os.path.join(p, "lib", "libcublas.so.12") for p in (sys.prefix, os.environ.get("CUDA_HOME", ""))
                if p and os.path.isabs(p)]
        lib = None
        for name in ("libcublas.so.12", *here, "libcublas.so", ctypes.util.find_library("cublas")):
            if not name:
                continue
            try:
                lib = ctypes.CDLL(name)
                break
            except OSError as err:
                last = err
        try:
            if lib is None:
                raise OSError(f"libcublas not found ({last})")
            f = lib.cublasDgemmStridedBatched
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
                          ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
            for fn, args in (("cublasSetStream_v2", [ctypes.c_void_p, ctypes.c_void_p]),
                             ("cublasSetWorkspace_v2", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]),
                             ("cublasCreate_v2", [ctypes.POINTER(ctypes.c_void_p)])):
                g = getattr(lib, fn)
                g.restype, g.argtypes = ctypes.c_int, args
        except (OSError, AttributeError) as err:          # missing library or symbol
            _FAILED = f"direct cuBLAS unavailable: {err}"
            raise OSError(_FAILED) from err
        _LIB = lib
    return _LIB


def available() -> bool:
    """The direct cuBLAS library can be loaded (else the callers fall back to cupy.matmul)."""
    try:
        _lib()
    except OSError:
        return False
    return True


def _check(st: int, what: str) -> None:
    if st != 0:
        raise RuntimeError(f"cuBLAS {what} failed (status {st})")


def _handle(cp):
    dev = cp.cuda.Device().id
    h = _HANDLES.get(dev)
    if h is None:
        lib = _lib()
        hp = ctypes.c_void_p()
        _check(lib.cublasCreate_v2(ctypes.byref(hp)), "create")
        # from CuPy's default pool, not the caller's allocator (run_graph allocates in a private
        # pool per captured graph): the workspace lives with the handle, for the process
        with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):
            ws = cp.empty(_WORKSPACE, dtype=cp.uint8)
        _check(lib.cublasSetWorkspace_v2(hp, ctypes.c_void_p(ws.data.ptr), _WORKSPACE), "workspace")
        h = _HANDLES[dev] = (hp, ws)
    return h[0]


def gemm_rm(cp, rows: int, cols: int, k: int, a_ptr: int, lda: int, sa: int, b_ptr: int, ldb: int,
            sb: int, c_ptr: int, ldc: int, sc: int, batch: int) -> None:
    """Row-major float64 C_b (rows x cols) = A_b (rows x k) @ B_b (k x cols), b < batch; leading
    dimensions and batch strides in elements, pointers in bytes (device). Column-major view:
    C^T = B^T A^T."""
    lib = _lib()
    h = _handle(cp)
    _check(lib.cublasSetStream_v2(h, ctypes.c_void_p(cp.cuda.get_current_stream().ptr)), "set stream")
    _check(lib.cublasDgemmStridedBatched(h, _OP_N, _OP_N, cols, rows, k, ctypes.addressof(_ONE),
                                         b_ptr, ldb, sb, a_ptr, lda, sa, ctypes.addressof(_ZERO),
                                         c_ptr, ldc, sc, batch), "DgemmStridedBatched")
