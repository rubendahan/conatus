"""CUDA backend: device arrays, transforms, batched bordered banded solves and IMEX on the GPU.

Everything stays on the device between calls. Host <-> device copies are explicit and only happen in
``to_device`` / ``DeviceArray.to_host`` (spec §57): no operation here copies implicitly.
``available()`` is False when the module was built without CUDA or no device is visible.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp

from ..ir.bases import Basis, Chebyshev, ComplexFourier, RealFourier
from ..reference.transforms import grid_size
from . import _core
from .banded import csr_args

__all__ = ["available", "device_name", "synchronize", "to_device", "forward", "backward",
           "BandedBatch", "Imex", "pool_stats", "pool_trim"]

_cu = getattr(_core, "cuda", None)


def available() -> bool:
    return _cu is not None and bool(_cu.available())


def unavailable_reason() -> str:
    """Why ``backend='cuda'`` cannot run here, naming the one case that applies (the messages said "no device, or a core built without CUDA" although the two are known apart);
    "" when it can."""
    if _cu is None:
        return ("the compiled core was built without CUDA (conatus._core has no CUDA part): "
                "rebuild it with CUDA (docs/site/install.md, GPU section)")
    if not _cu.available():
        return ("the core has CUDA but no GPU is visible to this process (a login or CPU node, "
                "or CUDA_VISIBLE_DEVICES empty)")
    return ""


def _need():
    if not available():
        from ..errors import BackendError
        raise BackendError(
            "CUDA backend not available: build with CONATUS_CUDA=ON and run on a GPU node",
            why=unavailable_reason(),
            fix="bash core/build.sh on a machine with nvcc (CONATUS_CUDA=ON), or "
                "pip install . -Ccmake.define.CONATUS_CUDA=ON; then run on a GPU node, or use "
                "the CPU backend")
    return _cu


def device_name() -> str:
    return _cu.device_name() if _cu is not None else "none (built without CUDA)"


def synchronize() -> None:
    _need().synchronize()


def pool_stats() -> dict:
    live, cached, n_alloc, n_reuse = _need().pool_stats()
    return {"live_bytes": live, "cached_bytes": cached, "n_alloc": n_alloc, "n_reuse": n_reuse}


def pool_trim() -> None:
    """Return the core pool's cached (freed) device blocks to the driver."""
    _need().pool_trim()


def _is_complex(dtype) -> bool:
    return np.issubdtype(np.dtype(dtype), np.complexfloating)


def to_device(a: np.ndarray):
    """Explicit host -> device copy of a 2D array (rows are transformed / solved independently)."""
    cu = _need()
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"to_device expects a 2D (batch, n) array, got shape {a.shape}")
    cls = cu.DeviceArrayZ if _is_complex(a.dtype) else cu.DeviceArrayD
    return cls.from_host(np.ascontiguousarray(a, dtype=np.complex128 if cls is cu.DeviceArrayZ else np.float64))


def _complex(d) -> bool:
    return type(d).__name__.endswith("Z")


def forward(d, b: Basis, dealias: bool = True):
    """Device rows of grid values (batch, M) -> device coefficients (batch, N), along the last axis."""
    cu = _need()
    M, N = grid_size(b, dealias), b.size
    if d.shape[1] != M:
        raise ValueError(f"expected {M} grid points per row, got {d.shape[1]}")
    if isinstance(b, Chebyshev):
        return cu.cheb_forward(d, N)
    if isinstance(b, RealFourier):
        if _complex(d):
            raise TypeError("RealFourier takes real data")
        return cu.realfourier_forward(d, N)
    if isinstance(b, ComplexFourier):
        if not _complex(d):
            raise TypeError("ComplexFourier takes complex data (convert on the host explicitly)")
        return cu.complexfourier_forward(d, N)
    raise NotImplementedError(f"CUDA transforms: {type(b).__name__} not supported yet")


def backward(d, b: Basis, dealias: bool = True):
    """Device coefficients (batch, N) -> device grid values (batch, M)."""
    cu = _need()
    M, N = grid_size(b, dealias), b.size
    if d.shape[1] != N:
        raise ValueError(f"expected {N} coefficients per row, got {d.shape[1]}")
    if isinstance(b, Chebyshev):
        return cu.cheb_backward(d, M)
    if isinstance(b, RealFourier):
        return cu.realfourier_backward(d, M)
    if isinstance(b, ComplexFourier):
        return cu.complexfourier_backward(d, M)
    raise NotImplementedError(f"CUDA transforms: {type(b).__name__} not supported yet")


class BandedBatch:
    """One bordered banded system per mode (same n and border rows), factored and solved on the GPU.

    ``pivot=False`` skips row interchanges: only valid if the caller knows the banded part is
    diagonally dominant (nothing is assumed silently; the default pivots).

    ``level_schedule`` (default): solves stream the LU factors without their zeros, one dependency
    level per warp step, same arithmetic per row as the band kernel. ``level_merge`` (default) also
    merges consecutive levels into steps of up to 32 rows, each row rewritten in terms of the
    values present at the start of its step: about 6x fewer steps on spectral operators, results
    equal to the unmerged ones to rounding (products of factor entries formed on the host).
    """

    def __init__(self, mats: Sequence, border_rows: Sequence[int], pivot: bool = True, dtype=None,
                 level_schedule: bool = True, level_merge: bool = True, col_border: bool = False):
        cu = _need()
        dtype = dtype or sp.csr_matrix(mats[0]).dtype
        self.dtype = np.complex128 if _is_complex(dtype) else np.float64
        cls = cu.BandedBatchZ if self.dtype == np.complex128 else cu.BandedBatchD
        self.border_rows = [int(i) for i in border_rows]
        self._s = cls([csr_args(A, self.dtype) for A in mats], self.border_rows, bool(pivot),
                      bool(level_schedule), bool(level_merge), bool(col_border))

    n = property(lambda self: self._s.n)
    kl = property(lambda self: self._s.kl)
    ku = property(lambda self: self._s.ku)
    rank = property(lambda self: self._s.rank)
    # column border (first r columns out of the band, exact; core/include/conatus/banded.hpp)
    col_border = property(lambda self: self._s.col_border)
    batch = property(lambda self: self._s.batch)
    factor_ms = property(lambda self: self._s.factor_ms)
    # level-scheduled solve (same factors, same arithmetic per row; see core/src/cuda/banded.cu)
    level_scheduled = property(lambda self: self._s.level_scheduled)
    level_depth = property(lambda self: (self._s.level_depth_forward, self._s.level_depth_backward))
    # warp steps actually executed (forward, backward): the depth, or fewer with merged levels
    level_steps = property(lambda self: (self._s.level_steps_forward, self._s.level_steps_backward))
    level_merged = property(lambda self: self._s.level_merged)
    level_stream_bytes = property(lambda self: self._s.level_stream_bytes)
    level_note = property(lambda self: self._s.level_note)

    # transposed level-scheduled solves and release of the band LU (a core built
    # before them has neither: reported as not available)
    level_transposed = property(lambda self: bool(getattr(self._s, "level_transposed", False)))
    band_released = property(lambda self: bool(getattr(self._s, "band_released", False)))
    level_t_stream_bytes = property(lambda self: float(getattr(self._s, "level_t_stream_bytes", 0.0)))
    level_t_note = property(lambda self: str(getattr(self._s, "level_t_note", "core without it")))

    def compact(self) -> bool:
        """Build the transposed level schedule, then release the band LU (the level streams hold
        the same factors without their zeros). False (nothing changed) when the core lacks it or
        a schedule does not apply (``level_note`` / ``level_t_note`` say why)."""
        if not hasattr(self._s, "build_transpose_schedule"):
            return False
        self._s.build_transpose_schedule()
        if not self._s.level_transposed:
            return False
        self._s.release_band()
        return True

    def solve_mapped(self, rhs, rows, x, cols) -> None:
        """Out-of-place solve with the gather/scatter fused in the kernel (CuPy arrays):
        b_m[i] = rhs[rows[m, i]], x[cols[m, i]] = x_m[i]; ``rows``/``cols`` int64 (batch, n)
        device arrays or None (contiguous (batch, n) layout)."""
        import cupy as cp
        # The kernel reads raw pointers: a wrong dtype, shape or a strided view would be read as
        # other memory (no error), so the arrays are checked here (the step loop calls the
        # pointer entry point directly with maps it built itself).
        for name, a in (("rhs", rhs), ("x", x)):
            if a.dtype != self.dtype or not a.flags.c_contiguous:
                raise ValueError(f"BandedBatch.solve_mapped: {name} must be a contiguous "
                                 f"{np.dtype(self.dtype).name} array (got {a.dtype}, "
                                 f"contiguous={a.flags.c_contiguous})")
        for name, a, full in (("rows", rows, rhs), ("cols", cols, x)):
            if a is None:
                if full.size < self.batch * self.n:
                    raise ValueError(f"BandedBatch.solve_mapped: {name}=None needs at least "
                                     f"batch * n = {self.batch * self.n} entries")
            elif a.dtype != np.int64 or a.shape != (self.batch, self.n) or not a.flags.c_contiguous:
                raise ValueError(f"BandedBatch.solve_mapped: {name} must be a contiguous int64 "
                                 f"({self.batch}, {self.n}) array (got {a.dtype} {a.shape})")
        p = lambda a: 0 if a is None else int(a.data.ptr)          # noqa: E731
        self._s.solve_mapped_ptr(p(rhs), p(rows), p(x), p(cols), int(cp.cuda.get_current_stream().ptr))

    def solve(self, d) -> None:
        """Solve in place; ``d`` is a (batch, n) DeviceArray of the right dtype."""
        self._s.solve(d)

    def solve_transpose(self, d) -> None:
        """Solve A_k^T x = b_k in place (plain transpose, no conjugation, same factors)."""
        self._s.solve_transpose(d)

    def prepare_transpose(self) -> None:
        """Build the transposed Woodbury data now (needed before capturing a CUDA graph)."""
        self._s.prepare_transpose()

    def explain(self) -> str:
        return (f"cuda.BandedBatch(batch={self.batch}, n={self.n}, common band kl={self.kl} ku={self.ku}, "
                f"border rank={self.rank}"
                + (f" (rows and columns, {self._s.col_border_nnz} column-border entries)" if self.col_border else "")
                + f", pivot={'partial (in band)' if self._s.pivot else 'none (declared)'}, "
                f"capacitance cond_1 max={self._s.capacitance_cond_max:.3g}, factor {self.factor_ms:.3g} ms); "
                "one warp per mode: banded LU + Woodbury/Schur on the boundary rows; solve: "
                + (f"level-scheduled sparse factors (depth {self.level_depth[0]} + {self.level_depth[1]}"
                   # a level wider than one warp chunk is never merged: it stays one step of more
                   # than 32 rows (it used to be reported as part of "steps of <= 32 rows")
                   + (f", merged into {self.level_steps[0]} + {self.level_steps[1]} warp steps "
                      "(a merged step holds <= 32 rows; a wider level stays a step of its own)"
                      if self.level_merged else ", one level per step")
                   + f", {self.level_stream_bytes / 2**20:.1f} MiB streamed)" if self.level_scheduled else
                   f"band kernel ({self.level_note})")
                + ("; transposed solves level-scheduled too "
                   f"({self.level_t_stream_bytes / 2**20:.1f} MiB)" if self.level_transposed else "")
                + ("; band LU released (level streams only)" if self.band_released else ""))


class Imex:
    """Batched IMEX integrator on the GPU: M_k dX/dt + L_k X = F for each mode k.

    F is zero or the pointwise model ``F = g * X + h`` (g, h: (batch, n) host arrays, copied once).
    Scheme and dt are required: nothing is chosen for the user.
    """

    def __init__(self, Ms: Sequence, Ls: Sequence, border_rows: Sequence[int], scheme: str, dt: float,
                 g=None, h=None, pivot: bool = True, dtype=None):
        cu = _need()
        dt_ = dtype or np.result_type(*[sp.csr_matrix(A).dtype for A in list(Ms) + list(Ls)])
        self.dtype = np.complex128 if _is_complex(dt_) else np.float64
        cls = cu.ImexZ if self.dtype == np.complex128 else cu.ImexD
        self.scheme, self.dt = scheme, float(dt)
        self._s = cls([csr_args(A, self.dtype) for A in Ms], [csr_args(A, self.dtype) for A in Ls],
                      [int(i) for i in border_rows], scheme, self.dt, bool(pivot))
        def conv(a):
            return None if a is None else np.ascontiguousarray(a, dtype=self.dtype).ravel()
        self._pointwise = g is not None or h is not None
        self._s.set_pointwise_rhs(conv(g), conv(h))

    n_factorizations = property(lambda self: self._s.n_factorizations)

    def run(self, d, t0: float, nsteps: int) -> float:
        """Advance the (batch, n) DeviceArray ``d`` in place; returns the final time (a float
        number of steps is refused, never truncated)."""
        from ..timestep.integrator import check_steps
        return self._s.run(d, float(t0), check_steps(nsteps, "cuda.Imex.run"))

    def explain(self) -> str:
        return (f"cuda.Imex(scheme={self.scheme}, dt={self.dt:g}, constant step, "
                f"F={'g*X+h' if self._pointwise else 'none'}, factorizations={self.n_factorizations}); "
                "state resident on the device, batched bordered banded solves")
