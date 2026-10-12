"""Liveness memory planner of an IVP time step (spec §56–§59).

From the *compiled* objects of an ``IVPSolver`` (stacked system, buckets, fused RHS, scheme) — no
GPU and no step needed — the planner writes the **program of device allocations** that the CUDA
backend performs (setup, ``set_state``, then time steps), buffer by buffer, in program order:

* persistent: CSR operators, bucket/gather/scatter maps, state, fused-RHS constants, cuFFT plan
  work areas (plan cache), banded factors (C++ core pool, one set per (a, b) key);
* per step: RK stage states, F(X) vectors, multistep history, IMEX temporaries (axpy results,
  copies of the RK combinations), F workspace (scatter arrays, derivative memo, stacked factor
  coefficients and grids, transform temporaries, pointwise outputs, gather temporaries), solve
  workspace (``rhs[rows]`` copies, result vector).

Every size mirrors the code that allocates it (``timestep/device.py``, ``timestep/fused.py``,
``timestep/xp_transforms.py``, ``timestep/integrator.py``, CuPy's ``cupy/fft/_fft.py`` and
``core/src/cuda/banded.cu``); a temporary dies where CPython drops its last reference.

From that program:

* **liveness** — each buffer's interval [alloc, free]; peak of live bytes and its breakdown by
  category (state, RHS, history, integrator, F coefficients / grids, FFT, solve, operators, ...);
* **pool prediction** — the program is replayed through a model of CuPy's memory pool (512-byte
  rounding, best-fit bins, split and merge of chunks) and of the core pool (exact-size cache), which
  gives the bytes the process *holds* (what an out-of-memory error depends on);
* **reuse proposal** — a static offset assignment of the step's transient buffers (greedy by size
  on the interval graph, as in static memory planners): arena size and which buffer reuses whose
  memory. It is a *proposal*: the runtime does not apply it yet (the integrator is not modified).

Declared limits (errors, never silent approximations): only ``f_backend='fused'`` without parity
reduction, the CUDA backend, and SBDF start-up ``'lower_order'`` are modelled. cuFFT work areas are
asked to cuFFT itself when a GPU is present (``cufftMakePlan1d`` without allocation) and otherwise
*estimated* (flag in the report). The CUDA context, kernel images and library handles are not
modelled buffer by buffer: they are counted as one declared constant, the process floor
``CUDA_PROCESS_FLOOR_BYTES`` (measured, see below), shown on its own line next to the totals.

Known runtime behaviour ("memo reference cycle"): the derivative memo of
``FusedRHS.evaluate`` is a recursive closure, i.e. a reference cycle; its arrays (and the scatter
arrays they view) are freed only when CPython's cyclic collector runs. The planner models the
deterministic program (freed at return, as after the one-line fix) and reports, separately, the
bytes each uncollected evaluation adds (``gc_pending_per_F``).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

__all__ = ["MemoryReport", "memory_report", "plan_memory", "CATEGORIES", "simulate_cupy_pool",
           "StepProgram", "arena_offsets"]

MiB = 2 ** 20

# Process floor of a CUDA run: device memory a process holds beyond the
# two allocators (CUDA context, kernel images of the core and of CuPy, cuFFT/cuBLAS handles and
# plan caches), which no buffer model predicts. Measured as nvidia-smi's per-process figure minus
# (CuPy pool held + core pool live + cached), after the time steps, 24 cases per GPU (RB 2D
# 256x64..1024x256 and channel 3D 32^3..96^3, RK443 and SBDF2): A100 439-496 MiB
# (run), L40S 449-504 MiB (run); it grows
# slightly with the problem (cuFFT plans). Declared conservative value: 512 MiB, the largest
# measured floor rounded up. CUDA 12.9 / CuPy 13-14 / driver 580: another stack may differ.
CUDA_PROCESS_FLOOR_BYTES = 512 * MiB
CUDA_PROCESS_FLOOR_SOURCE = ("measured: nvidia-smi per-process minus allocator-held bytes, A100 "
                             "439-496 MiB, L40S 449-504 MiB (24 cases each); declared 512 MiB")

CATEGORIES = (
    "state", "RK stages", "RHS F(X)", "multistep history", "integrator temporaries",
    "F coefficients", "F grids", "FFT temporaries", "cuFFT work areas", "solve workspace",
    "operators (CSR + maps)", "constants", "banded factors (core pool)",
    "F workspace (persistent)")


# ====================================================================== program
class StepProgram:
    """Ordered device allocations and frees (bytes as requested, before pool rounding).

    ``events``: (kind, buffer, phase, step) with kind 'a' (alloc) or 'f' (free); buffers carry a
    name, a category and a pool ('cupy' or 'core')."""

    def __init__(self) -> None:
        self.size: list[int] = []
        self.name: list[str] = []
        self.cat: list[str] = []
        self.pool: list[str] = []
        self.events: list[tuple[str, int, str, int]] = []
        self.phase, self.step = "setup", -1

    def alloc(self, nbytes: int, cat: str, name: str, pool: str = "cupy") -> int:
        """Record an allocation (size, category, pool) and return its buffer id."""
        b = len(self.size)
        self.size.append(int(nbytes))
        self.cat.append(cat)
        self.name.append(name)
        self.pool.append(pool)
        self.events.append(("a", b, self.phase, self.step))
        return b

    def free(self, *bufs) -> None:
        """Record the release of one or more buffers (None ignored)."""
        for b in bufs:
            if b is not None:
                self.events.append(("f", b, self.phase, self.step))

    def retag(self, b: int, cat: str) -> None:
        """Change the category of buffer ``b`` (e.g. scratch that becomes state)."""
        self.cat[b] = cat

    # ------------------------------------------------------------------ liveness
    def intervals(self) -> dict[int, tuple[int, int | None]]:
        """{buffer: (alloc event index, free event index or None if alive at the end)}."""
        iv: dict = {}
        for t, (k, b, _p, _s) in enumerate(self.events):
            if k == "a":
                iv[b] = (t, None)
            else:
                iv[b] = (iv[b][0], t)
        return iv

    def live_peak(self, pool: str = "cupy", from_step: int | None = None) -> tuple[int, int, dict]:
        """(peak live bytes, event index of the peak, bytes by category at the peak)."""
        live, peak, at, alive = 0, -1, 0, set()
        best: set = set()
        for t, (k, b, _p, s) in enumerate(self.events):
            if self.pool[b] != pool:
                continue
            if k == "a":
                live += self.size[b]
                alive.add(b)
            else:
                live -= self.size[b]
                alive.discard(b)
            if (from_step is None or s >= from_step) and live > peak:
                peak, at, best = live, t, set(alive)
        cats: dict = {}
        for b in best:
            cats[self.cat[b]] = cats.get(self.cat[b], 0) + self.size[b]
        return max(peak, 0), at, cats


# ====================================================================== pool models
class _Chunk:
    """A piece of one device allocation of the CuPy pool (neighbours linked, compared by identity)."""
    __slots__ = ("size", "free", "prev", "next")

    def __init__(self, size: int, free: bool, prev=None, nxt=None) -> None:
        self.size, self.free, self.prev, self.next = size, free, prev, nxt


class _CupyPool:
    """Model of CuPy's ``SingleDeviceMemoryPool`` (one stream): sizes rounded up to 512 bytes; a
    request takes a free chunk from the smallest non-empty size bin that fits (split if larger;
    within a bin the oldest free chunk first), otherwise a new device allocation of exactly the
    rounded size; a freed chunk merges with its free neighbours inside the same allocation.
    Replaying 12 measured traces (RB 2D, canal 3D) reproduces the held bytes exactly; taking the
    newest chunk instead missed two of them by 8.8 % (run)."""
    UNIT = 512

    def __init__(self) -> None:
        self.held = 0
        self.bins: dict[int, dict] = {}     # size -> {id: chunk}, insertion ordered (oldest first)
        self.sizes: list[int] = []          # sorted sizes of non-empty bins

    def _push(self, ch: _Chunk) -> None:
        lst = self.bins.get(ch.size)
        if lst is None:
            lst = self.bins[ch.size] = {}
            bisect.insort(self.sizes, ch.size)
        lst[id(ch)] = ch

    def _remove(self, ch: _Chunk) -> None:
        lst = self.bins[ch.size]
        del lst[id(ch)]
        if not lst:
            del self.bins[ch.size]
            self.sizes.pop(bisect.bisect_left(self.sizes, ch.size))

    def malloc(self, size: int) -> _Chunk | None:
        """A used chunk of the rounded size; None for a zero-size request."""
        size = -(-int(size) // self.UNIT) * self.UNIT
        if size == 0:
            return None
        i = bisect.bisect_left(self.sizes, size)
        if i < len(self.sizes):
            lst = self.bins[self.sizes[i]]
            ch = lst[next(iter(lst))]
            self._remove(ch)
            if ch.size > size:                        # split: head used, tail stays free
                tail = _Chunk(ch.size - size, True, ch, ch.next)
                if ch.next is not None:
                    ch.next.prev = tail
                ch.next, ch.size = tail, size
                self._push(tail)
            ch.free = False
            return ch
        self.held += size
        return _Chunk(size, False)

    def free(self, ch: _Chunk | None) -> None:
        if ch is None:
            return
        ch.free = True
        nxt = ch.next
        if nxt is not None and nxt.free:              # merge with the next free chunk
            self._remove(nxt)
            ch.size += nxt.size
            ch.next = nxt.next
            if nxt.next is not None:
                nxt.next.prev = ch
        prv = ch.prev
        if prv is not None and prv.free:              # merge into the previous free chunk
            self._remove(prv)
            prv.size += ch.size
            prv.next = ch.next
            if ch.next is not None:
                ch.next.prev = prv
            ch = prv
        self._push(ch)


class _CorePool:
    """Model of the C++ core pool (``core/src/cuda/memory.cu``): 64-byte rounding, freed blocks
    cached by exact size and reused only for the same size, never returned to the driver."""

    def __init__(self, trim: bool = False) -> None:
        self.live = self.cached = 0
        self.free_blocks: dict[int, int] = {}
        # trim: freed blocks go back to the driver at once (DeviceSystem.compact_factors calls
        # pool_trim after each released band); peak: high-water mark of live + cached bytes
        self.trim, self.peak = bool(trim), 0

    def malloc(self, size: int) -> int:
        size = -(-int(size) // 64) * 64
        if size == 0:
            return 0
        if self.free_blocks.get(size, 0):
            self.free_blocks[size] -= 1
            self.cached -= size
        self.live += size
        self.peak = max(self.peak, self.live + self.cached)
        return size

    def free(self, size: int) -> None:
        if size:
            self.live -= size
            if self.trim:
                return
            self.free_blocks[size] = self.free_blocks.get(size, 0) + 1
            self.cached += size


def simulate_cupy_pool(events, sizes=None) -> int:
    """Bytes held by the CuPy pool after replaying ``events``. Either a StepProgram's events with
    ``sizes`` (list per buffer), or measured events (kind 'm'/'f', bytes, ptr, ...) with
    ``sizes=None`` (used to check the pool model itself against a measured trace)."""
    pool = _CupyPool()
    chunks: dict = {}
    for e in events:
        if sizes is None:
            kind, nbytes, key = e[0], e[1], e[2]
            if kind == "m":
                chunks[key] = pool.malloc(nbytes)
            elif kind == "f":
                pool.free(chunks.pop(key, None))
        else:
            kind, b = e[0], e[1]
            if kind == "a":
                chunks[b] = pool.malloc(sizes[b])
            else:
                pool.free(chunks.pop(b, None))
    return pool.held


# ====================================================================== cuFFT work areas
_CUFFT_TYPES = {("C2C", False): 0x69, ("R2C", False): 0x6a, ("C2R", False): 0x6c}   # Z2Z, D2Z, Z2D


class _CufftSizer:
    """Work-area bytes of a cuFFT 1D plan (n, type, batch). With a GPU: asked to cuFFT
    (``cufftCreate`` + ``cufftSetAutoAllocation(0)`` + ``cufftMakePlan1d``: nothing allocated).
    Without: an *estimate* (0 for lengths handled in one pass, else one complex copy of the
    batch), flagged in the report."""

    def __init__(self, query: bool) -> None:
        self.lib = None
        self.source = "estimated"
        if query:
            try:
                import ctypes
                import ctypes.util
                import os
                import sys
                cands = [os.path.join(sys.prefix, "lib", "libcufft.so"),
                         ctypes.util.find_library("cufft") or "libcufft.so"]
                for c in cands:
                    try:
                        self.lib = ctypes.CDLL(c)
                        break
                    except OSError:
                        continue
                if self.lib is not None:
                    self.source = "cufft"
            except Exception:           # noqa: BLE001 -- no cuFFT: the estimate is flagged
                self.lib = None

    def __call__(self, n: int, vtype: str, batch: int) -> int:
        if self.lib is not None:
            import ctypes
            h = ctypes.c_int(0)
            ws = ctypes.c_size_t(0)
            lib = self.lib
            if lib.cufftCreate(ctypes.byref(h)) == 0:
                try:
                    lib.cufftSetAutoAllocation(h, 0)
                    st = lib.cufftMakePlan1d(h, int(n), _CUFFT_TYPES[(vtype, False)], int(batch),
                                             ctypes.byref(ws))
                    if st == 0:
                        return int(ws.value)
                finally:
                    lib.cufftDestroy(h)
            self.source = "estimated"                  # query failed: fall back, flagged
        half = n // 2 if vtype != "C2C" else n
        one_pass = n <= 1024 and (vtype == "C2C" or half % 2 == 0)
        return 0 if one_pass else int(batch) * int(half) * 16


# ====================================================================== array model
@dataclass
class _A:
    """A modelled device array: buffer (None for views of other buffers), shape, complex,
    C-contiguous and 'fresh' (owns its memory: ``base is None``)."""
    buf: int | None
    shape: tuple
    cplx: bool
    contig: bool = True
    fresh: bool = True

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    def nbytes(self) -> int:
        return self.size * (16 if self.cplx else 8)


def _new(P: StepProgram, shape, cplx: bool, cat: str, name: str) -> _A:
    shape = tuple(int(s) for s in shape)
    a = _A(None, shape, cplx)
    a.buf = P.alloc(a.nbytes(), cat, name)
    return a


def _with(shape, axis, n) -> tuple:
    s = list(shape)
    s[axis] = n
    return tuple(s)


class _XTModel:
    """Allocation program of ``timestep/xp_transforms.py`` on CuPy (``cupy/fft/_fft.py``: zero
    padding, input copies, plan cache with work areas, outputs). Also records every cuFFT
    execution (length, type, batch) for the flop estimates of :mod:`conatus.runtime.profile`."""

    def __init__(self, P: StepProgram, cufft: _CufftSizer, plan_cache_size: int = 16) -> None:
        from ..ir.bases import Chebyshev, ComplexFourier, RealFourier
        self.Cheb, self.CF, self.RF = Chebyshev, ComplexFourier, RealFourier
        self.P = P
        self.cufft = cufft
        self.plan_cache: list = []               # LRU keys, most recent last
        self.plan_cache_size = int(plan_cache_size)
        self.plan_bufs: dict = {}
        self.half: set = set()
        self.ffts: list = []                     # (n, vtype, batch) of each execution

    # ------------------------------------------------------------------ FFT (CuPy _fft.py)
    def _plan(self, n: int, vtype: str, batch: int) -> None:
        key = (n, vtype, batch)
        if key in self.plan_cache:
            self.plan_cache.remove(key)
            self.plan_cache.append(key)
            return
        ws = self.cufft(n, vtype, batch)
        self.plan_bufs[key] = self.P.alloc(ws, "cuFFT work areas", f"cufft {vtype} n={n} batch={batch}")
        self.plan_cache.append(key)
        if len(self.plan_cache) > self.plan_cache_size:          # LRU eviction frees the area
            old = self.plan_cache.pop(0)
            self.P.free(self.plan_bufs.pop(old))

    def _exec_fft(self, a: _A, axis: int, vtype: str, out_size: int | None = None) -> _A:
        P = self.P
        last = axis == len(a.shape) - 1
        copy = (not last) or (not a.fresh) or (not a.contig) or vtype == "C2R"
        c = _new(P, a.shape, a.cplx, "FFT temporaries", f"fft input copy {vtype}") if copy else None
        n = a.shape[axis]
        n_out = out_size if out_size is not None else n
        batch = a.size // n
        self._plan(n_out, vtype, batch)
        self.ffts.append((n_out, vtype, batch))
        if vtype == "C2C":
            shp, oc = a.shape, True
        elif vtype == "R2C":
            shp, oc = _with(a.shape, axis, n // 2 + 1), True
        else:
            shp, oc = _with(a.shape, axis, n_out), False
        out = _new(P, shp, oc, "FFT temporaries", f"fft out {vtype}")
        if c is not None:
            P.free(c.buf)
        out.fresh = out.contig = last
        return out

    def _fft(self, a: _A, axis: int, vtype: str, n: int | None = None) -> _A:
        """cupy.fft.{fft, ifft, rfft, irfft} along ``axis`` (``_cook_shape`` then ``_exec_fft``)."""
        z = None
        if n is not None:
            target = n // 2 + 1 if vtype == "C2R" else n
            if a.shape[axis] < target:                  # zero-padded copy
                z = _new(self.P, _with(a.shape, axis, target), a.cplx, "FFT temporaries", "fft zero pad")
                a = z
            elif a.shape[axis] > target:
                a = _A(None, _with(a.shape, axis, target), a.cplx, False, False)
        out = self._exec_fft(a, axis, vtype, out_size=n if vtype == "C2R" else None)
        if z is not None:
            self.P.free(z.buf)
        return out

    def _ew(self, shape, cplx: bool, name: str, cat: str = "FFT temporaries") -> _A:
        """Result of an elementwise CuPy operation (fresh, C-contiguous)."""
        return _new(self.P, shape, cplx, cat, name)

    # ------------------------------------------------------------------ xp_transforms
    def backward(self, c: _A, b, axis: int) -> _A:
        P = self.P
        M, N = b.grid_size, b.size
        if isinstance(b, self.Cheb):
            if c.cplx:
                fr = self.backward(_A(None, c.shape, False, False, False), b, axis)
                fi = self.backward(_A(None, c.shape, False, False, False), b, axis)
                t = self._ew(fi.shape, True, "1j * imag")
                P.free(fi.buf)
                r = self._ew(fi.shape, True, "real + 1j imag")
                P.free(fr.buf, t.buf)
                return r
            if N not in self.half:
                self.half.add(N)
                P.alloc(N * 8, "constants", f"chebyshev half {N}")
            tw = self._ew((N,), True, "twiddle * half")
            a = self._ew(c.shape, True, "twiddle * c")
            P.free(tw.buf)
            out = self._fft(a, axis, "C2R", n=2 * M)
            f = self._ew(_with(c.shape, axis, M), False, "irfft[:M] * 2M")
            P.free(out.buf, a.buf)
            return f
        if isinstance(b, self.RF):
            K = N // 2
            F = self._ew(_with(c.shape, axis, M // 2 + 1), True, "rfft spectrum (zeros)")
            t1 = self._ew(_with(c.shape, axis, K - 1), True, "1j * sin")
            t2 = self._ew(t1.shape, True, "cos + 1j sin")
            P.free(t1.buf)
            t3 = self._ew(t1.shape, True, "/2")
            P.free(t2.buf, t3.buf)
            t4 = self._ew(F.shape, True, "F * M")
            out = self._fft(t4, axis, "C2R", n=M)
            P.free(t4.buf, F.buf)
            return out
        if isinstance(b, self.CF):
            F = self._ew(_with(c.shape, axis, M), True, "spectrum (zeros)")
            t = self._ew(F.shape, True, "F * M")
            out = self._fft(t, axis, "C2C")
            P.free(t.buf, F.buf)
            return out
        raise NotImplementedError(type(b).__name__)

    def forward(self, f: _A, b, axis: int) -> _A:
        P = self.P
        M, N = b.grid_size, b.size
        if isinstance(b, self.Cheb):
            if f.cplx:
                fr = self.forward(_A(None, f.shape, False, False, False), b, axis)
                fi = self.forward(_A(None, f.shape, False, False, False), b, axis)
                t = self._ew(fi.shape, True, "1j * imag")
                P.free(fi.buf)
                r = self._ew(fi.shape, True, "real + 1j imag")
                P.free(fr.buf, t.buf)
                return r
            out = self._fft(f, axis, "R2C", n=2 * M)
            t1 = self._ew(_with(f.shape, axis, N), True, "twiddle * F")
            c = self._ew(t1.shape, False, "(.).real * 2/M")
            P.free(t1.buf, out.buf)
            return c
        if isinstance(b, self.RF):
            K = N // 2
            out = self._fft(f, axis, "R2C")
            Fd = self._ew(out.shape, True, "rfft / M")
            P.free(out.buf)
            c = self._ew(_with(f.shape, axis, N), f.cplx, "coefficients (zeros)")
            for nm in ("2 Re", "2 Im"):
                t = self._ew(_with(f.shape, axis, K - 1), False, nm)
                P.free(t.buf)
            P.free(Fd.buf)
            return c
        if isinstance(b, self.CF):
            out = self._fft(f, axis, "C2C")
            Fd = self._ew(out.shape, True, "fft / M")
            P.free(out.buf)
            c = self._ew(_with(f.shape, axis, N), True, "coefficients (zeros)")
            P.free(Fd.buf)
            return c
        raise NotImplementedError(type(b).__name__)


class _Model(_XTModel):
    """Writes the allocation program of an ``IVPSolver(backend='cuda', f_backend='fused')``."""

    def __init__(self, host, fused, scheme: str, rhs_exprs, *, startup: str = "lower_order",
                 cufft_query: bool = False, plan_cache_size: int = 16, fc2d: bool | None = None,
                 set_state_fields: tuple = ()) -> None:
        if fused is None:
            raise NotImplementedError("memory planner: only f_backend='fused' is modelled "
                                      "(the evaluator's temporaries are not written down)")
        if getattr(fused, "mode", "polynomial") != "polynomial":
            raise NotImplementedError("memory planner: the general-form fused RHS (derivatives of "
                                      "products) is not modelled yet; polynomial form only")
        if getattr(fused, "reflection", None) is not None:
            raise NotImplementedError("memory planner: the parity-reduced fused RHS is not modelled yet")
        from ..timestep.integrator import scheme_family
        if scheme_family(scheme) != "rk" and startup != "lower_order":
            raise NotImplementedError("memory planner: multistep start-up 'rk443' is not modelled "
                                      "(model startup='lower_order')")
        self.host, self.fu, self.scheme = host, fused, scheme
        self.rhs_exprs = list(rhs_exprs)
        super().__init__(StepProgram(), _CufftSizer(cufft_query), plan_cache_size)
        self.n = int(host.n)
        self.cplx = np.dtype(host.dtype).kind == "c"
        self.it = 16 if self.cplx else 8
        # DeviceSystem.compact_factors: band LU released after the transposed level schedule
        # (Wt/St prepared at factor time); both level streams are not modelled (declared)
        self.compact_factors = bool(getattr(host, "_plan_compact_factors", False))
        # level streams of the factors (estimated from sampled modes, runtime/level_stream.py):
        # DeviceSystem.level_merge, number of sampled modes per batch (0: not counted)
        self.level_merge = bool(getattr(host, "_plan_level_merge", True))
        self.stream_samples = int(getattr(host, "_plan_stream_samples", 3))
        # the factor keys of the model have no dt: the actual (a, b) of a key needs it (the fill
        # of the LU, hence the stream, depends on b / a); without dt the streams are not counted
        self.stream_dt = getattr(host, "_plan_dt", None)
        self.streams_estimated = False
        self.core_bufs: dict = {}                # (a, b) key -> list of core buffers
        self._shared_flags: dict = {}            # key -> per complex-mode bucket: shares factors
        self.conv: dict = {}
        self.twiddles: set = set()
        self.X = None
        self.hist_MX: list = []
        self.hist_F: list = []
        self.hist_LX: list = []                  # CN multistep family only
        self.gc_pending_per_F = 0
        # 2D RealFourier x Chebyshev on the GPU: IVPSolver evaluates F with the FourierChebyshev2D
        # pipeline (persistent work buffers and cuFFT plans built once, one output per call), not
        # with the generic fused F modelled below. The planner modelled the generic F only, so its
        # RB 2D figures were of another program since the-512 merge (gpu4). ``fc2d``: the
        # solver's actual choice (memory_report), else the solver's own rule (plan_memory).
        if fc2d is None:
            from ..timestep.fc2d import FourierChebyshev2DRHS
            fc2d = FourierChebyshev2DRHS.reason(fused, host, assume_gpu=True) is None
        self.fc2d = bool(fc2d)
        # variables given as grid data to set_state (a grid array and its coefficients each,
        # transient); () models a state given in coefficients
        self.set_state_fields = tuple(set_state_fields)
        self.rk_fused = None          # None: the runtime's rule (real systems fused); tests force it

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        P, host, it = self.P, self.host, self.it
        P.phase = "setup"
        for nm, A in (("M", host.M), ("L", host.L)):
            A = A.tocsr()
            P.alloc((A.shape[0] + 1) * 4, "operators (CSR + maps)", f"{nm}.indptr")
            P.alloc(A.nnz * 4, "operators (CSR + maps)", f"{nm}.indices")
            P.alloc(A.nnz * it, "operators (CSR + maps)", f"{nm}.data")
        for bi, (_idx, rows, cols) in enumerate(host._buckets):
            P.alloc(rows.size * 8, "operators (CSR + maps)", f"bucket{bi}.rows")
            P.alloc(cols.size * 8, "operators (CSR + maps)", f"bucket{bi}.cols")
        for bi, (_idx, *arrs) in enumerate(host._cbuckets):
            for a in arrs:
                P.alloc(a.size * 8, "operators (CSR + maps)", f"cbucket{bi}.map")
        # index maps: int32 on the device when every index fits (timestep/device.py _index_map)
        isz = 4 if self.n < 2 ** 31 - 1 else 8
        for bi, (_keys, rows, cols, _wr, _wc) in enumerate(getattr(host, "_zbuckets", [])):
            P.alloc(np.asarray(rows).size * isz, "operators (CSR + maps)", f"zbucket{bi}.rows")
            P.alloc(np.asarray(cols).size * isz, "operators (CSR + maps)", f"zbucket{bi}.cols")
        for k, sel in enumerate(host._col_sel):
            P.alloc(sel.size * isz, "operators (CSR + maps)", f"col_sel{k}")
            P.alloc(sel.size * isz, "operators (CSR + maps)", f"cols_flat{k}")
        for key, sel in host._row_sel.items():
            P.alloc(sel.size * isz, "operators (CSR + maps)", f"row_sel{key}")
            P.alloc(sel.size * isz, "operators (CSR + maps)", f"rows_flat{key}")
        self.X = P.alloc(self.n * it, "state", "X (zeros)")
        for f, g in host.plan._ncc_data.items():          # NCC coefficients on the device
            P.alloc(int(np.asarray(g).size) * 8, "constants", f"ncc {f.name}")
        fu = self.fu
        for _ax, b in enumerate(fu.bases):                  # Fourier derivative multipliers
            if isinstance(b, self.CF):
                P.alloc(b.size * 16, "constants", f"i k ({b.coord.name})")
            elif isinstance(b, self.RF):
                P.alloc(b.size * 8, "constants", f"k ({b.coord.name})")
        for b in fu.bases:                                  # xp_transforms twiddles (cached)
            for sg in (1.0, -1.0):
                key = (b.grid_size, b.size, sg)
                if key not in self.twiddles:
                    self.twiddles.add(key)
                    P.alloc(b.size * 16, "constants", f"twiddle {key}")

        if self.fc2d:
            self._setup_fc2d()

    def _setup_fc2d(self) -> None:
        """Allocations of ``FourierChebyshev2DRHS.__init__`` (timestep/fc2d.py), in its order."""
        from ..timestep.fc2d import host_maps
        P, fu, host = self.P, self.fu, self.host
        bx, bz = fu.bases
        Nx, Nz, Mx, Mz = bx.size, bz.size, bx.grid_size, bz.grid_size
        Kx1 = Mx // 2 + 1
        flen = Mz                                          # chebyshev='dct' (the default)
        zlen = flen // 2 + 1
        nf, no = len(fu.factor_list), len(fu.outputs)
        mp = host_maps(fu, host)
        # IVPSolver._build_fc2d: the constant right-hand sides (boundary values) evaluated once on
        # the device (a scalar, then embedded on the equation's shape), copied to the host
        emb = []
        for k, (e, shape_k) in enumerate(zip(self.rhs_exprs, host.eq_shapes, strict=False)):
            if e is not None and not e.free_fields():
                small = P.alloc(8, "F coefficients", "constant value")
                emb.append(P.alloc(int(np.prod(shape_k)) * 8, "F coefficients", f"embedded constant eq{k}"))
                P.free(small)
        P.free(*emb)
        for key in ("gmap", "fvar_used", "xvar", "xpos", "fsteps", "fnsteps", "kx", "rowmap", "rowsrc",
                    "lamid", "cptr", "cind", "cval"):
            P.alloc(np.asarray(mp[key]).nbytes, "operators (CSR + maps)", f"fc2d {key}")
        P.alloc(zlen * 16, "constants", "fc2d twiddle backward")
        P.alloc(Nz * 16, "constants", "fc2d twiddle forward")
        P.alloc(self.n * 8, "constants", "fc2d constant RHS b0")
        for shape, it, name in (((mp["nused"], Nz, Nx), 8, "V"), ((nf, Nz, Nx), 8, "Cf"),
                                ((nf, Nz, Nx), 8, "Cs"), ((nf * Nx, zlen), 16, "Zin"),
                                ((nf * Nx, flen), 8, "R1"), ((nf * Mz, Kx1), 16, "Xin"),
                                ((nf, Mz, Mx), 8, "G"), ((no, Mz, Mx), 8, "O"),
                                ((no * Mz, Kx1), 16, "S"), ((no * Nx, flen), 8, "R2"),
                                ((no * Nx, zlen), 16, "Q")):
            # persistent work buffers of the pipeline (built once, reused by every F)
            P.alloc(int(np.prod(shape)) * it, "F workspace (persistent)", f"fc2d {name}")
        for n, vtype, batch in ((flen, "C2R", nf * Nx), (Mx, "C2R", nf * Mz), (Mx, "R2C", no * Mz),
                                (flen, "R2C", no * Nx)):
            P.alloc(self.cufft(n, vtype, batch), "cuFFT work areas", f"fc2d cufft {vtype} n={n} batch={batch}")

    def set_state(self) -> None:
        P = self.P
        P.phase = "set_state"
        host = self.host
        coeffs = []
        for v, shape in zip(host.variables, host.var_shapes, strict=False):
            if v.name in self.set_state_fields:
                nb = int(np.prod(shape)) * self.it
                g = P.alloc(nb, "state", f"grid data {v.name}")
                coeffs.append(P.alloc(nb, "state", f"coefficients {v.name}"))
                P.free(g)
        new = P.alloc(self.n * self.it, "state", "X")
        P.free(self.X)
        P.free(*coeffs)
        self.X = new

    # ------------------------------------------------------------------ F (fused)
    def _scatter(self) -> dict:
        P, host = self.P, self.host
        rhs_vars = {f.name for e in self.rhs_exprs if e is not None for f in e.free_fields()}
        data = {}
        for k, (v, shape) in enumerate(zip(host.variables, host.var_shapes, strict=False)):
            if v.name not in rhs_vars:
                continue
            a = _new(P, (int(np.prod(shape)),), self.cplx, "F coefficients", f"scatter {v.name}")
            t = P.alloc(host._col_sel[k].size * self.it, "F coefficients", "X[sel]")
            P.free(t)
            a.shape = tuple(shape)
            data[v.name] = a
        return data

    def _diff(self, c: _A, ax: int, b) -> _A:
        P = self.P
        if isinstance(b, self.Cheb):
            cm = None
            if not (ax == 0 and c.contig):
                cm = _new(P, c.shape, c.cplx, "F coefficients", "moveaxis copy")
            out = _new(P, c.shape, c.cplx, "F coefficients", f"d/d{b.coord.name}")
            P.free(cm.buf if cm is not None else None)
            out.contig = ax == 0
            out.fresh = False
            return out
        if isinstance(b, self.RF):
            sw = _new(P, c.shape, c.cplx, "F coefficients", "cos/sin swap")
            neg = P.alloc(c.nbytes() // 2, "F coefficients", "-sin")
            P.free(neg)
            out = _new(P, c.shape, c.cplx, "F coefficients", f"d/d{b.coord.name}")
            P.free(sw.buf)
            return out
        return _new(P, c.shape, True, "F coefficients", f"d/d{b.coord.name}")

    def F(self) -> int:
        """Program of ``IVPSolver.F`` (fused); returns the buffer of the stacked RHS vector."""
        P, fu, host = self.P, self.fu, self.host
        P.phase = "F"
        if self.fc2d:                                      # the 2D pipeline: one new output array
            self.gc_pending_per_F = 0
            return P.alloc(self.n * 8, "RHS F(X)", "b (2D pipeline output)")
        data = self._scatter()
        P.phase = "F.fused"
        shape = tuple(b.size for b in fu.bases)
        nf, no = len(fu.factor_list), len(fu.outputs)
        cdt = bool(fu.cplx or any(isinstance(b, self.CF) for b in fu.bases))
        memo: dict = {}
        # low-dimensional factors (FusedRHS._classify_low): their own small grids, the stack holds
        # the full factors only
        low = fu.low if getattr(fu, "_low_on", False) else None
        lowbuf = None
        if low is not None:
            gsz = 16 if cdt else 8
            lowbuf = P.alloc(int(fu.low_size) * gsz, "F grids", "low-dimensional factor grids")
        nG = max(1, sum(x is None for x in low)) if low is not None else nf
        G = _new(P, (nG,) + shape, cdt, "F coefficients", "G (stacked factor coefficients)")

        def deriv(f, comp, a):
            key = (f.name, comp, a)
            r = memo.get(key)
            if r is None:
                src = data.get(f.name)
                if src is None:
                    # not a state variable: an NCC profile or a StageTimeFused time coefficient
                    # (constant field). Their coefficients and derivatives are lower-dimensional:
                    # modelled as 8-byte arrays, i.e. not counted (declared)
                    r = _A(None, (), False, True, False) if not any(a) else _new(
                        P, (), False, "F coefficients", f"d {f.name} (small, not counted)")
                elif not any(a):
                    if tuple(src.shape[len(comp):]) != shape:
                        r = _new(P, shape, src.cplx, "F coefficients", "embedded factor")
                    else:
                        r = _A(None, shape, src.cplx, True, False)
                else:
                    ax = max(i for i, n in enumerate(a) if n)
                    prev = deriv(f, comp, tuple(n - (i == ax) for i, n in enumerate(a)))
                    r = self._diff(prev, ax, fu.bases[ax])
                memo[key] = r
            return r

        for (f, comp, a) in fu.factor_list:
            deriv(f, comp, a)
        G = self._batched(G, lambda g: self._chain(g, [(self.backward, ax) for ax in fu._backward_order]),
                          "F grids", "batched backward output")
        if not (G.contig and G.fresh):
            Gc = _new(P, G.shape, G.cplx, "F grids", "ascontiguousarray(G)")
            P.free(G.buf)
            G = Gc
        P.retag(G.buf, "F grids")
        gshape = G.shape[1:]
        out_c = bool(fu.cplx) or G.cplx
        Gk = None
        if fu.cplx and not G.cplx:
            Gk = _new(P, G.shape, True, "F grids", "G as complex")
        O = _new(P, (no,) + gshape, out_c, "F grids", "O (pointwise outputs)")
        P.free(Gk.buf if Gk is not None else None)
        O = self._batched(O, lambda o: self._chain(o, [(self.forward, ax) for ax in fu._backward_order[::-1]]),
                          "F coefficients", "batched forward output")
        res: dict = {}
        for k, _comp, _m in fu.outputs:
            if k not in res:
                eq = fu.equations[k]
                cs = tuple(c.dim for c in eq.lhs.tensorsig)
                res[k] = _new(P, cs + shape, O.cplx, "F coefficients", f"res eq{k}")
        memo_bytes = sum(P.size[r.buf] for r in memo.values() if r.buf is not None)
        P.free(O.buf, G.buf, lowbuf)
        P.free(*[r.buf for r in memo.values()])                  # deterministic (memo cycle, see docstring)
        P.phase = "F"
        # constant right-hand sides (boundary values): evaluator + _embed, every call
        eqs = host.plan.problem.equations
        outs: dict = {k: res[k] for k in res}
        for k, (e, shape_k) in enumerate(zip(self.rhs_exprs, host.eq_shapes, strict=False)):
            if e is not None and k not in outs:
                small = P.alloc(8, "F coefficients", "constant value")
                outs[k] = _new(P, shape_k, False, "F coefficients", f"embedded constant eq{k}")
                P.free(small)
        P.phase = "F.gather"
        b = self._gather(outs, eqs)
        P.phase = "F"
        P.free(*[a.buf for a in outs.values()])
        scatter_bytes = sum(P.size[a.buf] for a in data.values())
        P.free(*[a.buf for a in data.values()])
        self.gc_pending_per_F = memo_bytes + scatter_bytes
        P.retag(b, "RHS F(X)")
        return b

    def _chain(self, a: _A, ops) -> _A:
        """Transforms applied in turn along the given axes (each frees its input)."""
        for fn, ax in ops:
            an = fn(a, self.fu.bases[ax], ax + 1)
            self.P.free(a.buf)
            a = an
        return a

    def _batched(self, A: _A, fn, cat: str, name: str) -> _A:
        """FusedRHS._batched: fn on slices of at most ``transform_batch`` fields, results copied
        into one array allocated after the first slice; one call when the batch holds them all."""
        k = getattr(self.fu, "transform_batch", None)
        n = A.shape[0]
        if not k or n <= k:
            return fn(A)
        out = None
        for i0 in range(0, n, k):
            m = min(k, n - i0)
            r = fn(_A(None, (m,) + tuple(A.shape[1:]), A.cplx, A.contig, False))
            if out is None:
                out = _new(self.P, (n,) + tuple(r.shape[1:]), r.cplx, cat, name)
            self.P.free(r.buf)
        self.P.free(A.buf)
        return out

    def _gather(self, outs: dict, eqs) -> int:
        P, host = self.P, self.host
        b = P.alloc(self.n * self.it, "RHS F(X)", "b (stacked RHS)")
        zc = host.plan.layout.zcoord
        coords = host.plan.domain.coordsys.coords
        for (k, lam), sel in host._row_sel.items():
            c = outs.get(k)
            if c is None:
                continue
            conv = None
            if lam > 0:
                ax = eqs[k].lhs.rank + list(coords).index(zc)
                if c.shape[ax] > 1:
                    if lam not in self.conv:
                        S = host.plan.layout.zconv(lam).tocsr()
                        ph, P.phase = P.phase, "F.gather"
                        self.conv[lam] = [P.alloc((S.shape[0] + 1) * 4, "constants", f"S{lam}.indptr"),
                                          P.alloc(S.nnz * 4, "constants", f"S{lam}.indices"),
                                          P.alloc(S.nnz * self.it, "constants", f"S{lam}.data")]
                        P.phase = ph
                    rc = None
                    if not (ax == 0 and c.contig):
                        rc = P.alloc(c.size * (16 if c.cplx else 8), "F coefficients", "reshape copy")
                    cast = None
                    if c.cplx != self.cplx:
                        cast = P.alloc(c.size * self.it, "F coefficients", "astype")
                    conv = _new(P, c.shape, self.cplx, "F coefficients", f"S{lam} @ c")
                    P.free(cast, rc)
                    conv.contig = ax == 0
                    conv.fresh = False
                    c = conv
            rs = None
            if not c.contig:
                rs = P.alloc(c.size * (16 if c.cplx else 8), "F coefficients", "reshape(-1) copy")
            fl = P.alloc(sel.size * (16 if c.cplx else 8), "F coefficients", "c[flat]")
            P.free(rs, fl)
            P.free(conv.buf if conv is not None else None)
        return b

    # ------------------------------------------------------------------ linear algebra
    def matvec(self, cat: str, name: str) -> int:
        ph, self.P.phase = self.P.phase, "matvec"
        b = self.P.alloc(self.n * self.it, cat, name)
        self.P.phase = ph
        return b

    def _factor(self, key) -> None:
        if key in self.core_bufs:
            return
        P, host = self.P, self.host
        bufs = []

        def buf(nbytes, name, keep=True):
            b = P.alloc(nbytes, "banded factors (core pool)", name, pool="core")
            if keep:
                bufs.append(b)
            return b
        zb = list(getattr(host, "_zbuckets", []))
        # complex-mode buckets sharing their partner's factors (checked exactly on the host
        # matrices of this key, timestep/shared_factors.py): two (n,) or (batch, n) sign arrays
        # checked on the actual (a, b) of the key when dt is known (the relation is a property
        # of a M + b L, not of M and L separately: step 3); without dt on the
        # model key's coefficients (RK: (1, a_ii)); keys that are not two floats: not shared
        ab = self._actual_coefficients(key)
        coef = ab if ab is not None else key[-2:]
        shared = [sh for _m, _nb, sh in host._zbucket_matrices(*coef)] if zb and \
            all(isinstance(v, float) for v in coef) else [None] * len(zb)
        self._shared_flags[key] = [sh is not None for sh in shared]
        allb = list(host._buckets) + list(host._cbuckets) + zb
        nrc = len(host._buckets) + len(host._cbuckets)
        streams = None
        if self.stream_samples > 0 and ab is not None:
            from .level_stream import host_bucket_streams
            streams = host_bucket_streams(host, *ab, samples=self.stream_samples,
                                          merge=self.level_merge)
            self.streams_estimated = True
        for bi, ((label, n, r, kl, ku), (idx, *_rest)) in enumerate(zip(host.bands(), allb, strict=False)):
            T = 16 if "(complex" in label else self.it
            B = len(idx)
            sh = shared[bi - nrc] if bi >= nrc else None
            if sh is not None:
                buf(sh.dr.size * 8, f"{label}: signs D_r (shared factors)")
                buf(sh.dc.size * 8, f"{label}: signs D_c (shared factors)")
                continue
            ldab = 2 * kl + ku + 1
            cpt = self.compact_factors
            band = buf(ldab * n * B * T, f"{label}: band LU", keep=not cpt)
            buf(n * 4, f"{label}: perm")
            ipiv = buf(B * n * 4, f"{label}: ipiv", keep=not cpt)
            vt = None
            if r > 0:
                vt = buf(r * n * B * T, f"{label}: Vt", keep=not cpt)
                buf(r * n * B * T, f"{label}: W")
                buf(r * r * B * T, f"{label}: capacitance")
                buf(r * B * 4, f"{label}: capacitance pivots")
            info = buf(B * 4, "info", keep=False)
            cond = buf(B * 8, "cond", keep=False)
            P.free(info, cond)
            scr = None
            if n * T > 40 * 1024:
                scr = buf(B * n * T, f"{label}: scratch", keep=not cpt)
            st = streams[bi] if streams is not None else None
            est = st is not None and st.get("applies")
            # level-scheduled solve (banded.cu build_level_schedule): the factor stream (L' and U
            # without their zeros, merged levels) with its gather permutation, offsets and sparse
            # V^T, estimated from sampled modes (runtime/level_stream.py); without the estimate,
            # only the maps are counted (declared in the report)
            if est:
                buf(int(st["forward_bytes"]), f"{label}: level stream + maps (estimated)")
            else:
                buf(B * n * 4, f"{label}: level gather permutation")
                buf((B + 1) * 8, f"{label}: level stream offsets")
                if r > 0:
                    buf(B * (r + 1) * 8, f"{label}: sparse V^T row pointers")
            if cpt:                              # build_transpose_schedule, then release_band
                if r > 0:
                    buf(r * n * B * T, f"{label}: Wt (transposed Woodbury)")
                    buf(r * r * B * T, f"{label}: St")
                    buf(r * B * 4, f"{label}: St pivots")
                if est and st.get("transposed_bytes") is not None:
                    buf(int(st["transposed_bytes"]), f"{label}: transposed level stream + positions (estimated)")
                else:
                    buf(B * n * 4, f"{label}: transposed level positions")
                    buf((B + 1) * 8, f"{label}: transposed stream offsets")
                P.free(band, ipiv, vt, scr)
        self.core_bufs[key] = bufs

    def _actual_coefficients(self, key):
        """(a, b) of the solve a M + b L behind a model key (integrator.py), or None without dt."""
        dt = self.stream_dt
        if dt is None:
            return None
        if key[0] == "SBDF":
            from ..timestep.integrator import sbdf_coefficients
            return sbdf_coefficients(int(key[1]))[0][0] / dt, 1.0
        if key[0] == "CN":
            return float(key[1]) / dt, float(key[2])
        return float(key[0]), float(key[1]) * dt          # RK: (1, a_ii) -> (1, dt a_ii)

    def solve(self, key, rhs: int) -> int:
        P, host, it = self.P, self.host, self.it
        ph, P.phase = P.phase, "solve"
        self._factor(key)
        x = P.alloc(self.n * it, "solve workspace", "x (solution)")
        # device.py _solve_here: the real buckets gather rhs[rows] and scatter x[cols] inside the
        # solve kernel, and the complexified buckets of a real right-hand side split z = rhs[rc] +
        # i rhs[rs] in the kernel too: no staging copies (the model kept the copies of the
        # earlier runtime: 4 transient buffers per step too many, gpu4)
        prev = None
        for (_idx, rc, _rs, _cc, _cs) in (host._cbuckets if self.cplx else ()):
            t1 = P.alloc(rc.size * 8, "solve workspace", "rhs[rc]")
            t2 = P.alloc(rc.size * 8, "solve workspace", "rhs[rs]")
            t3 = P.alloc(rc.size * 16, "solve workspace", "1j * rhs[rs]")
            P.free(t2)
            Z = P.alloc(rc.size * 16, "solve workspace", "Z")
            P.free(t1, t3, prev)
            prev = Z
        P.free(prev)
        # complex modes (device.py): fused gather / scatter kernels (real systems, the default):
        # one complex Z per bucket; otherwise the CuPy expressions (gather, product, running sum,
        # astype copy, sign scaling of a shared bucket, then x[c] += (w Z).real per position)
        shared = self._shared_flags.get(key, ())
        fused = not self.cplx
        for bi, (_keys, rows, _cols, wr, _wc) in enumerate(getattr(host, "_zbuckets", [])):
            m = np.asarray(rows)[0].size
            if fused:
                Z = P.alloc(m * 16, "solve workspace", "Z (fused gather)")
                P.free(Z)
                continue
            acc = None
            for _w in wr:
                g = P.alloc(m * 8, "solve workspace", "rhs[rows_p]")
                t = P.alloc(m * 16, "solve workspace", "w_p rhs[rows_p]")
                P.free(g)
                new = P.alloc(m * 16, "solve workspace", "sum_p") if acc is not None else t
                if acc is not None:
                    P.free(acc, t)
                acc = new
            Z = P.alloc(m * 16, "solve workspace", "Z (astype)")
            P.free(acc)
            if bi < len(shared) and shared[bi]:
                Zs = P.alloc(m * 16, "solve workspace", "Z * D_r")
                P.free(Z)
                Z = P.alloc(m * 16, "solve workspace", "Z * D_c")
                P.free(Zs)
            for _w in wr:
                t = P.alloc(m * 16, "solve workspace", "w_p Z")
                g = P.alloc(m * 8, "solve workspace", "x[cols_p]")
                a = P.alloc(m * 8, "solve workspace", "x[cols_p] + (w_p Z).real")
                P.free(t, g, a)
            P.free(Z)
        P.phase = ph
        return x

    # ------------------------------------------------------------------ schemes
    def rk_step(self) -> None:
        from ..timestep.integrator import rk_tableau
        P, V = self.P, self.n * self.it
        A, H, c = rk_tableau(self.scheme)
        s = len(c) - 1
        P.phase = "step"
        X = [self.X]
        Fs: list = [None] * (s + 1)
        LX: list = [None] * (s + 1)
        # integrator.rk_stages on the GPU: M x0 and L X_k are deferred (lincomb.CsrApply). Real
        # float64 systems: rk_rhs evaluates the whole combination, CSR products included, in one
        # kernel (one new rhs per stage, nothing else). Complex systems: the array path, where each
        # deferred product is computed at its first use and kept, and each term makes one
        # temporary (gpu4: the model allocated M X0 and L X_k up front, as before)
        fused = (not self.cplx) if self.rk_fused is None else bool(self.rk_fused)
        MX0 = None
        rhs = None
        for i in range(1, s + 1):
            j = i - 1
            if any(A[m][j] != 0 for m in range(j + 1, s + 1)):
                Fs[j] = self.F()
                P.phase = "step"
            if fused:
                new = P.alloc(V, "integrator temporaries", "rhs (one kernel)")
            else:
                if MX0 is None:
                    MX0 = self.matvec("integrator temporaries", "M X0")
                new = P.alloc(V, "integrator temporaries", "rhs = M X0 (copy)")
                for k in range(i):
                    for coef, isH in ((A[i][k], False), (H[i][k], True)):
                        if coef != 0:
                            if isH and LX[k] is None:
                                LX[k] = self.matvec("integrator temporaries", f"L X{k}")
                            t = P.alloc(V, "integrator temporaries", "dt a F / dt h L X")
                            P.free(t)
            P.free(rhs)
            rhs = new
            Xi = self.solve((1.0, float(H[i][i])), rhs)
            P.phase = "step"
            P.retag(Xi, "RK stages" if i < s else "state")
            X.append(Xi)
        P.free(rhs, MX0, *Fs, *LX, *X[1:-1])
        P.free(self.X)
        self.X = X[-1]

    def sbdf_step(self) -> None:
        from ..timestep.integrator import sbdf_coefficients
        P, V = self.P, self.n * self.it
        q_max = int(self.scheme[4:])
        P.phase = "step"
        mx = self.matvec("multistep history", "M X^n")
        self.hist_MX.insert(0, mx)
        f = self.F()
        P.phase = "step"
        P.retag(f, "multistep history")
        self.hist_F.insert(0, f)
        while len(self.hist_MX) > q_max:
            P.free(self.hist_MX.pop(), self.hist_F.pop())
        q = min(q_max, len(self.hist_MX))
        a, _c = sbdf_coefficients(q)
        rhs = P.alloc(V, "integrator temporaries", "rhs (zeros)")
        for _j in range(1, q + 1):
            t1 = P.alloc(V, "integrator temporaries", "c F")
            t2 = P.alloc(V, "integrator temporaries", "a/dt M X")
            t3 = P.alloc(V, "integrator temporaries", "c F - a/dt M X")
            P.free(t1, t2, t3)
        x = self.solve(("SBDF", q), rhs)                   # one factorization per order q
        P.phase = "step"
        P.retag(x, "state")
        P.free(self.X, rhs)
        self.X = x

    def cn_step(self) -> None:
        """Mirror of ``IVPSolver._cn_step``: history of M X, L X and F; the right-hand side
        sum c F - (a/dt) M X - b L X accumulated from a zero vector (one kernel per stage on the
        GPU: one output, no per-term temporary); one factorization per coefficient set."""
        from ..timestep.integrator import cn_coefficients
        P, V = self.P, self.n * self.it
        a_s, b_s, _c = cn_coefficients(self.scheme)
        s = len(a_s) - 1
        P.phase = "step"
        self.hist_MX.insert(0, self.matvec("multistep history", "M X^n"))
        self.hist_LX.insert(0, self.matvec("multistep history", "L X^n"))
        f = self.F()
        P.phase = "step"
        P.retag(f, "multistep history")
        self.hist_F.insert(0, f)
        while len(self.hist_MX) > s:
            P.free(self.hist_MX.pop(), self.hist_LX.pop(), self.hist_F.pop())
        a, b, _c = cn_coefficients(self.scheme if len(self.hist_MX) >= s else "CNAB1")
        zero = P.alloc(V, "integrator temporaries", "rhs base (zeros)")
        rhs = P.alloc(V, "integrator temporaries", "rhs")
        P.free(zero)
        x = self.solve(("CN", a[0], b[0]), rhs)            # one factorization per (a0, b0)
        P.phase = "step"
        P.retag(x, "state")
        P.free(self.X, rhs)
        self.X = x

    def run(self, steps: int) -> StepProgram:
        from ..timestep.integrator import scheme_family
        self.setup()
        self.set_state()
        step = {"sbdf": self.sbdf_step, "cn": self.cn_step, "rk": self.rk_step}[
            scheme_family(self.scheme)]
        for k in range(int(steps)):
            self.P.step = k
            step()
        return self.P


# ====================================================================== report
@dataclass
class MemoryReport:
    """Declared (predicted) device memory of an IVP run; ``explain()`` prints the table.

    ``categories``: bytes live at the predicted peak, by category (CuPy pool + core pool);
    ``persistent``: bytes alive after the last step, by category; ``predicted``: totals
    (peak live, pool-held, core pool, total held, planned arena); ``reuse``: the liveness
    proposal; ``estimated``: names of fields that are estimates rather than exact sizes."""
    scheme: str
    n: int
    steps: int
    categories: dict = field(default_factory=dict)
    persistent: dict = field(default_factory=dict)
    predicted: dict = field(default_factory=dict)
    reuse: dict = field(default_factory=dict)
    estimated: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    program: StepProgram | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """JSON-ready summary of the report."""
        return {k: getattr(self, k) for k in ("scheme", "n", "steps", "categories", "persistent",
                                               "predicted", "reuse", "estimated", "notes")}

    def compare(self, measured: dict) -> dict:
        """Relative gaps (predicted - measured) / measured for the quantities both sides have.
        ``measured``: the summary of :func:`conatus.runtime.memtrace.trace_device_memory`."""
        p, m = self.predicted, measured
        pairs = {
            "cupy_peak_live": (p["cupy_peak_live_bytes"], m.get("cupy_peak_live_bytes")),
            "cupy_pool_held": (p["cupy_pool_held_bytes"], m.get("cupy_pool_held_bytes")),
            "core_pool": (p["core_pool_bytes"], (m.get("core_live_bytes") or 0) + (m.get("core_cached_bytes") or 0)),
            "total_held": (p["total_held_bytes"], (m.get("cupy_pool_held_bytes") or 0)
                           + (m.get("core_live_bytes") or 0) + (m.get("core_cached_bytes") or 0)),
        }
        return {k: {"predicted": int(a), "measured": int(b), "rel_gap": (a - b) / b if b else None}
                for k, (a, b) in pairs.items() if b is not None}

    def explain(self, measured: dict | None = None) -> str:
        p = self.predicted
        lines = [f"Memory report (declared by the liveness planner, not measured): {self.scheme}, "
                 f"n = {self.n}, {self.steps} steps modelled"]
        lines.append("  live at the predicted peak, by category:")
        w = max(len(c) for c in CATEGORIES)
        for c in CATEGORIES:
            v = self.categories.get(c, 0)
            if v:
                lines.append(f"    {c:<{w}}  {v / MiB:10.2f} MiB   (persistent {self.persistent.get(c, 0) / MiB:.2f})")
        lines.append(f"  CuPy pool: peak live {p['cupy_peak_live_bytes'] / MiB:.2f} MiB, held "
                     f"{p['cupy_pool_held_bytes'] / MiB:.2f} MiB (pool model: 512 B rounding, best fit, split/merge)")
        lines.append(f"  core pool (banded factors, {p['factor_keys']} key(s)): {p['core_pool_bytes'] / MiB:.2f} MiB")
        lines.append(f"  held by the allocators (CuPy pool + core pool): {p['total_held_bytes'] / MiB:.2f} MiB")
        lines.append(f"  + CUDA process floor (context, kernel images, cuFFT/cuBLAS handles; declared "
                     f"constant, not modelled): {p['cuda_process_floor_bytes'] / MiB:.0f} MiB "
                     f"({CUDA_PROCESS_FLOOR_SOURCE})")
        lines.append(f"  TOTAL device memory of the process: {p['total_with_floor_bytes'] / MiB:.2f} MiB")
        if "peak_bound_bytes" in p:
            lines.append(f"  peak bound (CuPy pool held + core pool high-water, factorization included, "
                         f"+ process floor): {p['peak_with_floor_bytes'] / MiB:.2f} MiB "
                         f"(allocators alone {p['peak_bound_bytes'] / MiB:.2f} MiB)")
        r = self.reuse
        lines.append(f"  reuse proposal (static liveness plan of one step's {r['n_transient']} transient buffers): "
                     f"arena {r['arena_bytes'] / MiB:.2f} MiB vs {r['sum_bytes'] / MiB:.2f} MiB without reuse; "
                     f"{r['n_reusing']} buffers reuse freed memory; planned total "
                     f"{p['planned_total_bytes'] / MiB:.2f} MiB (not applied by the runtime)")
        for ex in r.get("examples", [])[:6]:
            lines.append(f"    {ex}")
        if self.estimated:
            lines.append("  ESTIMATED (not exact): " + ", ".join(self.estimated))
        lines.append(f"  F memo reference cycle: each uncollected F evaluation keeps "
                     f"{p['gc_pending_per_F_bytes'] / MiB:.2f} MiB until CPython's cyclic GC runs (not in the totals)")
        if measured is not None:
            lines.append("  vs measured:")
            for k, v in self.compare(measured).items():
                g = v["rel_gap"]
                lines.append(f"    {k:<16} predicted {v['predicted'] / MiB:10.2f}  measured "
                             f"{v['measured'] / MiB:10.2f} MiB  gap {100 * g:+.1f} %" if g is not None else "")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def arena_offsets(P: StepProgram, step: int) -> list[tuple[int, int, int, int, int]]:
    """Static offset assignment of the transient CuPy buffers of ``step`` (greedy by decreasing
    size on the interval graph): list of (offset, end, alloc time, free time, buffer). Two buffers
    whose lifetimes overlap never overlap in address (checked by the tests)."""
    iv = P.intervals()
    ts = [t for t, e in enumerate(P.events) if e[3] == step]
    if not ts:
        return []
    first, last = ts[0], ts[-1]
    bufs = [b for b, (a, f) in iv.items()
            if P.pool[b] == "cupy" and a >= first and f is not None and f <= last and P.size[b] > 0]
    bufs.sort(key=lambda b: (-P.size[b], iv[b][0]))
    placed: list = []
    for b in bufs:
        a, f = iv[b]
        sz = -(-P.size[b] // 512) * 512
        busy = sorted((o, e) for o, e, a2, f2, _ in placed if a2 < f and a < f2)
        off = 0
        for o, e in busy:
            if off + sz <= o:
                break
            off = max(off, e)
        placed.append((off, off + sz, a, f, b))
    return placed


def _arena_plan(P: StepProgram, step: int) -> dict:
    """Summary of :func:`arena_offsets`: arena size, bytes without reuse, reusing buffers."""
    iv = P.intervals()
    placed = arena_offsets(P, step)
    if not placed:
        return dict(n_transient=0, arena_bytes=0, sum_bytes=0, n_reusing=0, examples=[])
    arena = max((e for _o, e, *_ in placed), default=0)
    reusing, examples = 0, []
    for off, end, a, _f, b in placed:
        prev = [b2 for o2, e2, a2, f2, b2 in placed if f2 <= a and o2 < end and off < e2]
        if prev:
            reusing += 1
            if len(examples) < 12 and P.size[b] >= 2 ** 20:
                p = max(prev, key=lambda x: iv[x][1])
                examples.append(f"{P.name[b]} [{P.cat[b]}, {P.size[b] / MiB:.2f} MiB] reuses "
                                f"{P.name[p]} [{P.cat[p]}]")
    return dict(n_transient=len(placed), arena_bytes=int(arena),
                sum_bytes=int(sum(e - o for o, e, *_ in placed)), n_reusing=reusing,
                examples=examples)


def _report(model: _Model, steps: int) -> MemoryReport:
    P = model.run(steps)
    last = steps - 1
    peak, _at, cats_c = P.live_peak("cupy")
    core_peak, _t, cats_k = P.live_peak("core")
    sizes = P.size
    cupy_events = [e for e in P.events if P.pool[e[1]] == "cupy"]
    held = simulate_cupy_pool(cupy_events, sizes)
    core = _CorePool(trim=model.compact_factors)
    for k, b, _p, _s in P.events:
        if P.pool[b] == "core":
            (core.malloc(sizes[b]) if k == "a" else core.free(-(-sizes[b] // 64) * 64))
    # compact factors: the released band storage is returned to the driver (pool_trim)
    core_bytes = core.live + core.cached
    iv = P.intervals()
    persistent: dict = {}
    for b, (_a, f) in iv.items():
        if f is None:
            persistent[P.cat[b]] = persistent.get(P.cat[b], 0) + sizes[b]
    cats = dict(cats_c)
    for c, v in cats_k.items():
        cats[c] = cats.get(c, 0) + v
    reuse = _arena_plan(P, last)
    pers_cupy = sum(sizes[b] for b, (_a, f) in iv.items() if f is None and P.pool[b] == "cupy")
    rep = MemoryReport(scheme=model.scheme, n=model.n, steps=steps, categories=cats,
                       persistent=persistent, reuse=reuse, program=P)
    rep.predicted = dict(
        cupy_peak_live_bytes=int(peak), cupy_pool_held_bytes=int(held),
        core_pool_bytes=int(core_bytes), total_held_bytes=int(held + core_bytes),
        persistent_cupy_bytes=int(pers_cupy), planned_arena_bytes=int(reuse["arena_bytes"]),
        planned_total_bytes=int(pers_cupy + reuse["arena_bytes"] + core_bytes),
        factor_keys=len(model.core_bufs), gc_pending_per_F_bytes=int(model.gc_pending_per_F),
        cufft_work_bytes=int(sum(sizes[b] for b in model.plan_bufs.values())),
        cufft_source=model.cufft.source,
        # the factorization is transient: the core pool's high-water mark (band LU and level
        # streams of a batch alive together) on top of everything the CuPy pool holds is an
        # upper bound of the device peak ( step 3: "held" alone missed it)
        core_pool_peak_bytes=int(core.peak), peak_bound_bytes=int(held + core.peak),
        # declared process floor (CUDA context, kernels, library handles), its own line
        cuda_process_floor_bytes=int(CUDA_PROCESS_FLOOR_BYTES),
        # which F program was modelled: the 2D RealFourier x Chebyshev pipeline or the generic fused F
        f_path="2D pipeline" if model.fc2d else "fused",
        total_with_floor_bytes=int(held + core_bytes + CUDA_PROCESS_FLOOR_BYTES),
        peak_with_floor_bytes=int(held + core.peak + CUDA_PROCESS_FLOOR_BYTES))
    if model.cufft.source != "cufft":
        rep.estimated.append("cuFFT work areas (no cuFFT query: estimate)")
    if getattr(model.fu, "_ffc", None) is not None:
        rep.estimated.append("F transforms: the batched 3D path (use_ffc) is modelled as the per-axis one")
    if model.streams_estimated:
        rep.estimated.append(f"level streams of the factors ({model.stream_samples} sampled modes per "
                             "batch, host replay of the device factorization and stream builder)")
    else:
        rep.notes.append("level streams of the factors not counted (stream_samples = 0 or non-float keys)")
    rep.notes.append("CUDA context, kernel images, cuFFT/cuBLAS handles: not modelled buffer by buffer, "
                     "counted as the declared process floor CUDA_PROCESS_FLOOR_BYTES")
    rep.notes.append("deterministic program: F derivative memo freed at return (memo reference cycle: the "
                     "runtime frees it at the next cyclic GC)")
    return rep


def _steps_for(scheme: str) -> int:
    """Steps modelled: enough for the steady state (SBDF q: q start-up steps + 2; CN: history
    levels + 2; RK: 3)."""
    from ..timestep.integrator import cn_coefficients, scheme_family
    fam = scheme_family(scheme)
    if fam == "sbdf":
        return int(scheme[4:]) + 2
    return len(cn_coefficients(scheme)[0]) + 1 if fam == "cn" else 3


def memory_report(solver, steps: int | None = None, cufft_query: bool | None = None,
                  set_state_fields: tuple = ()) -> MemoryReport:
    """Declared memory of ``solver`` (``IVPSolver``, backend 'cuda', f_backend 'fused') from its
    compiled objects; nothing is measured. ``steps``: time steps modelled (default: steady state);
    ``set_state_fields``: variables given to ``set_state`` as grid data (transient copies)."""
    if solver.backend != "cuda":
        from ..errors import UnsupportedError
        raise UnsupportedError(
            f"memory_report: this solver runs on backend={solver.backend!r}, which has no device "
            "memory to plan",
            why="the liveness planner models the device allocations of backend='cuda'",
            fix="to size a GPU run before it exists (no GPU needed): "
                "conatus.runtime.memory.plan_memory(solver.plan, solver.scheme), or "
                "plan_memory(m.discretize(N), scheme) from a native model; or build the solver "
                "with backend='cuda' on a GPU node")
    host = getattr(solver.system, "host", solver.system)
    q = _gpu_present() if cufft_query is None else cufft_query
    host._plan_dt = float(solver.dt)                    # actual factor keys (level streams)
    m = _Model(host, solver._fused, solver.scheme, solver._rhs_exprs, startup=solver.startup,
               cufft_query=q, fc2d=getattr(solver, "_fc2d", None) is not None,
               set_state_fields=set_state_fields)
    return _report(m, steps or _steps_for(solver.scheme))


def plan_memory(plan, scheme: str, *, f_backend: str = "fused", reduced: bool | None = None,
                ordering: str = "interleaved", startup: str = "lower_order",
                steps: int | None = None, cufft_query: bool | None = None,
                set_state_fields: tuple = (), **system_options) -> MemoryReport:
    """Same as :func:`memory_report` before building a solver (CPU only, no device needed): the
    stacked system and the fused RHS are compiled on the host exactly as ``IVPSolver`` does.
    ``system_options`` (e.g. ``complexify=True``) are passed to ``GlobalSystem`` unchanged."""
    from ..timestep.system import GlobalSystem
    if f_backend != "fused":
        from ..errors import UnsupportedError
        raise UnsupportedError("plan_memory: only f_backend='fused' is modelled",
                               fix="plan_memory(plan, scheme) with the default f_backend='fused'")
    host = GlobalSystem(plan, reduced=reduced, ordering=ordering, **system_options)
    pb = plan.problem
    rhs = [None if eq.rhs.is_zero() else eq.rhs for eq in pb.equations]
    from ..timestep.fused import FusedRHS
    if reduced and pb.symmetry is not None:
        raise NotImplementedError("plan_memory: the parity-reduced fused RHS is not modelled yet")
    fu = FusedRHS(plan.domain, pb.equations, rhs, dict(plan.params), dtype=host.dtype)
    q = _gpu_present() if cufft_query is None else cufft_query
    m = _Model(host, fu, scheme, rhs, startup=startup, cufft_query=q, set_state_fields=set_state_fields)
    return _report(m, steps or _steps_for(scheme))


def _gpu_present() -> bool:
    try:
        from . import cuda
        return bool(cuda.available())
    except Exception:          # noqa: BLE001 -- no CUDA build: CPU estimate, flagged
        return False

