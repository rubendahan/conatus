"""``solver.profile(steps)``: time per phase with explicit GPU synchronisation, and a simple roofline
.

**Measured.** Every phase of :mod:`conatus.runtime.phases` is bracketed by two CUDA events recorded
on the current stream (GPU) or by ``time.perf_counter`` (CPU). After the timed steps the device is
synchronised and each span read with ``cudaEventElapsedTime``. A span is the *GPU stream timeline*
between the two records: it contains the kernels of the phase and the gaps in which the stream waited
for the host to launch the next one (a launch-bound phase shows its host cost this way). Inclusive and
exclusive (minus nested phases) times are reported per step. The wall time per step is measured
twice, with and without the probe, so the probe's own cost is visible.

**Estimated** (every such field is listed in ``ProfileReport.estimated``): bytes moved and flops of
each call, from the arguments of the call (array shapes, band widths, nnz, pointwise monomials):

* transforms: the allocation program of ``xp_transforms`` (:class:`conatus.runtime.memory._XTModel`)
  gives the bytes written (each temporary written once, read once, plus the input read); flops
  5 n log2 n per complex length-n FFT, 2.5 n log2 n per real one;
* pointwise products: (factors + outputs) x points x itemsize; flops from the monomials;
* spectral derivatives: 3 x coefficient bytes; 2 flops per coefficient;
* banded solves (per mode): band LU (2kl+ku+1)n, pivots, Woodbury factors 2rn, right-hand side read
  and written; flops 2 n (2 kl + ku) + 4 r n + 2 r^2 (x4 complex);
* CSR matvec: nnz (value + index) + row pointers + x read + y written; 2 nnz flops;
* IMEX arithmetic (``step`` exclusive): 2 n per copy, 5 n per axpy (temporary + in-place add);
* gather/scatter copies (``solve`` exclusive): rhs[rows] and x[cols] with their int64 maps.
  ``F.fused`` exclusive, ``F`` exclusive, ``F.scatter`` and ``F.gather`` are not estimated.

**Declared** peaks: memory bandwidth = 2 x memory clock x bus width (device attributes); FP64 peak =
2 x FP64 FMA units per SM (spec table per compute capability) x SMs x clock. A user can pass
``peaks=`` instead. Roofline: attainable = min(FP64 peak, intensity x bandwidth).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .phases import PhaseProbe

__all__ = ["profile", "ProfileReport", "DevicePeaks", "device_peaks", "GROUPS"]

# coarse groups (the phases they sum, exclusive times); together they cover the whole step
GROUPS = {
    "transforms": ("F.transforms",),
    "nonlinear products": ("F.products",),
    "spectral derivatives": ("F.derivatives",),
    "banded solve": ("solve.banded",),
    "copies (gather/scatter, stacking)": ("solve", "F.scatter", "F.gather", "F.fused"),
    "CSR matvec": ("matvec",),
    "IMEX arithmetic": ("step",),
    "host glue / evaluator": ("F", "F.evaluator"),
}

# FP64 FMA units per SM by compute capability (vendor architecture whitepapers)
_FP64_FMA_PER_SM = {(7, 0): 32, (7, 5): 2, (8, 0): 32, (8, 6): 2, (8, 7): 2, (8, 9): 2, (9, 0): 64}


@dataclass
class DevicePeaks:
    """Declared peaks of the device (not measured): bytes/s and FP64 flop/s."""
    name: str
    bandwidth: float
    fp64: float
    source: str

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (flop/byte) where the roofline turns from memory to compute."""
        return self.fp64 / self.bandwidth


def device_peaks() -> DevicePeaks | None:
    """Declared peaks of the current CUDA device from its attributes (None without a GPU)."""
    try:
        import cupy as cp
        rt = cp.cuda.runtime
        dev = cp.cuda.Device().id
        # CUDA device attribute enums (cudaDeviceAttr): stable across CUDA versions
        at = {k: int(rt.deviceGetAttribute(v, dev)) for k, v in dict(
            clock=13, sms=16, mem_clock=36, bus=37, major=75, minor=76).items()}
        props = rt.getDeviceProperties(dev)
    except Exception:            # noqa: BLE001 -- no device: no declared peaks
        return None
    cc = (at["major"], at["minor"])
    bw = 2.0 * at["mem_clock"] * 1e3 * at["bus"] / 8.0
    fma = _FP64_FMA_PER_SM.get(cc)
    fp64 = float("nan") if fma is None else 2.0 * fma * at["sms"] * at["clock"] * 1e3
    return DevicePeaks(props["name"].decode(), bw, fp64,
                       f"declared: 2 x memory clock x bus width; FP64 = 2 x {fma} FMA/SM (sm_{cc[0]}{cc[1]}) "
                       "x SMs x boost clock (device attributes)")


# ====================================================================== work estimates
class _Desc:
    """Shape-only stand-in of an array argument: recorded during the timed steps (cheap, keeps no
    device memory alive), turned into estimates after the timing."""
    __slots__ = ("shape", "dtype", "ndim", "flags", "base")

    class _Flags:
        __slots__ = ("c_contiguous",)

        def __init__(self, c: bool) -> None:
            self.c_contiguous = c

    def __init__(self, a) -> None:
        self.shape, self.dtype, self.ndim = tuple(a.shape), a.dtype, a.ndim
        self.flags = _Desc._Flags(bool(a.flags.c_contiguous))
        self.base = None if getattr(a, "base", None) is None else True


def _describe(x):
    if isinstance(x, np.generic):                  # numpy scalars (kernel sizes) stay values
        return x
    if hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "flags"):
        return _Desc(x)
    if isinstance(x, tuple):
        return tuple(_describe(v) for v in x)
    return x


def _nbytes(a) -> int:
    return int(np.prod(a.shape)) * np.dtype(a.dtype).itemsize


def _fft_flops(n: int, vtype: str, batch: int) -> float:
    per = 5.0 if vtype == "C2C" else 2.5
    return per * n * math.log2(max(n, 2)) * batch


class _Estimator:
    """Bytes / flops of one call, from its arguments (see module docstring). None: not estimated."""

    def __init__(self, solver) -> None:
        self.solver = solver
        fu = getattr(solver, "_fused", None)
        # StageTimeFused (time-dependent RHS, conatus.adjoint.stagetime) is a duck-typed wrapper:
        # the compiled kernel and its plan are in .inner (gpu1: AttributeError 'cplx' made the
        # profile fail)
        fu = getattr(fu, "inner", fu)
        self.pp_flops = None
        if fu is not None:
            mul, add = (6, 2) if fu.cplx else (1, 1)
            self.pp_flops = sum(mul * len(fs) + add for *_k, monos in fu.outputs for _c, fs in monos)
            self.nfo = len(fu.factor_list) + len(fu.outputs)
            self.fu_isz = 16 if fu.cplx else 8
        sysm = solver.system
        self.n = int(sysm.n)
        self.T = np.dtype(sysm.dtype).itemsize

    def __call__(self, label: str, fn, args: tuple, kw: dict):
        try:
            return getattr(self, "_" + label.replace(".", "_"))(fn, args, kw)
        except AttributeError:
            return None
        except Exception:        # noqa: BLE001 -- an estimate never breaks a run; it is absent
            return None

    def _F_transforms(self, fn, args, kw):
        from ..timestep import xp_transforms as XT
        from .memory import StepProgram, _A, _CufftSizer, _XTModel
        name = getattr(fn, "__name__", "")
        if len(args) >= 3 and hasattr(args[1], "shape"):                 # XT.forward/backward
            _xp, arr, b = args[:3]
            axis = (args[3] if len(args) > 3 else kw.get("axis", 0)) % arr.ndim
            m = _XTModel(StepProgram(), _CufftSizer(False))
            a = _A(None, tuple(arr.shape), np.iscomplexobj(arr), bool(arr.flags.c_contiguous),
                   getattr(arr, "base", None) is None)
            (m.backward if fn is XT.backward or name == "backward" else m.forward)(a, b, axis)
            wrote = sum(m.P.size)
            return (_nbytes(arr) + 2 * wrote, sum(_fft_flops(*f) for f in m.ffts))
        a = args[0]                                                       # direct cupy rfft/irfft
        axis = kw.get("axis", -1) % a.ndim
        n = kw.get("n", args[1] if len(args) > 1 else None) or a.shape[axis]
        batch = _nbytes(a) // max(1, a.shape[axis] * np.dtype(a.dtype).itemsize)
        return (4 * _nbytes(a), _fft_flops(int(n), "R2C", int(batch)))

    def _F_products(self, fn, args, kw):
        if self.pp_flops is None:
            return None
        P = int(args[2][2])
        return (self.nfo * P * self.fu_isz, float(self.pp_flops) * P)

    def _F_derivatives(self, fn, args, kw):
        if getattr(fn, "__name__", "") != "_diff":
            return (0, 0.0)                    # _full: a view in the common case
        c = args[0]
        cx = 2 if np.iscomplexobj(c) else 1
        return (3 * _nbytes(c), 2.0 * cx * int(np.prod(c.shape)))

    def _solve_banded(self, fn, args, kw):
        s = getattr(fn, "__self__", None)
        n, kl, ku, r, batch = s.n, s.kl, s.ku, s.rank, s.batch
        cplx = type(s).__name__.endswith("Z")
        T = 16 if cplx else 8
        if getattr(s, "level_scheduled", False):   # factor stream + V^T B^-1 + b, x, maps
            per_b = s.level_stream_bytes / batch + r * n * T + 2 * n * T + 3 * n * 8 + n * 4
            per_f = 2 * (s.level_stream_bytes / batch) / (10 if not cplx else 18) + 2 * r * n
            return (batch * per_b, (4.0 if cplx else 1.0) * batch * per_f)
        per_b = (2 * kl + ku + 1) * n * T + n * 4 + 2 * r * n * T + 2 * n * T
        per_f = 2 * n * (2 * kl + ku) + 4 * r * n + 2 * r * r
        return (batch * per_b, (4.0 if cplx else 1.0) * batch * per_f)

    def _matvec(self, fn, args, kw):
        A = getattr(fn, "__self__", None)
        nnz = int(A.data.size) if hasattr(A.data, "size") else int(A.nnz)
        rows = int(A.shape[0])
        T = self.T
        return (nnz * (T + 4) + (rows + 1) * 4 + 2 * rows * T, (4.0 if T == 16 else 1.0) * 2 * nnz)

    def _solve(self, fn, args, kw):
        if self.solver.backend != "cuda":
            return None              # CPU: the host banded solve is not split out of `solve`
        # exclusive part: rhs[rows] (read + write), x[cols] (read + write), two int64 maps
        return (4 * self.n * self.T + 2 * self.n * 8, 0.0)

    def _step(self, fn, args, kw):
        s, V = self.solver, self.n * self.T
        if s.order:
            q = s.order
            return (V + q * 10 * V, 2.0 * 3 * q * self.n)
        if getattr(s, "family", "") == "cn":
            # one-kernel accumulation of <= 3 terms per level (read F, M X, L X) and the zero base
            terms = sum(1 for co in s._cn for x in co[1:] if x != 0)
            return (2 * V + terms * V, 2.0 * self.n * terms)
        A, H, _c = s._rk
        terms = sum(1 for i in range(1, len(A)) for k in range(i) for co in (A[i][k], H[i][k]) if co)
        stages = len(A) - 1
        return (2 * V * stages + 5 * V * terms, 2.0 * self.n * terms)


# ====================================================================== report
@dataclass
class ProfileReport:
    """Per-phase times (measured) and roofline (estimated / declared). Times in seconds per step."""
    backend: str
    device: str
    steps: int
    wall_per_step: float
    wall_per_step_unprobed: float
    phases: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    peaks: DevicePeaks | None = None
    estimated: list = field(default_factory=list)
    measured: list = field(default_factory=list)
    declared: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("backend", "device", "steps", "wall_per_step",
                                            "wall_per_step_unprobed", "phases", "groups",
                                            "estimated", "measured", "declared", "notes")}
        d["peaks"] = None if self.peaks is None else dict(
            name=self.peaks.name, bandwidth=self.peaks.bandwidth, fp64=self.peaks.fp64, source=self.peaks.source)
        return d

    def explain(self) -> str:
        ms = 1e3
        lines = [f"Profile ({self.backend}, {self.device}): {self.steps} timed steps; wall per step "
                 f"{self.wall_per_step_unprobed * ms:.3f} ms unprobed, {self.wall_per_step * ms:.3f} ms "
                 "probed (measured, device synchronised)"]
        lines.append("  phase               calls/step  incl ms  excl ms  share  | est. GB/s  est. GF/s  "
                     "intensity  %roof  bound")
        tot = sum(p["exclusive"] for p in self.phases.values()) or 1.0
        for lab, p in sorted(self.phases.items(), key=lambda kv: -kv[1]["exclusive"]):
            rf = p.get("roofline") or {}
            extra = ""
            if rf:
                extra = (f"| {rf['achieved_GBps']:9.1f}  {rf['achieved_GFlops']:9.2f}  "
                         f"{rf['intensity']:9.3f}  {100 * rf['fraction_of_roof']:5.1f}  {rf['bound']}"
                         if rf.get("fraction_of_roof") is not None else
                         f"| {rf['achieved_GBps']:9.1f}  {rf['achieved_GFlops']:9.2f}  {rf['intensity']:9.3f}")
            lines.append(f"  {lab:<18} {p['calls']:10.1f}  {p['inclusive'] * ms:7.3f}  {p['exclusive'] * ms:7.3f}"
                         f"  {100 * p['exclusive'] / tot:4.1f}%  {extra}")
        lines.append("  groups (exclusive, ms/step): " + ", ".join(
            f"{g} {v * ms:.3f}" for g, v in sorted(self.groups.items(), key=lambda kv: -kv[1])))
        if self.peaks is not None:
            lines.append(f"  peaks ({self.peaks.source}): {self.peaks.bandwidth / 1e9:.0f} GB/s, "
                         f"FP64 {self.peaks.fp64 / 1e12:.2f} TF/s, ridge {self.peaks.ridge:.2f} flop/B")
        lines.append("  measured: " + ", ".join(self.measured))
        lines.append("  ESTIMATED: " + ", ".join(self.estimated))
        if self.declared:
            lines.append("  declared: " + ", ".join(self.declared))
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def profile(solver, steps: int = 10, *, warmup: int = 2, peaks: DevicePeaks | None = None,
            restore: bool = True) -> ProfileReport:
    """Time ``steps`` steps of ``solver`` phase by phase (see module docstring).

    ``warmup`` steps first (factorizations, cuFFT plans, kernel compilation are not timed). Then
    ``steps`` unprobed steps (wall time) and ``steps`` probed steps. ``restore=True`` puts the
    state, multistep history and clock back afterwards, so profiling does not advance the run."""
    from ..timestep.integrator import check_steps
    steps, warmup = check_steps(steps, "profile()", "steps"), check_steps(warmup, "profile()", "warmup")
    if steps < 1:                         # the time per step divides by it
        from ..errors import OptionError
        raise OptionError("profile(steps=0): nothing to time", fix="profile(steps=10)")
    gpu = solver.backend == "cuda"
    if gpu:
        import cupy as cp
        sync = cp.cuda.get_current_stream().synchronize
    else:
        def sync() -> None:
            return None
    saved = None
    if restore:
        def cp_(a):
            return a.copy()
        saved = (cp_(solver.X), [cp_(a) for a in solver._MXh], [cp_(a) for a in solver._Fh],
                 solver.t, solver.iteration, getattr(solver, "_t_origin", None),
                 getattr(solver, "_it_origin", None), solver.n_F,
                 [cp_(a) for a in getattr(solver, "_LXh", ())])     # CN family: L X history
    solver.run(warmup)
    sync()
    t0 = time.perf_counter()
    solver.run(steps)
    sync()
    wall_un = (time.perf_counter() - t0) / steps

    est = _Estimator(solver)
    calls: list = []                  # [label, start, end, child_incl_list_index, est]
    stack: list = []

    def now():
        if gpu:
            e = cp.cuda.Event()
            e.record()
            return e
        return time.perf_counter()

    def enter(label, fn, args, kw):
        # the work description is recorded before the start mark; estimates are computed after
        work = (fn, _describe(args), {k: _describe(v) for k, v in kw.items()})
        rec = [label, now(), None, [], work]
        if stack:
            stack[-1][3].append(rec)
        stack.append(rec)
        calls.append(rec)

    def leave(label):
        rec = stack.pop()
        rec[2] = now()

    probe = PhaseProbe(solver, enter, leave)
    sync()
    t0 = time.perf_counter()
    # the end-of-call checks (finite state, run-time resolution: one scatter of the state) are not
    # part of a step; off during the probed run so that the phases partition the steps (
    # a resolution check that is not yet warned runs at the end of every call)
    fe, solver._finite_every = solver._finite_every, 0
    try:
        with probe:
            solver.run(steps)
    finally:
        solver._finite_every = fe
    sync()
    wall = (time.perf_counter() - t0) / steps

    def span(rec) -> float:
        if gpu:
            return cp.cuda.get_elapsed_time(rec[1], rec[2]) * 1e-3
        return rec[2] - rec[1]

    incl = {id(r): span(r) for r in calls}
    for r in calls:
        r[4] = est(r[0], *r[4])
    agg: dict = {}
    for r in calls:
        lab = r[0]
        a = agg.setdefault(lab, dict(calls=0, inclusive=0.0, exclusive=0.0, bytes=0.0, flops=0.0,
                                     estimated_calls=0))
        a["calls"] += 1
        a["inclusive"] += incl[id(r)]
        a["exclusive"] += incl[id(r)] - sum(incl[id(c)] for c in r[3])
        if r[4] is not None:
            a["bytes"] += r[4][0]
            a["flops"] += r[4][1]
            a["estimated_calls"] += 1
    pk = peaks if peaks is not None else (device_peaks() if gpu else None)
    phases: dict = {}
    for lab, a in agg.items():
        p = dict(calls=a["calls"] / steps, inclusive=a["inclusive"] / steps,
                 exclusive=a["exclusive"] / steps)
        if a["estimated_calls"] == a["calls"] and a["exclusive"] > 0 and a["bytes"] > 0:
            t = a["exclusive"]
            by, fl = a["bytes"] / steps, a["flops"] / steps
            rf = dict(bytes=by, flops=fl, intensity=fl / by, achieved_GBps=by / (t / steps) / 1e9,
                      achieved_GFlops=fl / (t / steps) / 1e9, fraction_of_roof=None, bound=None)
            if pk is not None and math.isfinite(pk.fp64):
                roof = min(pk.fp64, rf["intensity"] * pk.bandwidth)
                rf["attainable_GFlops"] = roof / 1e9
                rf["fraction_of_roof"] = (fl / (t / steps)) / roof if roof > 0 and fl > 0 else \
                    (by / (t / steps)) / pk.bandwidth
                rf["bound"] = "memory" if rf["intensity"] < pk.ridge else "compute"
            p["roofline"] = rf
        phases[lab] = p
    groups = {g: sum(phases[p]["exclusive"] for p in labs if p in phases) for g, labs in GROUPS.items()}
    groups = {g: v for g, v in groups.items() if v > 0}
    rep = ProfileReport(backend=solver.backend, device=(pk.name if pk is not None else "cpu"),
                        steps=int(steps), wall_per_step=wall, wall_per_step_unprobed=wall_un,
                        phases=phases, groups=groups, peaks=pk)
    rep.measured = ["wall_per_step", "wall_per_step_unprobed", "phases.*.inclusive", "phases.*.exclusive",
                    "groups"]
    rep.estimated = ["phases.*.roofline.bytes", "phases.*.roofline.flops", "phases.*.roofline.intensity",
                     "phases.*.roofline.achieved_GBps", "phases.*.roofline.achieved_GFlops",
                     "phases.*.roofline.fraction_of_roof", "phases.*.roofline.bound"]
    if pk is not None:
        rep.declared = ["peaks.bandwidth", "peaks.fp64", "phases.*.roofline.attainable_GFlops"]
    rep.notes.append("GPU spans are stream-timeline intervals between CUDA events: they include the gaps "
                     "in which the stream waited for the host (launch-bound phases)" if gpu else
                     "CPU backend: host wall-clock spans (time.perf_counter)")
    rep.notes.append("%roof is against the declared DRAM bandwidth: above 100 % means the phase's data "
                     "stayed in the L2 cache (the byte estimate counts every access as DRAM traffic)")
    rep.notes.append("roofline only for phases whose every call has an estimate; F.fused, F, F.scatter, "
                     "F.gather exclusive parts are not estimated")
    if restore and saved is not None:
        from collections import deque
        solver.X = saved[0]
        solver._MXh, solver._Fh = deque(saved[1]), deque(saved[2])
        if hasattr(solver, "_LXh"):
            solver._LXh = deque(saved[8])
        solver.t, solver.iteration = saved[3], saved[4]
        if saved[5] is not None:
            solver._t_origin, solver._it_origin = saved[5], saved[6]
        solver.n_F = saved[7]
    return rep
