"""Measured device memory of an ``IVPSolver(backend='cuda')``: the reference the memory
planner (:mod:`conatus.runtime.memory`) is compared to. Nothing here predicts anything.

Three measurements, all reported, never merged:

* **CuPy pool trace** — a :class:`cupy.cuda.MemoryHook` records every pool ``malloc``/``free``
  (rounded size, pointer) and every new device allocation of the pool (``alloc``: the pool grows),
  each tagged with the open phase (:class:`~conatus.runtime.phases.PhaseProbe`) and the step
  index. From it: peak of live bytes, per-phase peaks, bytes held by the pool.
* **Core pool** — the C++ caching pool of ``conatus._core`` (banded factors): live + cached bytes,
  sampled at phase exits (it has no hook).
* **Driver view** — ``cudaMemGetInfo`` delta and the per-process ``nvidia-smi`` figure (they also
  contain the CUDA context, cuFFT/cuBLAS handles and kernel images, which no model predicts).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .phases import PhaseProbe

__all__ = ["DeviceMemoryTrace", "trace_device_memory", "nvidia_smi_process_bytes"]


def nvidia_smi_process_bytes(pid: int | None = None) -> int | None:
    """Device memory of process ``pid`` as reported by ``nvidia-smi`` (None if unavailable)."""
    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(float(parts[1]) * 2**20)          # nvidia-smi reports MiB
    return None


@dataclass
class DeviceMemoryTrace:
    """Event log and summary of one traced run (bytes). ``events``: (kind, bytes, ptr, phase, step)
    with kind 'm' (pool malloc), 'f' (pool free), 'a' (pool grows: new device allocation)."""
    events: list = field(default_factory=list)
    core_samples: list = field(default_factory=list)     # (phase, step, live, cached)
    summary: dict = field(default_factory=dict)

    def phase_peaks(self) -> dict:
        """Peak of CuPy live bytes while each phase was the innermost open phase."""
        live, peaks = 0, {}
        for kind, b, _p, ph, _s in self.events:
            if kind == "m":
                live += b
            elif kind == "f":
                live -= b
            else:
                continue
            peaks[ph] = max(peaks.get(ph, 0), live)
        return peaks

    def growth_by_phase(self) -> dict:
        """Bytes of new device allocations of the CuPy pool, per phase (where the pool grew)."""
        g: dict = {}
        for kind, b, _p, ph, _s in self.events:
            if kind == "a":
                g[ph] = g.get(ph, 0) + b
        return g


def trace_device_memory(make_solver, set_state, steps: int, *, warmup: int = 0,
                        collect_cycles_after_F: bool = False) -> tuple[Any, DeviceMemoryTrace]:
    """Build the solver with ``make_solver()`` under the hook (setup allocations are traced with
    phase 'setup'), call ``set_state(solver)`` (phase 'set_state'), take ``warmup + steps`` steps.
    Returns (solver, trace). The default CuPy pool must be fresh for the numbers to mean anything
    (run one configuration per process).

    ``collect_cycles_after_F=True`` runs ``gc.collect()`` when ``FusedRHS.evaluate`` returns: the
    derivative memo (a reference cycle, see) is then freed where the
    one-line fix would free it. This is a *declared measurement mode* (it emulates the fixed
    runtime); the default measures the runtime as it is."""
    import gc
    import cupy as cp
    from cupy.cuda import memory_hook

    from . import cuda

    tr = DeviceMemoryTrace()
    state = {"phase": "setup", "step": -1}
    stack: list = []

    class Hook(memory_hook.MemoryHook):
        name = "ConatusTraceHook"

        def alloc_postprocess(self, **kw):
            tr.events.append(("a", int(kw["mem_size"]), int(kw["mem_ptr"]), state["phase"], state["step"]))

        def malloc_postprocess(self, **kw):
            tr.events.append(("m", int(kw["mem_size"]), int(kw["mem_ptr"]), state["phase"], state["step"]))

        def free_postprocess(self, **kw):
            tr.events.append(("f", int(kw["mem_size"]), int(kw["mem_ptr"]), state["phase"], state["step"]))

    def core_sample(ph):
        st = cuda.pool_stats()
        tr.core_samples.append((ph, state["step"], int(st["live_bytes"]), int(st["cached_bytes"])))

    def enter(label, _fn, _a, _kw):
        stack.append(label)
        state["phase"] = label

    def leave(label):
        if collect_cycles_after_F and label == "F.fused":
            gc.collect()
        stack.pop()
        state["phase"] = stack[-1] if stack else "between"
        if label in ("solve", "step"):
            core_sample(label)

    cp.zeros(1)                                       # context + pool exist before the baseline
    cp.cuda.Device().synchronize()
    free0, total = cp.cuda.runtime.memGetInfo()
    core0 = cuda.pool_stats()
    pool = cp.get_default_memory_pool()
    held0 = int(pool.total_bytes())
    with Hook():
        solver = make_solver()
        core_sample("setup")
        state["phase"] = "set_state"
        set_state(solver)
        state["phase"] = "between"
        with PhaseProbe(solver, enter, leave):
            for k in range(warmup + steps):
                state["step"] = k
                solver.step()
        cp.cuda.Device().synchronize()
    free1, _ = cp.cuda.runtime.memGetInfo()
    core1 = cuda.pool_stats()
    live, peak, peak_steady = 0, 0, 0
    for kind, b, _p, _ph, st in tr.events:
        if kind == "m":
            live += b
        elif kind == "f":
            live -= b
        else:
            continue
        peak = max(peak, live)
        if st >= warmup:
            peak_steady = max(peak_steady, live)
    tr.summary = dict(
        cupy_pool_held_bytes=int(pool.total_bytes()) - held0,
        cupy_pool_used_bytes=int(pool.used_bytes()),
        cupy_peak_live_bytes=peak, cupy_peak_live_steady_bytes=peak_steady,
        cupy_live_end_bytes=live,
        core_live_bytes=int(core1["live_bytes"]) - int(core0["live_bytes"]),
        core_cached_bytes=int(core1["cached_bytes"]) - int(core0["cached_bytes"]),
        memgetinfo_delta_bytes=int(free0 - free1), device_total_bytes=int(total),
        nvidia_smi_process_bytes=nvidia_smi_process_bytes(),
        steps=int(steps), warmup=int(warmup), collect_cycles_after_F=bool(collect_cycles_after_F))
    return solver, tr
