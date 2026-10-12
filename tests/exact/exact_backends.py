"""Backend matrix shared by the closed-form tests of tests/exact/ (audit holes 1 and 2).

One IVP is run through every time-stepping path Conatus has, and each path is compared with the
*same closed form* (never with another Conatus path):

  cpu            IVPSolver(backend="cpu")                       evaluator F
  cpu-fused      IVPSolver(backend="cpu", f_backend="fused")    fused F (C++)
  cuda           IVPSolver(backend="cuda")                      device solves, evaluator F
  cuda-fused     IVPSolver(backend="cuda", f_backend="fused")   fused F on the GPU
  cuda-graph     IVPSolver(backend="cuda").run_graph(n)         captured CUDA graph (RK or SBDF)

The CUDA modes carry the ``gpu`` marker and are skipped without a device; they run in a GPU
"""
from __future__ import annotations

import pytest

from conatus.timestep import IVPSolver


def _cuda_ok() -> bool:
    """True when the compiled core sees a CUDA device (import errors count as no device)."""
    try:
        from conatus.runtime import cuda
        return bool(cuda.available())
    except Exception:           # no compiled CUDA module on this node: the gpu cases are skipped
        return False


_GPU = [pytest.mark.gpu, pytest.mark.skipif(not _cuda_ok(), reason="no CUDA device on this node")]

MODES = ["cpu", "cpu-fused",
         pytest.param("cuda", marks=_GPU), pytest.param("cuda-fused", marks=_GPU),
         pytest.param("cuda-graph", marks=_GPU)]


def make_solver(plan, scheme: str, dt: float, mode: str, **kw) -> IVPSolver:
    """IVPSolver for one mode of the matrix above (every option explicit, nothing inferred)."""
    backend = "cuda" if mode.startswith("cuda") else "cpu"
    f_backend = "fused" if mode.endswith("fused") else "evaluator"
    return IVPSolver(plan, scheme, dt, backend=backend, f_backend=f_backend, **kw)


def advance(s: IVPSolver, nsteps: int, mode: str) -> None:
    """Advance ``nsteps`` steps; the graph mode replays a captured step in two chunks (so the
    replay restarts from written-back buffers once, the way a production run does)."""
    if mode == "cuda-graph":
        first = nsteps // 2
        s.run_graph(first)
        s.run_graph(nsteps - first)
    else:
        s.run(nsteps)
    assert s.iteration == nsteps
