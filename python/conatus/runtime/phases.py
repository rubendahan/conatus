"""Phase probes on a running ``IVPSolver``: who is executing, for profiling and memory tracing.

A :class:`PhaseProbe` temporarily wraps the entry points of one solver's step (instance attributes
and proxies only; ``timestep/integrator.py`` and ``timestep/fused.py`` are not modified) and calls
``on_enter(label)`` / ``on_exit(label)`` around each of them. Everything is restored on exit, so a
probed solver computes exactly the same numbers (the wrappers never touch arrays) with the same
kernels and allocations: code that dispatches on a matrix type sees the wrapped matrix through
``_MatProxy.probed_target`` (before that, a probed GPU RK step ran another program, gpu4).

Phase labels (a call inside another phase is nested; a phase never nests inside itself):

========================  ================================================================
``step``                  one time step (:meth:`IVPSolver.step`)
``matvec``                ``M @ X`` and ``L @ X`` (device CSR or SciPy); on the GPU a real RK step
                          forms ``M x0`` and ``L X_k`` inside its right-hand-side kernel: timed in ``step``
``solve``                 ``system.solve``: gather ``rhs[rows]``, banded solves, scatter
``solve.banded``          the batched bordered banded kernels only (``solve_ptr``, GPU)
``F``                     explicit right-hand side ``solver.F``
``F.scatter``             state vector -> coefficient arrays of the RHS unknowns
``F.fused``               fused polynomial RHS (``FusedRHS.evaluate``)
``F.derivatives``         factor coefficients: embedding and spectral derivatives
``F.transforms``          grid <-> coefficient transforms (``xp_transforms``, direct FFTs)
``F.products``            the generated pointwise kernel (products and sums, fused)
``F.evaluator``           oracle evaluator (``f_backend='evaluator'``), not split further
``F.gather``              equation coefficient arrays -> stacked right-hand side
========================  ================================================================

What is *not* inside a child phase is attributed to the parent's exclusive part: for ``step`` it is
the IMEX vector arithmetic (axpy, copies of the RK combinations), for ``solve`` the gather/scatter
copies, for ``F.fused`` the stacking of factors, real projections and output embedding.
"""
from __future__ import annotations

from typing import Any
from collections.abc import Callable

__all__ = ["PhaseProbe", "PHASES"]

PHASES = ("step", "matvec", "solve", "solve.banded", "F", "F.scatter", "F.fused", "F.derivatives",
          "F.transforms", "F.products", "F.evaluator", "F.gather")


class _MatProxy:
    """Stands for ``system.M`` / ``system.L``: ``A @ X`` is timed, everything else delegated
    (operator dispatch looks up ``__matmul__`` on the type, so an instance attribute is not enough)."""

    def __init__(self, A, probe: PhaseProbe) -> None:
        self._A, self._probe = A, probe
        # the wrapped matrix, for code that dispatches on the matrix type (lincomb: M x0 and L X_k
        # inside the RK right-hand-side kernel); an in-kernel product is then timed with "step"
        self.probed_target = A

    def __matmul__(self, B):
        return self._probe._call("matvec", self._A.__matmul__, B)

    def __getattr__(self, name):
        return getattr(self._A, name)


class _BandedProxy:
    """Stands for a ``BandedBatch._s`` (nanobind object): ``solve_ptr`` / ``solve`` are timed."""

    def __init__(self, s, probe: PhaseProbe) -> None:
        self._s, self._probe = s, probe

    def solve_ptr(self, *a):
        return self._probe._call("solve.banded", self._s.solve_ptr, *a)

    def solve(self, *a):
        return self._probe._call("solve.banded", self._s.solve, *a)

    def solve_mapped_ptr(self, *a):          # gather/scatter fused in the kernel
        return self._probe._call("solve.banded", self._s.solve_mapped_ptr, *a)

    def solve_split_ptr(self, *a):           # complexified pairs, split in the kernel
        return self._probe._call("solve.banded", self._s.solve_split_ptr, *a)

    def __getattr__(self, name):
        return getattr(self._s, name)


class _KernelProxy:
    """Stands for ``FusedRHS._kernel`` (CuPy RawKernel): launches are labelled ``F.products``."""

    def __init__(self, k, probe: PhaseProbe) -> None:
        self._k, self._probe = k, probe

    def __call__(self, *a, **kw):
        return self._probe._call("F.products", self._k, *a, **kw)

    def __getattr__(self, name):
        return getattr(self._k, name)


class PhaseProbe:
    """Context manager installing phase wrappers on ``solver`` (see module docstring).

    on_enter(label, fn, args, kwargs) / on_exit(label) are called in program order (``fn``: the
    wrapped callable, e.g. a bound method whose ``__self__`` describes the work). ``stack`` holds the labels
    currently open. Banded proxies are installed on the factorizations existing at entry and on
    those created inside the probe (``system.factor`` is wrapped)."""

    def __init__(self, solver, on_enter: Callable[[str, Any, tuple, dict], None],
                 on_exit: Callable[[str], None]) -> None:
        self.solver = solver
        self.on_enter, self.on_exit = on_enter, on_exit
        self.stack: list[str] = []
        self._undo: list[Callable[[], None]] = []

    # ------------------------------------------------------------------ core
    def _call(self, label: str, fn, *a, **kw):
        if label in self.stack:                     # never nest a phase inside itself
            return fn(*a, **kw)
        self.stack.append(label)
        self.on_enter(label, fn, a, kw)
        try:
            return fn(*a, **kw)
        finally:
            self.on_exit(label)
            self.stack.pop()

    def _wrap_attr(self, obj, name: str, label: str) -> None:
        """Replace ``obj.name`` (bound method or module function) by a labelled wrapper."""
        orig = getattr(obj, name)
        had = name in getattr(obj, "__dict__", {})
        prev = obj.__dict__[name] if had else None

        def wrapped(*a, **kw):
            return self._call(label, orig, *a, **kw)
        setattr(obj, name, wrapped)
        self._undo.append(lambda: setattr(obj, name, prev) if had else delattr(obj, name))

    def _set(self, obj, name: str, value) -> None:
        prev = getattr(obj, name)
        setattr(obj, name, value)
        self._undo.append(lambda: setattr(obj, name, prev))

    # ------------------------------------------------------------------ install / restore
    def __enter__(self) -> PhaseProbe:
        s = self.solver
        sysm = s.system
        self._wrap_attr(s, "step", "step")
        self._wrap_attr(s, "F", "F")
        self._wrap_attr(sysm, "solve", "solve")
        self._wrap_attr(sysm, "scatter", "F.scatter")
        self._wrap_attr(sysm, "gather_rhs", "F.gather")
        self._set(sysm, "M", _MatProxy(sysm.M, self))
        self._set(sysm, "L", _MatProxy(sysm.L, self))
        self._probe_factors()
        orig_factor = sysm.factor

        def factor(a, b):                     # new factorizations get banded proxies too
            f = orig_factor(a, b)
            self._probe_factor_list(f)
            return f
        sysm.factor = factor
        self._undo.append(lambda: delattr(sysm, "factor"))
        fu = getattr(s, "_fused", None)
        if getattr(s, "_fc2d", None) is not None:   # 2D pipeline: F without the fused sub-phases
            self._wrap_attr(s._fc2d, "evaluate", "F.fused")
        elif fu is not None:
            self._wrap_attr(fu, "evaluate", "F.fused")
            # StageTimeFused (time-dependent RHS) wraps the compiled FusedRHS in .inner: its
            # sub-phases are the inner kernel's (gpu1)
            fu = getattr(fu, "inner", fu)
            self._wrap_attr(fu, "_diff", "F.derivatives")
            self._wrap_attr(fu, "_full", "F.derivatives")
            if getattr(fu, "_kernel", None) is not None:
                self._set(fu, "_kernel", _KernelProxy(fu._kernel, self))
            from ..timestep import xp_transforms as XT
            self._wrap_attr(XT, "forward", "F.transforms")
            self._wrap_attr(XT, "backward", "F.transforms")
            fft = getattr(fu.xp, "fft", None)
            if fft is not None:                   # direct FFTs of the parity-reduced path
                for name in ("rfft", "irfft"):
                    self._wrap_attr(fft, name, "F.transforms")
        else:
            self._wrap_attr(s.evaluator, "evaluate", "F.evaluator")
        return self

    def _probe_factors(self) -> None:
        for f in getattr(self.solver.system, "_factors", {}).values():
            self._probe_factor_list(f)

    def _probe_factor_list(self, f: Any) -> None:
        for batch in f:
            s = getattr(batch, "_s", None)
            if s is not None and not isinstance(s, _BandedProxy) and hasattr(s, "solve_ptr"):
                self._set(batch, "_s", _BandedProxy(s, self))

    def __exit__(self, *exc) -> None:
        while self._undo:
            self._undo.pop()()
