"""IMEX integrator executed in C++: M dX/dt + L X = F(X, t) (Dedalus conventions)."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp

from . import _core
from .banded import _factor_dtype, csr_args, solve_operand

__all__ = ["ImexIntegrator", "SCHEMES"]

SCHEMES = ("SBDF1", "SBDF2", "SBDF3", "SBDF4", "RK111", "RK222", "RK443")


class ImexIntegrator:
    """Constant-step IMEX integrator. The step loop runs in C++; F may be a Python callback
    ``F(x, t) -> array`` (tests / prototypes) or None (F = 0).

    Nothing is chosen for the user: scheme and dt are required arguments.

    The dtype follows M and L (complex if either is), or ``dtype``; a complex state, matrix or
    F(x, t) for a real integrator is refused, never cut to its real part.
    SBDF keeps a history: a second :meth:`run` must continue from the (x, t) the previous one
    returned; :meth:`reset` starts a new trajectory.
    """

    def __init__(self, M, L, border_rows: Sequence[int], scheme: str, dt: float,
                 F: Callable[[np.ndarray, float], np.ndarray] | None = None, dtype=None):
        from ..errors import OptionError, did_you_mean
        from ..timestep.integrator import check_dt_backend
        if scheme not in SCHEMES:
            raise OptionError(f"ImexIntegrator: unknown scheme {scheme!r}",
                              fix=f"one of {SCHEMES}." + did_you_mean(str(scheme), SCHEMES))
        check_dt_backend(dt, "cpu", "ImexIntegrator")
        M, L = sp.csr_matrix(M), sp.csr_matrix(L)
        self.dtype = _factor_dtype([M, L], dtype)
        cls = _core.ImexZ if self.dtype == np.complex128 else _core.ImexD
        n, mp, mi, md = csr_args(M, self.dtype, "ImexIntegrator (M)")
        _, lp, li, ld = csr_args(L, self.dtype, "ImexIntegrator (L)")
        self.scheme, self.dt, self.border_rows = scheme, float(dt), [int(i) for i in border_rows]
        self._s = cls(n, mp, mi, md, lp, li, ld, self.border_rows, scheme, self.dt)
        self._F = F
        self._last = None                             # (x bytes, t) returned by the last run (SBDF)
        if F is not None:
            # no reference to self (the core owns this callable; a cycle through it is visible
            # to the garbage collector anyway since finding 43, but none is needed here)
            dtype_ = self.dtype
            self._s.set_rhs(lambda x, t: solve_operand(F(np.asarray(x), t), dtype_,
                                                       "ImexIntegrator: F(x, t)"))

    n_factorizations = property(lambda self: self._s.n_factorizations)
    iteration = property(lambda self: self._s.iteration)

    def run(self, x0: np.ndarray, t0: float, nsteps: int) -> tuple[np.ndarray, float]:
        """(x, t) after ``nsteps`` steps from (x0, t0), the loop in C++ (a float number of steps
        is refused, never truncated)."""
        from ..timestep.integrator import check_steps
        from ..errors import OptionError
        nsteps = check_steps(nsteps, "ImexIntegrator.run")
        x0 = solve_operand(x0, self.dtype, "ImexIntegrator.run: x0")
        if self.history_length and self._last != (x0.tobytes(), float(t0)):
            raise OptionError(f"ImexIntegrator.run: {self.scheme} holds the history of the previous "
                              "run, and (x0, t0) is not the state and time it returned",
                              why="the multistep history of another trajectory would enter this one "
                                  "silently (review finding 42)",
                              fix="call reset() to start a new trajectory (lower-order startup "
                                  "again), or pass the (x, t) returned by the previous run")
        x, t = self._s.run(x0, float(t0), nsteps)
        x = np.asarray(x)
        self._last = (x.tobytes(), float(t)) if self.history_length else None
        return x, t

    history_length = property(lambda self: self._s.history_length)

    def reset(self) -> None:
        """Forget the multistep history and the iteration count; the factorizations are kept
        (they depend on M, L, dt and the scheme only). The next :meth:`run` starts a new
        trajectory from its x0 with the lower-order startup, as a fresh integrator would."""
        self._s.reset()
        self._last = None

    def explain(self) -> str:
        """Scheme, step, F and the cached factorizations."""
        return (f"ImexIntegrator(scheme={self.scheme}, dt={self.dt:g}, constant step, "
                f"F={'callback' if self._F else 'none'}, factorizations cached={self.n_factorizations}, "
                f"history levels={self.history_length}: a next run() continues from the returned "
                "(x, t), reset() starts a new trajectory); "
                "LHS a0 M + L (SBDF) or M + dt H_ii L (RK), bordered banded solve in C++")
