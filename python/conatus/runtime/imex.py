"""IMEX integrator executed in C++: M dX/dt + L X = F(X, t) (Dedalus conventions)."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp

from . import _core
from .banded import csr_args

__all__ = ["ImexIntegrator", "SCHEMES"]

SCHEMES = ("SBDF1", "SBDF2", "SBDF3", "SBDF4", "RK111", "RK222", "RK443")


class ImexIntegrator:
    """Constant-step IMEX integrator. The step loop runs in C++; F may be a Python callback
    ``F(x, t) -> array`` (tests / prototypes) or None (F = 0).

    Nothing is chosen for the user: scheme and dt are required arguments.
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
        dt_ = dtype or np.result_type(M.dtype, L.dtype)
        self.dtype = np.complex128 if np.issubdtype(dt_, np.complexfloating) else np.float64
        cls = _core.ImexZ if self.dtype == np.complex128 else _core.ImexD
        n, mp, mi, md = csr_args(M, self.dtype)
        _, lp, li, ld = csr_args(L, self.dtype)
        self.scheme, self.dt, self.border_rows = scheme, float(dt), [int(i) for i in border_rows]
        self._s = cls(n, mp, mi, md, lp, li, ld, self.border_rows, scheme, self.dt)
        self._F = F
        if F is not None:
            dtype_ = self.dtype                       # no reference to self: avoids a C++ <-> Python cycle
            self._s.set_rhs(lambda x, t: np.ascontiguousarray(F(np.asarray(x), t), dtype=dtype_))

    n_factorizations = property(lambda self: self._s.n_factorizations)
    iteration = property(lambda self: self._s.iteration)

    def run(self, x0: np.ndarray, t0: float, nsteps: int) -> tuple[np.ndarray, float]:
        """(x, t) after ``nsteps`` steps from (x0, t0), the loop in C++ (a float number of steps
        is refused, never truncated)."""
        from ..timestep.integrator import check_steps
        nsteps = check_steps(nsteps, "ImexIntegrator.run")
        x, t = self._s.run(np.ascontiguousarray(x0, dtype=self.dtype), float(t0), nsteps)
        return np.asarray(x), t

    def explain(self) -> str:
        """Scheme, step, F and the cached factorizations."""
        return (f"ImexIntegrator(scheme={self.scheme}, dt={self.dt:g}, constant step, "
                f"F={'callback' if self._F else 'none'}, factorizations cached={self.n_factorizations}); "
                "LHS a0 M + L (SBDF) or M + dt H_ii L (RK), bordered banded solve in C++")
