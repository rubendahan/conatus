"""Ensembles: a batch of initial conditions and/or parameter values advanced together.

``Ensemble(solver, B)`` advances B copies of the discretized problem of ``solver`` from B initial
states; ``Ensemble.from_solvers([...])`` builds a parameter ensemble from one solver per parameter
set (same discretization, scheme and step; only parameter values differ). The step is the one of
:class:`IVPSolver` — same IMEX scheme, same coefficients, same start-up of multistep schemes, same
order of operations — applied to arrays (B, n): one F evaluation, one sparse product and one
stacked solve per stage for the whole ensemble (``conatus.ensembles.batched``).

Agreement with single runs (measured in tests/ensembles, never assumed):

* ``rhs="loop"`` (oracle, not vectorized): F member by member with ``IVPSolver.F``; the rest of the
  step is batched. Every member is **bit-identical** to its single run on the CPU.
* ``rhs="batched"`` (the vectorized path): F in the fused polynomial/general form; each step agrees
  with the single run **to rounding** (relative 1e-15 to 1e-14 per step); on a chaotic trajectory
  that difference grows like any rounding difference (Lyapunov time), so members of a chaotic
  ensemble agree with single runs statistically, not pointwise, after a few Lyapunov times.
  A member of a batch is bit-identical to the same member run alone through the same batched path
  (B = 1) on the CPU (tested).

Everything is explicit: scheme, dt, start-up and backend come from the solver(s), the RHS path is
an argument, and ``explain()`` prints the choices (replicated factors, per-member coefficients).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..errors import OptionError
from ..timestep.integrator import (check_callback, check_params_unchanged, check_steps,
                                   require_family, rk_tableau, sbdf_coefficients)
from .batched import BatchedRHS, BatchedSystem
from .._deprecation import renames

__all__ = ["Ensemble"]


class Ensemble:
    """B members of one discretized problem advanced together (see module docstring)."""

    def __init__(self, solver, size: int, *, rhs: str) -> None:
        from ..errors import check_count
        self._setup([solver], np.zeros(check_count(size, "size", "Ensemble"), int), rhs)

    @classmethod
    def from_solvers(cls, solvers: Sequence, *, rhs: str, member_of: Sequence[int] | None = None
                     ) -> Ensemble:
        """Parameter ensemble: member b uses ``solvers[member_of[b]]`` (default: one member per
        solver). The solvers must share the discretization, scheme, dt, start-up and backend."""
        self = cls.__new__(cls)
        if member_of is None:
            mo = np.arange(len(solvers))
        else:
            raw = np.asarray(member_of)
            mo = raw.astype(int)
            # indices are never truncated (finding 66: 1.7 became 1)
            if raw.dtype == bool or not np.array_equal(mo, raw) or raw.ndim != 1 \
                    or (mo.size and (mo.min() < 0 or mo.max() >= len(solvers))):
                raise OptionError(f"Ensemble.from_solvers: member_of={member_of!r} must be a 1-D "
                                  f"sequence of integer indices in [0, {len(solvers)})",
                                  fix="member_of=[0, 0, 1, 1], for example")
        self._setup(list(solvers), mo, rhs)
        return self

    def _setup(self, solvers: list, member_of: np.ndarray, rhs: str) -> None:
        if rhs not in ("batched", "loop"):
            raise OptionError(f"Ensemble: rhs must be 'batched' or 'loop', got {rhs!r}",
                              why="the RHS path decides whether members are bit-identical to "
                                  "single runs (loop) or vectorized (batched)",
                              fix="rhs='batched' (vectorized) or rhs='loop' (oracle)")
        s0 = solvers[0]
        # The ensemble step mirrors the SBDF and RK arithmetic only: a CN multistep solver would
        # otherwise be stepped as RK with no tableau (or as SBDF); refused, never substituted.
        require_family(s0, ("sbdf", "rk"), "Ensemble",
                       fix="build the member solvers with an SBDF1-4 or IMEX Runge–Kutta scheme "
                           "(RK111, RK222, RK443, RKSMR, RKGFY); the CN multistep family "
                           "(CNAB1/2, MCNAB2, CNLF2) runs with IVPSolver only")
        for s in solvers[1:]:
            for a in ("scheme", "dt", "startup", "startup_substeps", "backend"):
                if getattr(s, a) != getattr(s0, a):
                    raise OptionError(f"Ensemble.from_solvers: solvers differ in {a} "
                                      f"({getattr(s, a)!r} vs {getattr(s0, a)!r})",
                                      fix="build every member's solver with the same numerics")
        if len(member_of) < 1:
            raise OptionError("Ensemble: at least one member")
        self.solvers, self.member_of = solvers, member_of
        self.B = len(member_of)
        self.scheme, self.dt = s0.scheme, s0.dt
        self.startup, self.startup_substeps = s0.startup, s0.startup_substeps
        self.order = s0.order
        self._rk = s0._rk
        self.backend = s0.backend
        self.xp = s0.xp
        self.rhs_mode = rhs
        self.sys = BatchedSystem([s.system for s in solvers], member_of)
        self.rhs = BatchedRHS(solvers, member_of, mode=rhs)
        self.n = self.sys.n
        self.dtype = self.sys.dtype
        self.X = self.xp.zeros((self.B, self.n), dtype=self.dtype)
        self.t, self.iteration = 0.0, 0
        self._MXh: list = []
        self._Fh: list = []
        self.n_F = 0

    # ------------------------------------------------------------------ state
    def set_states(self, X, t: float = 0.0) -> None:
        """Initial states (B, n) (global vectors, e.g. ``solver.X`` of set_state calls); resets
        the multistep history."""
        X = self.xp.asarray(X, dtype=self.dtype)
        if X.shape != (self.B, self.n):
            raise OptionError(f"Ensemble.set_states: expected shape {(self.B, self.n)}, got {X.shape}")
        self.X = X.copy()
        self.t, self.iteration, self._t0 = float(t), 0, float(t)
        self._MXh, self._Fh = [], []

    def F(self, X):
        self.n_F += 1
        return self.rhs(X)

    # ------------------------------------------------------------------ steps
    @renames("conatus.ensembles.ensemble:Ensemble.rk_stages")
    def rk_stages(self, x0, dt: float, tableau) -> list:
        """Stage states of one IMEX RK step for the batch: same arithmetic, same order as
        ``IVPSolver.rk_stages`` (stage j's F and L X computed only if a later stage uses them)."""
        A, H, c = tableau
        s = len(c) - 1
        sysm = self.sys
        X = [x0]
        Fs: list = [None] * (s + 1)
        LX: list = [None] * (s + 1)
        MX0 = sysm.M(x0)
        for i in range(1, s + 1):
            j = i - 1
            if any(A[m][j] != 0 for m in range(j + 1, s + 1)):
                Fs[j] = self.F(X[j])
            if any(H[m][j] != 0 for m in range(j + 1, s + 1)):
                LX[j] = sysm.L(X[j])
            rhs = MX0.copy()
            for k in range(i):
                if A[i][k] != 0:
                    rhs += (dt * A[i][k]) * Fs[k]
                if H[i][k] != 0:
                    rhs -= (dt * H[i][k]) * LX[k]
            X.append(sysm.solve(1.0, dt * H[i][i], rhs))
        return X

    def _sbdf_step(self, q_max: int) -> None:
        """Mirror of ``IVPSolver._sbdf_step`` (history of M X and F, newest first)."""
        sysm, dt = self.sys, self.dt
        self._MXh.insert(0, sysm.M(self.X))
        self._Fh.insert(0, self.F(self.X))
        del self._MXh[self.order:], self._Fh[self.order:]
        q = min(q_max, len(self._MXh))
        a, c = sbdf_coefficients(q)
        rhs = self.xp.zeros((self.B, self.n), dtype=self.dtype)
        for j in range(1, q + 1):
            rhs += c[j] * self._Fh[j - 1] - (a[j] / dt) * self._MXh[j - 1]
        self.X = sysm.solve(a[0] / dt, 1.0, rhs)

    def step(self) -> None:
        """One step for every member (mirror of ``IVPSolver.step``)."""
        for k, s in enumerate(self.solvers):    # the members never pass through IVPSolver.step:
            check_params_unchanged(s, f"Ensemble.step (parameter set {k})")   # same check (A23)
        if self.order:
            if self.startup == "rk443" and self.iteration < self.order - 1:
                self._MXh.insert(0, self.sys.M(self.X))
                self._Fh.insert(0, self.F(self.X))
                h = self.dt / self.startup_substeps
                X = self.X
                for _ in range(self.startup_substeps):
                    X = self.rk_stages(X, h, rk_tableau("RK443"))[-1]
                self.X = X
            else:
                self._sbdf_step(self.order)
        else:
            self.X = self.rk_stages(self.X, self.dt, self._rk)[-1]
        self.iteration += 1
        self.t = self._t0 + self.iteration * self.dt

    @renames("conatus.ensembles.ensemble:Ensemble.run")
    def run(self, steps: int, callback: Callable[[Ensemble], None] | None = None,
            every: int | None = None) -> None:
        """Advance ``steps`` steps; ``callback(self)`` after every step whose iteration is a
        multiple of ``every`` (both explicit, as in ``IVPSolver.run``)."""
        check_callback(callback, every, "Ensemble.run")
        for _ in range(check_steps(steps, "Ensemble.run", "steps")):
            self.step()
            if callback is not None and self.iteration % every == 0:
                callback(self)

    # ------------------------------------------------------------------ output
    def states(self) -> np.ndarray:
        """Host copy of the member states (B, n): a copy on every backend (finding 68: on NumPy the live state array was returned, so writing to it changed the
        ensemble)."""
        return self.X.copy() if self.xp is np else self.xp.asnumpy(self.X)

    def explain(self) -> str:
        st = ("" if not self.order else f", startup={self.startup}"
              + (f" ({self.startup_substeps} RK443 substeps)" if self.startup == "rk443" else ""))
        return (f"Ensemble({self.B} member(s), {len(self.solvers)} parameter set(s), "
                f"scheme={self.scheme}, dt={self.dt:g}{st}, backend={self.backend})\n"
                f"  step: IVPSolver arithmetic on (B, n) arrays (same stages, same order)\n"
                f"  {self.rhs.explain()}\n  {self.sys.explain()}")
