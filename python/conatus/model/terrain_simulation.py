"""Time integration of a native IVP on a mapped domain (:class:`TerrainSimulation`, built by
``Model.simulate`` on a :class:`~conatus.model.terrain.MappedBox`). It steps the coupled sparse
operators of :class:`~conatus.discretize.terrain_plan.TerrainPlan`; split out of that module, which
keeps the name importable (a lazy alias: the model layer imports the discretization, not the
reverse)."""
from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla

from ..discretize.compiler import _comps
from ..errors import (DtypeError, InputTypeError, OptionError, RankError, ResolutionError,
                      UnsupportedError)
from ..ir import expr as E
from ..discretize.terrain_plan import _DOC, TerrainPlan


class TerrainSimulation:
    """Time integration of a native IVP on a mapped domain (``Model.simulate``).

    M dX/dt + L X = F(X, t) with L, M the coupled sparse operators (metric implicit) and F the
    right-hand sides evaluated on the dealiased physical grid. Schemes: SBDF1-4 (startup
    ``lower_order``: orders 1..q-1 for the first steps, declared) and RK111/RK222/RK443 (IMEX
    Runge–Kutta, same tableaux as the flat solver). Rows of equations without time derivative
    (boundary conditions, constraints) are imposed at the stage time with their data
    (``bc_time="stage"`` convention). One sparse LU per distinct implicit coefficient."""

    SCHEMES = ("SBDF1", "SBDF2", "SBDF3", "SBDF4", "RK111", "RK222", "RK443")


    # checkpoints (critic review of the invariance audit: these were AttributeErrors)
    def _no_checkpoint(self, what: str):
        from ..errors import UnsupportedError
        raise UnsupportedError(f"{what} on a mapped domain: the mapped-box stepper has no "
                               "checkpoint format yet",
                               why="its state and multistep history are not written by "
                                   "cn.io (an exact restart needs both)",
                               fix="run the mapped case in one job, or record the fields you "
                                   "need with sim.grid(...) / cn.runs.track(...).save_field",
                               doc="docs/site/guide/mapped.md")

    def save(self, *args, **kw):
        """Refused: no checkpoint format for the mapped-box stepper (UnsupportedError)."""
        self._no_checkpoint("save()")

    def restart(self, *args, **kw):
        """Refused: no checkpoint format for the mapped-box stepper (UnsupportedError)."""
        self._no_checkpoint("restart()")

    def checkpoints(self, *args, **kw):
        """Refused: no checkpoint format for the mapped-box stepper (UnsupportedError)."""
        self._no_checkpoint("checkpoints()")

    def resume_or_start(self, *args, **kw):
        """Refused: no checkpoint format for the mapped-box stepper (UnsupportedError)."""
        self._no_checkpoint("resume_or_start()")

    def __init__(self, comp: Any, *, scheme: str, dt: float, backend: str = "cpu",
                 **kw: Any) -> None:
        from .model import ncc_options
        if backend != "cpu":
            raise UnsupportedError(f"backend={backend!r} on a mapped domain",
                                   why="the coupled sparse solve of the mapped discretization runs "
                                       "on the CPU only", fix='backend="cpu"', doc=_DOC)
        if scheme not in self.SCHEMES:
            raise OptionError(f"scheme={scheme!r} on a mapped domain",
                              fix="one of " + ", ".join(self.SCHEMES), doc=_DOC)
        bc_time = kw.pop("bc_time", "auto")
        if bc_time in ("rk", "step"):
            raise UnsupportedError(
                f"bc_time={bc_time!r} on a mapped domain",
                why="the mapped stepper imposes time-dependent boundary data at the stage times "
                    "only (the 'rk' variant of the flat solvers and the Dedalus 'step' convention "
                    "are not implemented here)",
                fix='bc_time="stage" (or leave the default "auto", which resolves to "stage")',
                doc=_DOC)
        if bc_time not in ("auto", "stage"):
            raise OptionError(f"bc_time={bc_time!r}", fix='"auto" or "stage"', doc=_DOC)
        # resolved mode (as solver.bc_time of the flat solvers) and the reason of an "auto" fallback
        self.bc_time = "stage"
        self._bc_note = ("" if bc_time == "stage" else
                         ' (bc_time="auto": the "rk" variant is not implemented on a mapped domain)')
        nkw = ncc_options(kw)
        from ..timestep.integrator import finite_cadence
        # finite-state check: same option and cadence as IVPSolver
        self._finite_every = finite_cadence(kw.pop("check_finite", True), "simulate()")
        extra = set(kw) - {"ncc_cutoff", "ncc_tail_tol"}
        if extra:
            raise OptionError(f"simulate(): option(s) {sorted(extra)} not supported on a mapped "
                              "domain", doc=_DOC)
        # finite: dt=inf passed "dt > 0" (review finding 37)
        if isinstance(dt, bool) or not (isinstance(dt, numbers.Real) and math.isfinite(dt) and dt > 0):
            raise OptionError(f"dt={dt!r}: a finite positive number is required", doc=_DOC)
        self.compiled = comp
        self.problem = comp.problem
        reg = getattr(comp.model, "_register", None)
        if reg is not None:                        # sim.id; named by later parameter changes
            reg(self)
        self.plan: TerrainPlan = comp.discretize(**nkw)
        self.scheme, self._dt = scheme, float(dt)
        P = self.plan
        P.check()
        gm = P.matrices()
        self.L, self.M = gm.L.tocsr(), gm.M.tocsr()
        self.X = np.zeros(P.ncol, dtype=complex)
        self._t, self._it = 0.0, 0
        self._tname = comp.time.name if getattr(comp, "time", None) is not None else None
        if self._tname is not None and any(
                self._tname in {p.name for p in eq.lhs.free_parameters()}
                for eq in self.problem.equations):
            raise UnsupportedError(
                f"a left-hand side depends on the time {self._tname} on a mapped domain",
                why="the coupled operator is factorized once per implicit coefficient; an "
                    "operator that changes in time would need a factorization per stage",
                fix=f"move the {self._tname}-dependent terms to the right-hand side (explicit)",
                doc=_DOC)
        # equations without time derivative: their rows are imposed at the stage time
        self.static = []
        for i, eq in enumerate(self.problem.equations):
            if not any(isinstance(n, E.TimeDerivative) for n in eq.lhs.walk()):
                if E.as_expr(eq.rhs).free_fields() & P.V:
                    raise UnsupportedError(f"equation [{i}] has no time derivative but its right "
                                           "side depends on the unknowns", doc=_DOC)
                self.static.append(i)
        mask = np.zeros(P.nrow, dtype=bool)
        for (i, c), r0 in P.eq_off.items():
            if i in self.static:
                mask[r0:r0 + P.rows(*P.eq_space[(i, c)])] = True
        self._static_rows = mask
        self._lu: dict = {}
        self._hist: list = []         # SBDF history of (X, F)

    # -- state ------------------------------------------------------------------------------
    @property
    def effective_order(self) -> int:
        """Global order of this run (audit invariance; review finding 17 on the flat
        solvers): the scheme's order, capped at 2 for SBDF3/SBDF4 (the start-up is always
        lower-order here) and for RK443 with time-dependent boundary data (imposed at the stage
        times: its stage order 2). Measured on the manufactured mapped diffusion problem of
        tests/model/test_mapped_box.py: SBDF3 2.01, SBDF4 2.03, RK443 1.97."""
        from ..timestep.integrator import _SCHEME_INFO
        p = _SCHEME_INFO[self.scheme][3]
        if self.scheme.startswith("SBDF"):
            p = min(p, 2)
        if self.scheme == "RK443" and self._tname is not None and any(
                self._tname in {q.name for q in E.as_expr(self.problem.equations[i].rhs)
                                .free_parameters()} for i in self.static):
            p = min(p, 2)
        return p

    @property
    def t(self) -> float:
        """Current simulation time."""
        return self._t

    @property
    def dt(self) -> float:
        """Time step (fixed at ``simulate``)."""
        return self._dt

    @property
    def iteration(self) -> int:
        """Number of steps taken."""
        return self._it

    def initial(self, **values: Any) -> TerrainSimulation:
        """Initial state by unknown name: callables of the coordinate names (physical z) or
        numbers; unknowns not given start at zero."""
        from .compiler import _coord_args, _on_all_grids
        P = self.plan
        names = tuple(c.name for c in P.coords)
        grids = P.physical_grid(False)
        for k, v in values.items():
            f = self.compiled.fields.get(k)
            if f is None or f not in P.V:
                raise OptionError(f"initial({k}=...): not an unknown", doc=_DOC)
            fn = (_on_all_grids(v, _coord_args(v, names, f"initial {k}"), names) if callable(v)
                  else (lambda val0: (lambda *g: np.full(grids[0].shape, val0)))(v))
            val = np.asarray(fn(*grids))
            # refused rather than cast or hidden (review finding 36: a deliberate small imaginary
            # part of real data was dropped by grid(); non-finite data as everywhere else)
            if not np.all(np.isfinite(val)):
                raise InputTypeError(f"initial({k}=...): non-finite values on the grid", doc=_DOC)
            if f.dtype.kind != "c" and np.iscomplexobj(val) and np.any(val.imag != 0):
                raise DtypeError(f"initial({k}=...): complex values (max |imag| = "
                                 f"{float(np.abs(val.imag).max()):.3e}) for the real unknown {k}",
                                 fix="pass real data (the unknowns of a native model are real)",
                                 doc=_DOC)
            for c in _comps(f.tensorsig):
                g = np.broadcast_to(val[c] if c else val, grids[0].shape)
                o = P.var_off[(f, c)]
                vec = P.forward(g, False, True)
                self.X[o:o + vec.size] = vec
        self._hist = []
        return self

    def coeffs(self, name: str) -> dict:
        """Coefficients of unknown ``name``: {component: vector of the coupled layout}, copies
        (``plan.split`` gives views of the state: an in-place change of the returned arrays
        used to change the simulation's state)."""
        return {c: v.copy() for c, v in self.plan.split(self.X)[name].items()}

    def grid(self, name: str) -> np.ndarray:
        """Values of unknown ``name`` on the N-point grid (x_j, xi_l)."""
        return self.plan.to_grid(name, self.coeffs(name))

    def coordinate(self, name: str) -> np.ndarray:
        """Physical coordinate ``name`` on the grid of :meth:`grid` (z depends on x)."""
        names = [c.name for c in self.plan.coords]
        if name not in names:
            raise OptionError(f"coordinate({name!r}): not a coordinate of the mapped domain",
                              fix="coordinates: " + ", ".join(names), doc=_DOC)
        return self.plan.physical_grid(False)[names.index(name)]

    # -- stepping ---------------------------------------------------------------------------
    def _F(self, X: np.ndarray, t: float) -> np.ndarray:
        P = self.plan
        if self._tname is not None:
            P.params[self._tname] = t
        P._state = {k: X[o:o + P.rows(*P.var_space[k])] for k, o in P.var_off.items()}
        return P.rhs_vector()

    def _g(self, t: float) -> np.ndarray:
        P = self.plan
        if self._tname is not None:
            P.params[self._tname] = t
        P._state = {k: np.zeros(P.rows(*P.var_space[k]), dtype=complex) for k in P.var_off}
        return P.rhs_vector(lambda i: i in self.static)

    def _solve(self, a: float, b: float, rhs: np.ndarray) -> np.ndarray:
        key = (a, b)
        if key not in self._lu:
            A = (a * self.M + b * self.L).tocsc()
            try:
                self._lu[key] = spla.splu(A)
            except RuntimeError as err:
                raise RankError(f"mapped domain: the implicit system is singular ({err})") from None
        return self._lu[key].solve(rhs)

    def step(self) -> None:
        """One step of the declared scheme (boundary and constraint rows at the stage time)."""
        dt, t = self._dt, self._t
        X = self.X
        if self.scheme.startswith("SBDF"):
            q = min(int(self.scheme[-1]), len(self._hist) + 1)
            from ..timestep.integrator import sbdf_coefficients
            a, c = sbdf_coefficients(q)
            Fn = self._F(X, t)
            hist = [(X, Fn)] + self._hist
            rhs = np.zeros_like(X)
            for j in range(1, q + 1):
                Xj, Fj = hist[j - 1]
                rhs += dt * c[j] * Fj - a[j] * (self.M @ Xj)
            rhs[self._static_rows] = dt * self._g(t + dt)[self._static_rows]
            Xn = self._solve(a[0], dt, rhs)
            self._hist = hist[:3]
        else:
            from ..timestep.integrator import rk_tableau
            A, H, cc = rk_tableau(self.scheme)
            s = len(cc)
            Xs, Fs, LXs = [X], [], []
            MX0 = self.M @ X
            for i in range(1, s):
                Fs.append(self._F(Xs[i - 1], t + cc[i - 1] * dt))
                LXs.append(self.L @ Xs[i - 1])
                rhs = MX0.copy()
                for j in range(i):
                    if A[i][j]:
                        rhs += dt * A[i][j] * Fs[j]
                    if H[i][j] and j < i:
                        rhs -= dt * H[i][j] * LXs[j]
                hii = H[i][i]
                rhs[self._static_rows] = dt * hii * self._g(t + cc[i] * dt)[self._static_rows]
                Xs.append(self._solve(1.0, dt * hii, rhs))
            Xn = Xs[-1]
        self.X = Xn
        self._t = t + dt
        self._it += 1

    def run(self, until: float | None = None, *, steps: int | None = None,
            callback: Callable | None = None, every: int | None = None) -> TerrainSimulation:
        """Advance to ``until`` (a whole number of steps) or by ``steps``; ``callback(sim)`` every
        ``every`` steps."""
        from ..timestep.integrator import check_callback, check_steps
        if (until is None) == (steps is None):     # with a fix line
            raise OptionError("run(): give until= or steps=" + ("" if until is None else ", not both"),
                              fix="run(until=<end time, a whole number of steps of dt>) or "
                                  "run(steps=100)", doc=_DOC)
        # the checks shared by every run (steps=2.7 took 2 steps
        # without a word here, and every=0.5 failed only at the first modulo)
        check_callback(callback, every, "run()")
        if steps is not None:
            steps = check_steps(steps, "run()", "steps")
        if steps is None:
            n = (until - self._t) / self._dt
            steps = int(round(n))
            if steps < 0:                      # (the fix said run(steps=-2))
                # ResolutionError, as Simulation.run (was OptionError here)
                raise ResolutionError(f"run(until={until}): the simulation is already at t = "
                                      f"{self._t:g}, after until",
                                      why="a run goes forward in time; until is an absolute time, "
                                          "not a duration",
                                      fix=f"run(until=t_end) with t_end > {self._t:g}", doc=_DOC)
            from ..timestep.integrator import is_whole    # the one whole-steps rule
            if not is_whole(n):
                raise ResolutionError(f"run(until={until}): not a whole number of steps of dt = "
                                      f"{self._dt:g}", why="the stop time must be a whole number of time steps",
                                      fix=f"run(steps={math.floor(n)})", doc=_DOC)
        from ..timestep.integrator import check_state_finite
        fe = self._finite_every
        for _ in range(steps):
            self.step()
            if fe and self._it % fe == 0:
                check_state_finite(self, "TerrainSimulation.run")
            # on iteration multiples, as IVPSolver.run: the same steps whatever the run chunks
            if callback is not None and self._it % every == 0:
                callback(self)
        if fe and steps:
            check_state_finite(self, "TerrainSimulation.run (end of the call)")
        return self

    def _order_text(self) -> str:
        """The global order said where it is below the scheme's (audit invariance)."""
        from ..timestep.integrator import _SCHEME_INFO
        p, eff = _SCHEME_INFO[self.scheme][3], self.effective_order
        if eff >= p:
            return f"; global order {p}"
        why = ("the lower-order start-up" if self.scheme.startswith("SBDF") else
               "time-dependent boundary data imposed at the stage times")
        return f"; formal order {p}, effective global order {eff} ({why})"

    def explain(self) -> str:
        """Scheme, dt, time, the implicit and explicit parts, the finite-state check and the
        plan."""
        return (f"TerrainSimulation: {self.scheme}, dt = {self._dt:g}, t = {self._t:g} "
                f"({self._it} steps); implicit: the coupled operator (metric terms implicit), "
                "explicit: right-hand sides on the dealiased physical grid; equations without "
                "time derivative imposed with their data at the stage time (bc_time = 'stage'"
                + self._bc_note + ")"
                + ("; SBDF startup lower_order (orders 1..q-1 for the first q-1 steps)"
                   if self.scheme.startswith("SBDF") else "")
                + self._order_text()
                + (f"; finite-state check every {self._finite_every} steps and at the end of "
                   "each run call (check_finite=False turns it off)" if self._finite_every
                   else "; finite-state check off (check_finite=False)")
                + "\n" + self.plan.explain())
