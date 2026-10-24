"""IVP solver: M dX/dt + L X = F(X, t), IMEX multistep and Runge–Kutta schemes.

Schemes (``SCHEMES``; catalogue, coefficients and observed orders),
constant step k = dt:

* SBDF1–4: semi-implicit backward differentiation (Wang & Ruuth 2008); same coefficients as the
  C++ ``ImexIntegrator``.
* CNAB1, CNAB2, MCNAB2, CNLF2: Crank–Nicolson-type multistep IMEX (Wang & Ruuth 2008,
  J. Comput. Math. 26, eqs. 2.5 with γ = 1/2, 2.9, 2.10, 2.11), written
  ``sum_j a_j M X^{n+1-j} + sum_j b_j L X^{n+1-j} = sum_{j>=1} c_j F^{n+1-j}``; they keep a
  history of M X, L X and F (``_MXh``, ``_LXh``, ``_Fh``).
* RK111, RK222, RK443 (Ascher, Ruuth & Spiteri 1997), RKSMR (Spalart, Moser & Rogers 1991,
  Appendix), RKGFY (Heun / Crank–Nicolson pair): IMEX Runge–Kutta tableaux (A, H, c), one implementation
  (:meth:`IVPSolver.rk_stages`).

The stacked global system of all Fourier groups is solved with batched C++ bordered banded
solvers, one factorization per LHS (a, b) cached. F: pseudospectral evaluator (NumPy,
dealiased products). The step loop is Python *over steps only*: every step is a fixed number of
vectorized calls (no loop over groups or modes).

Nothing is chosen for the user: scheme, dt and the startup of multistep schemes are
explicit. ``startup="lower_order"`` is the Dedalus convention (SBDF: orders 1..q during the first
q-1 steps, which caps the observed global order of SBDF3/4 at 2; CN family: CNAB1 for the first
step); ``startup="rk443"`` computes the first steps with RK443 substeps so the multistep scheme
keeps its nominal order.

Time-dependent boundary data (``bc_time``): an equation without time derivative (all its rows
have no mass: boundary conditions, constraints) whose right-hand side depends on the time parameter
and not on the unknowns is imposed at the time of each stage. With ``bc_time="stage"`` (the default
``"auto"`` takes it when g'(t) is not available, see below), the rows of these equations in the
right-hand side of a stage (or of a multistep step) are replaced by beta g(t_stage), beta the
coefficient of L in that solve, so L_row X_stage = g(t_stage) exactly and every scheme but RK443
keeps its order (RK443: 2, the stage order of its implicit part).
``bc_time="rk"`` imposes instead the stage values that the scheme's implicit tableau gives
when applied to g'(t), g_i = g(t_n) + dt sum_j H_ij g'(t_n + c_j dt): RK443 reaches about 2.8,
RK222/RKSMR 2 exactly; it needs g'(t) exactly (the boundary right-hand side differentiated by the
IR, :func:`conatus.ir.param_diff.diff_parameter`). The default ``bc_time="auto"`` takes "rk" when
that derivative exists and "stage" otherwise, with the reason in ``explain()`` (multistep steps
impose g(t_{n+1}) in both). ``bc_time="step"`` keeps the Dedalus convention (g carried by F with the
explicit weights; RK111/222/443, RKSMR, CNAB1/2 then drop to about first order), bit for bit. Never
switched silently: the choice and the equations it concerns are printed by ``explain()``; constant
boundary data are identical in both modes (no row is touched).

Modules that handle only some families (adjoint, ensembles, distributed solver, embedded error)
read ``solver.family`` ("sbdf", "cn", "rk") and refuse the others explicitly
(:func:`require_family`): ``solver.order`` is the SBDF order only (0 for every other scheme), so
no CN scheme is ever treated as SBDF.

The scheme catalogue and coefficients are in :mod:`conatus.timestep.schemes`, the checks shared
by every ``run`` in :mod:`conatus.timestep.run_checks`; this module re-exports their names.
"""
from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..discretize.plan import Plan
from ..errors import OptionError, did_you_mean, require_cupy
from ..ir import expr as E
from .lincomb import CsrApply, rk_rhs, sbdf_rhs
from .spectral import SpectralError, SpectralEvaluator
from .system import GlobalSystem
from .._deprecation import renames
from .schemes import (  # noqa: F401 (re-export: moved to schemes.py)
    BC_TIMES, CN_SCHEMES, RK_SCHEMES, SBDF_SCHEMES, SCHEMES, STARTUPS, _SCHEME_INFO,
    cn_coefficients, require_family, rk_tableau, sbdf_coefficients, scheme_description,
    scheme_family, setup_scheme,
)
from .run_checks import (  # noqa: F401 (re-export: moved to run_checks.py)
    FINITE_CHECK_EVERY, WHOLE_STEPS_TOL, _OTHER_FORMATS, backend_hint, check_callback,
    check_checkpoint_suffix, check_dt_backend, check_params_unchanged, check_state_finite,
    check_steps, check_substeps, check_time_parameter, finite_cadence, is_whole,
    solver_finite_check,
)

__all__ = ["IVPSolver", "SCHEMES", "SBDF_SCHEMES", "CN_SCHEMES", "RK_SCHEMES", "rk_tableau",
           "sbdf_coefficients", "cn_coefficients", "scheme_family", "scheme_description",
           "setup_scheme", "require_family", "STARTUPS", "BC_TIMES"]

#: Run-time resolution check: at every finite-state check the relative
#: spectral tail of each unknown along each direction (``conatus.resolution``: largest coefficient
#: among the last 10 % of the mode numbers over the largest one) is measured on the coefficients
#: in memory; above RUN_TAIL_TOL (the "under-resolved" verdict of ``resolution_report``,
#: TAIL_MARGINAL) an UnderResolvedWarning is issued once per solver and explain() says it. An
#: unknown whose largest coefficient is below RUN_TAIL_FLOOR times the largest of the state is
#: round-off-sized (e.g. a component that stays zero) and is not judged.
#: Level: 1e-3 in amplitude (1e-6 in energy, per mode), the
#: SPURIOUS_TAIL_TOL scale; the former 1e-4 warned on the resolved flagship KS run (L = 22,
#: N = 32: tail 1.0e-4 at the top of its 10 % window). A direction whose tail window holds fewer
#: than RUN_TAIL_MIN_MODES coefficients (a polynomial basis with N <= 20: two coefficients) is
#: not judged: one or two top coefficients are not a tail (they report "tail 1" when the top one
#: dominates); explain() lists the directions left out.
RUN_TAIL_TOL = 1e-3
RUN_TAIL_FLOOR = 1e-12
RUN_TAIL_MIN_MODES = 3
def _equation_rows(host, k: int) -> np.ndarray:
    """Global rows of equation k: ``rows_obj`` of the Cartesian system, else the rows that the
    right-hand-side map of the equation reaches (polar and spherical systems)."""
    if getattr(host, "rows_obj", None) is not None:
        return np.nonzero(np.asarray(host.rows_obj) == k)[0]
    return np.unique(host._rhs_maps[k].tocoo().row)


class IVPSolver:
    """Constant-step IMEX solver of a discretized IVP (see module docstring)."""

    # run_graph of a time-dependent F: rows of time factors built per block of this many steps
    _GRAPH_TABLE_ROWS = 1024

    def __init__(self, plan: Plan, scheme: str, dt: float, *, reduced: bool | None = None,
                 ordering: str = "interleaved", startup: str = "lower_order",
                 startup_substeps: int = 2, time_parameter: str | None = None,
                 backend: str = "cpu", f_backend: str = "evaluator",
                 complexify: bool = False, decouple: bool = False, bc_time: str = "auto",
                 check_finite: bool | int = True) -> None:
        setup_scheme(self, scheme, startup, where=f"IVPSolver(plan, {scheme!r}, ...)",
                     bc_time=bc_time)
        self._finite_every = finite_cadence(check_finite, f"IVPSolver(plan, {scheme!r}, ...)")
        check_dt_backend(dt, backend, f"IVPSolver(plan, {scheme!r}, ...)")
        self.plan = plan
        self._params0 = dict(plan.params)   # matrices built with these (check_params_unchanged)
        self.scheme, self.dt = scheme, float(dt)
        # implicit cross(W, u) with a constant W (Coriolis in a box): stable at any dt, but
        # its waves (frequency <= |W|) are damped by the implicit stages when |W| dt > 1
        from ..discretize.compiler import implicit_rotation_rate
        pb = getattr(plan, "problem", None)
        self._rotation = None if pb is None else implicit_rotation_rate(pb, getattr(plan, "params", {}))
        self._rotation_warned = False
        self._check_rotation_step()
        self.startup = startup
        self.startup_substeps = check_substeps(startup_substeps, f"IVPSolver(plan, {scheme!r}, ...)")
        check_time_parameter(plan, time_parameter, f"IVPSolver(plan, {scheme!r}, ...)")
        self.time_parameter = time_parameter
        self.backend = backend
        # complexify=True: RealFourier (cos, -sin) pairs solved as complex half-size systems (
        # with >= 2 RealFourier directions, one complex system per mode exp(i(kx x +- kz z)));
        # decouple=True: block-diagonal groups split into their components (both exact, opt-in)
        host = GlobalSystem(plan, reduced=reduced, ordering=ordering, complexify=complexify,
                            decouple=decouple)
        if backend == "cuda":
            cupy = require_cupy("IVPSolver(..., backend='cuda')")
            from .device import DeviceSystem
            self.xp = cupy
            self.system = DeviceSystem(host)
        else:
            self.xp = np
            self.system = host
        self.evaluator = SpectralEvaluator(plan.domain, xp=None if backend == "cpu" else self.xp)
        if f_backend not in ("evaluator", "fused"):
            raise OptionError(f"IVPSolver: f_backend={f_backend!r}",
                              why="the evaluator of F is an explicit choice",
                              fix="f_backend='evaluator' (the reference) or 'fused' (compiled "
                                  "kernels, same numbers)" + did_you_mean(str(f_backend),
                                                                          ("evaluator", "fused")))
        self.f_backend = f_backend
        self.X = self.xp.zeros(self.system.n, dtype=self.system.dtype)
        self.t = 0.0
        self.iteration = 0
        self._MXh: deque = deque()
        self._Fh: deque = deque()
        self.n_F = 0
        self._ncc = {f: self._grid_to_coeffs(f, g) for f, g in plan._ncc_data.items()}
        pb = plan.problem
        self._rhs_exprs = [None if eq.rhs.is_zero() else eq.rhs for eq in pb.equations]
        self._rhs_vars = frozenset(f.name for e in self._rhs_exprs if e is not None
                                   for f in e.free_fields())
        self._fused, self._fused_note = None, ""
        if f_backend == "fused":
            if time_parameter:
                raise OptionError("f_backend='fused' freezes the parameters: a time-dependent "
                                  "right-hand side (time_parameter) is refused",
                                  fix="f_backend='evaluator' (the reference evaluator of F, which "
                                      "sets the time at every stage)", doc="docs/site/guide/timesteppers.md")
            from .fused import FusedRHS
            # In a reflection subspace, F is evaluated parity-reduced along the first reflected
            # RealFourier axis (half the transforms' columns, half-grid products). Same numbers;
            # if the RHS does not allow it, the reason is kept and shown by explain().
            refl, self._fused_note = self._fused_reflection(plan, pb), ""
            kw = dict(xp=None if backend == "cpu" else self.xp, dtype=self.system.dtype)
            try:
                self._fused = FusedRHS(plan.domain, pb.equations, self._rhs_exprs, dict(plan.params),
                                       reflection=refl, symmetry=pb.symmetry if refl else None, **kw)
            except SpectralError as err:
                if refl is None:
                    raise
                self._fused_note = f"parity reduction not used ({err}); full-space fused F"
                self._fused = FusedRHS(plan.domain, pb.equations, self._rhs_exprs, dict(plan.params), **kw)
        # GPU, 2D RealFourier x Chebyshev: the whole stacked F in ~10 kernels (same algorithms)
        self._fc2d, self._fc2d_note, self._fc2d_fused = None, "", None
        if self._fused is not None and backend == "cuda":
            self._build_fc2d()

    def _build_fc2d(self) -> None:
        from .fc2d import FourierChebyshev2DRHS
        from .fused import FusedRHS
        fused, self._fc2d_note = self._fused, ""
        if getattr(fused, "reflection", None) is not None:
            # In a reflection subspace the parity-reduced evaluator (half-grid products, ~80
            # kernels) was slower on the GPU than the 2D pipeline on the full grid (512:
            # A100 256x64 RK443 1.71 against 1.06 ms/step in the full space, run
            #). The pipeline reads only the subspace's coefficients
            # (the others are zero) and writes only its rows: the same products on the same
            # dealiased grid, so the same F to rounding; the reduced evaluator stays the fallback.
            pb = self.plan.problem
            try:
                fused = FusedRHS(self.plan.domain, pb.equations, self._rhs_exprs, dict(self.plan.params),
                                 xp=self.xp, dtype=self.system.dtype)
            except SpectralError as err:
                self._fc2d, self._fc2d_fused = None, None
                self._fc2d_note = f"2D pipeline not used: full-grid RHS not available ({err})"
                return
        why = FourierChebyshev2DRHS.reason(fused, self.system)
        if why is not None:
            self._fc2d, self._fc2d_fused = None, None
            self._fc2d_note = f"2D pipeline not used: {why}"
            return
        ev, pb = self.evaluator, self.plan.problem
        const = [ev._embed(ev.evaluate(e, {}, dict(self.plan.params)), eq.lhs.rank, shape)
                 if e is not None and not e.free_fields() else None
                 for e, shape, eq in zip(self._rhs_exprs, self.system.eq_shapes, pb.equations, strict=True)]
        const = [None if c is None else self.xp.asnumpy(c) if hasattr(c, "device") else c
                 for c in const]
        self._fc2d = FourierChebyshev2DRHS(fused, self.system, self.system.host.gather_rhs(const))
        self._fc2d_fused = fused
        if fused is not self._fused:
            self._fc2d_note = ("2D pipeline on the full grid, restricted to the symmetry subspace "
                               "(faster on the GPU than the parity-reduced evaluator)")

    def _fused_reflection(self, plan: Plan, pb):
        """First declared reflection along a RealFourier axis, if the system is reduced."""
        if not getattr(self.system, "reduced", False) or pb.symmetry is None:
            return None
        from ..ir.bases import RealFourier
        for R in pb.symmetry.generators:
            bx = plan.domain.basis(R.coord)
            if isinstance(bx, RealFourier):
                return R
        return None

    # ------------------------------------------------------------------ state
    def _grid_to_coeffs(self, f: E.Field, g: np.ndarray) -> np.ndarray:
        c = self.plan._to_coeffs(np.asarray(g), f.domain, f.rank)
        return c if self.backend == "cpu" else self.xp.asarray(c)

    def set_state(self, data: Mapping[E.Field, Any], t: float = 0.0) -> None:
        """Initial condition: FieldData (explicit layout) or callables of the coordinate grids.
        Variables not given start at zero. Resets the multistep history."""
        coeffs = []
        for v, shape in zip(self.system.variables, self.system.var_shapes, strict=True):
            if v in data:
                g = self.plan._as_grid(v, data[v])
                coeffs.append(self._grid_to_coeffs(v, g))
            else:
                coeffs.append(np.zeros(shape))
        self.X = self.system.gather_state(coeffs)
        self._restart_path = None                   # a new initial state: not a continuation
        self._restart_changes = []
        self.t, self.iteration = float(t), 0
        self._t_origin, self._it_origin = float(t), 0
        self._t_initial = float(t)                  # start time of the chain (checkpoints)
        self._reset_history()

    def _reset_history(self) -> None:
        """Forget the multistep history (M X, L X, F of the previous steps)."""
        self._MXh.clear()
        self._Fh.clear()
        self._LXh.clear()

    def coeffs(self, name: str) -> np.ndarray:
        """Coefficient array of variable ``name`` (host array, the plan's layout)."""
        k = [v.name for v in self.system.variables].index(name)
        c = self.system.scatter(self.X)[k]
        return c if self.backend == "cpu" else self.xp.asnumpy(c)

    def grid(self, name: str) -> np.ndarray:
        """Values of variable ``name`` on its (non-dealiased) grid."""
        return self.plan.to_grid(name, self.coeffs(name))

    # ------------------------------------------------------------------ F
    def F(self, X: np.ndarray, t: float) -> np.ndarray:
        """Stacked explicit right-hand side F(X, t) (pseudospectral, dealiased products)."""
        self.n_F += 1
        if self._fc2d is not None:
            # 2D pipeline: reads the stacked X directly. Checked before the per-variable scatter
            # below, which it does not use (it cost three kernels per variable and per F, also
            # inside the captured graph). In a reflection subspace the pipeline has its own
            # full-grid FusedRHS (_fc2d_fused); both are kept in step with the parameters.
            params = dict(self.plan.params)              # fused F refuses time_parameter
            fz = self._fc2d_fused
            src = fz._src
            fz.check_params(params)
            if fz._src is not src:                       # parameters changed: recompiled
                if self._fused is not fz:
                    self._fused.check_params(params)
                self._build_fc2d()
            if self._fc2d is not None:
                return self._fc2d.evaluate(X)
        data = dict(self._ncc)
        # only the unknowns that appear in a right-hand side are expanded to coefficient arrays
        data.update((v, c) for v, c in zip(self.system.variables,
                                           self.system.scatter(X, only=self._rhs_vars), strict=True) if c is not None)
        params = dict(self.plan.params)
        if self.time_parameter:
            params[self.time_parameter] = t
        out = []
        ev = self.evaluator
        if self._fused is not None:
            self._fused.check_params(params)
            for c, e, shape, eq in zip(self._fused.evaluate(data), self._rhs_exprs,
                                       self.system.eq_shapes, self.plan.problem.equations, strict=True):
                if c is None and e is not None:       # constant RHS (BCs, gauges): evaluator
                    c = ev.evaluate(e, data, params)
                out.append(None if c is None else ev._embed(c, eq.lhs.rank, shape))
            return self.system.gather_rhs(out)
        for e, shape, eq in zip(self._rhs_exprs, self.system.eq_shapes, self.plan.problem.equations, strict=True):
            if e is None:
                out.append(None)
                continue
            c = ev.evaluate(e, data, params)
            out.append(ev._embed(c, eq.lhs.rank, shape))
        return self.system.gather_rhs(out)

    # ------------------------------------------------------------------ steps
    def _solve_state(self, a: float, b: float, rhs):
        """(a M + b L) X = rhs for a new state of the step, then the declared projection of the
        m = 0 coefficients onto real fields on curvilinear plans (timestep/real_projection.py;
        ``_m0proj`` is None elsewhere: the plain solve, same bits). The single place where a
        step's solve output is projected: RK stages, SBDF/CN steps, their CUDA graphs, the
        FlowMap replay and the tangent/adjoint (adjoint/flowmap.py) go through it."""
        x = self.system.solve(a, b, rhs)
        proj = getattr(self, "_m0proj", None)
        return x if proj is None else proj.apply(x)

    def _sbdf_step(self, q_max: int) -> None:
        sysm, dt = self.system, self.dt
        self._MXh.appendleft(sysm.M @ self.X)
        self._Fh.appendleft(self.F(self.X, self.t))
        while len(self._MXh) > self.order:
            self._MXh.pop()
            self._Fh.pop()
        q = min(q_max, len(self._MXh))
        a, c = sbdf_coefficients(q)
        rhs = sbdf_rhs(self.xp, sysm.n, sysm.dtype,
                       [(c[j], self._Fh[j - 1], a[j] / dt, self._MXh[j - 1]) for j in range(1, q + 1)])
        rhs = self.impose_bc(rhs, self.t + dt, 1.0)
        self.X = self._solve_state(a[0] / dt, 1.0, rhs)

    # ------------------------------------------------------------------ boundary data
    def _bc_data(self):
        """(equations, host rows, device rows) of the time-dependent boundary data, or None
        (cached).

        An equation qualifies when all its rows have no mass (no time derivative: a boundary
        condition or a constraint) and its right-hand side depends on the time parameter and on
        no unknown. A time-dependent algebraic right-hand side that also depends on the unknowns
        cannot be imposed at a stage time before the stage is known: refused with
        bc_time='auto', 'rk' or 'stage' by :meth:`_bc_resolve` (fix: bc_time='step')."""
        if "_bcd_cache" in self.__dict__:
            return self._bcd_cache
        self._bcd_cache = None
        tp = self.time_parameter
        if not tp:
            return None
        host = getattr(self.system, "host", self.system)
        M = abs(host.M).tocsr()
        has_mass = np.asarray(M.sum(axis=1)).ravel() > 0
        V = set(self.system.variables)
        eqs, mixed, rows = [], [], []
        exprs = [None if eq.rhs.is_zero() else eq.rhs for eq in self.plan.problem.equations]
        for k, e in enumerate(exprs):
            if e is None or tp not in {p.name for p in e.free_parameters()}:
                continue
            r = _equation_rows(host, k)
            if r.size == 0 or has_mass[r].any():
                continue                     # evolution equation: F at the stage times as usual
            if set(e.free_fields()) & V:
                mixed.append(k)
                continue
            eqs.append(k)
            rows.append(r)
        self._bc_mixed = mixed               # refused in the stage modes (_bc_resolve)
        if eqs:
            r = np.sort(np.concatenate(rows))
            self._bcd_cache = (tuple(eqs), r, self.xp.asarray(r))
        return self._bcd_cache

    def _data_rhs_eq(self, k: int, params: dict, e=None):
        """Coefficients of the (state-independent) right-hand side of equation k (or of the
        expression ``e`` in its place: its time derivative), laid out for gather_rhs (the
        evaluator path of :meth:`F`)."""
        ev, eq = self.evaluator, self.plan.problem.equations[k]
        e = self._rhs_exprs[k] if e is None else e
        return ev._embed(ev.evaluate(e, dict(self._ncc), params), eq.lhs.rank,
                         self.system.eq_shapes[k])

    def _data_rhs(self, t: float, deriv: bool = False):
        """Global right-hand-side vector holding only the time-dependent boundary data at t
        (``deriv``: their exact time derivative g'(t), bc_time='rk')."""
        eqs = self._bc_data()[0]
        params = dict(self.plan.params)
        params[self.time_parameter] = t
        dex = self._bc_resolve()[2] if deriv else {}
        out = [None if k not in eqs or (deriv and dex[k] is None)
               else self._data_rhs_eq(k, params, dex.get(k))
               for k in range(len(self.plan.problem.equations))]
        return self.system.gather_rhs(out)

    # The derivative path of _data_rhs exists for every solver that evaluates the boundary data
    # equation by equation; a subclass that cannot sets the reason here (bc_time='rk' refused,
    # 'auto' falls back to 'stage' and says why).
    _bc_rk_unsupported: str | None = None

    def _bc_resolve(self):
        """(mode, reason, {equation: d g/dt or None}) of the time-dependent boundary data,
        cached. mode: 'rk', 'stage' or 'step' (what is done); reason: why 'auto' gave this mode
        ('' when it is what was asked)."""
        if "_bc_resolved" in self.__dict__:
            return self._bc_resolved
        req = getattr(self, "_bc_request", "auto")
        bcd = self._bc_data() if (self.time_parameter and req != "step") else None
        mixed = getattr(self, "_bc_mixed", []) if req != "step" else []
        if mixed:
            raise OptionError(
                f"equation(s) {mixed} without time derivative have a right-hand side that depends "
                f"on both {self.time_parameter} and the unknowns: it cannot be imposed at the stage times",
                where=f"IVPSolver(..., bc_time={self._bc_request!r})",
                why="time-dependent boundary data are imposed at every stage time by default; a "
                    "value that depends on the unknowns of the stage is not known before it",
                fix="bc_time='step' (the Dedalus convention: carried by F with the explicit weights; "
                    "about first order with RK111/222/443, RKSMR, CNAB1/2)")
        if req in ("stage", "step"):
            res = (req, "", {})
        elif bcd is None:                   # nothing to impose: the request stands, no-op
            res = ("rk", "", {})
        else:
            from ..ir.param_diff import ParameterDerivativeError, diff_parameter
            why, dex = self._bc_rk_unsupported, {}
            if why is None:
                for k in bcd[0]:
                    try:
                        d = diff_parameter(self._rhs_exprs[k], self.time_parameter)
                    except ParameterDerivativeError as err:
                        why = f"no exact time derivative of the right-hand side of equation {k} ({err})"
                        break
                    dex[k] = None if d.is_zero() else d
            if why is None:
                res = ("rk", "", dex)
            elif req == "rk":
                raise OptionError(
                    f"bc_time='rk': {why}", where="IVPSolver(..., bc_time='rk')",
                    why="the stage values of the boundary data are the scheme applied to g'(t), "
                        "which must be exact (it is never approximated by differences)",
                    fix="bc_time='stage' (g at every stage time; RK443 then order 2) or write the "
                        "boundary value as an expression of the time parameter (sin(t), t**2, ...)")
            else:
                res = ("stage", why, {})
        self._bc_resolved = res
        return res

    @property
    def bc_time(self) -> str:
        """Convention actually used for time-dependent boundary data: 'rk', 'stage' or 'step'
        (``bc_time='auto'`` resolved; the reason of a fallback is in ``explain()``)."""
        return self._bc_resolve()[0]

    def bc_rows(self):
        """Device rows of the time-dependent boundary data imposed at stage times, or None (no
        such data, or bc_time='step'). FlowMap zeroes the same rows in the tangent and the
        adjoint (their right-hand side does not depend on the state)."""
        if getattr(self, "_bc_request", "auto") == "step" or self.bc_time == "step":
            return None
        bcd = self._bc_data()
        return None if bcd is None else bcd[2]

    def impose_bc(self, rhs, t: float, beta: float, G=None):
        """The rows of the time-dependent boundary data of a solve (a M + beta L) X = rhs
        replaced by beta g(t), so that L_row X = g(t) at the time t of the state computed
        (``G``: the boundary values to impose instead, a global vector -- bc_time='rk'). No-op
        (same array, same bits) without such data or with bc_time='step'."""
        rows = self.bc_rows()
        if rows is None:
            return rhs
        if G is None:
            G = self._data_rhs(t)
        rhs[rows] = beta * G[rows]
        return rhs

    def _bc_stage_values(self, t: float, dt: float, H, c):
        """bc_time='rk': function i -> global vector G_i = g(t) + dt sum_{j<=i} H_ij
        g'(t + c_j dt), the boundary values of stage i of one RK step from t (the scheme's
        implicit tableau applied to g'); None in the other modes. g' is evaluated once per
        abscissa c_j that is used."""
        if self.bc_time != "rk" or self.bc_rows() is None:
            return None
        g0 = self._data_rhs(t)
        dg: dict = {}

        def G(i: int):
            out = g0.copy()
            for j in range(i + 1):
                if H[i][j] != 0:
                    if j not in dg:
                        dg[j] = self._data_rhs(t + c[j] * dt, deriv=True)
                    out += (dt * H[i][j]) * dg[j]
            return out
        return G

    def _bc_text(self) -> str:
        """explain() line on time-dependent boundary data ('' when there are none)."""
        req = getattr(self, "_bc_request", "auto")
        bcd = self._bc_data() if self.time_parameter else None
        if bcd is None:
            return ("" if req != "step" else
                    "\n  boundary data: bc_time='step' (no time-dependent boundary data here)")
        eqs = ", ".join(str(k) for k in bcd[0])
        head = (f"\n  boundary data: right-hand sides of equation(s) {eqs} depend on "
                f"{self.time_parameter} (no time derivative): ")
        mode, why, _ = self._bc_resolve()
        multistep = self.family != "rk"
        if mode == "rk":
            return (head + "stage values from the scheme's implicit tableau applied to the exact "
                    f"d/d{self.time_parameter} of the data (bc_time='rk'"
                    + (", default 'auto'" if req == "auto" else "") + "; RK443 about order 2.8, "
                    "the others their nominal order"
                    + ("; multistep steps impose g(t_n+1), the RK startup substeps the RK values"
                       if multistep else "")
                    + "); bc_time='stage' imposes g at every stage time, bc_time='step' gives "
                      "the Dedalus convention")
        if mode == "stage":
            return (head + "imposed at every stage time (bc_time='stage'"
                    + (f", 'auto' fallback: {why}" if why else "")
                    + "; nominal order except RK443: 2); bc_time='step' gives the Dedalus "
                      "convention (carried by F with the explicit weights)")
        return (head + "carried by F with the explicit weights, the Dedalus "
                "convention (bc_time='step'); RK111/222/443, RKSMR, CNAB1/2 are then about first "
                "order in time; bc_time='auto' imposes them at the stage times")

    def _push_cn_history(self) -> None:
        """Record M X^n, L X^n, F(X^n, t^n) at the front of the CN history (s levels kept)."""
        sysm = self.system
        self._MXh.appendleft(sysm.M @ self.X)
        self._LXh.appendleft(sysm.L @ self.X)
        self._Fh.appendleft(self.F(self.X, self.t))
        while len(self._MXh) > self._nhist:
            self._MXh.pop()
            self._LXh.pop()
            self._Fh.pop()

    def _cn_coefficients_now(self, nhist: int):
        """Coefficients of this step: the scheme's once the history is full, CNAB1 before
        (startup='lower_order', the Dedalus convention; declared in explain())."""
        return self._cn if nhist >= self._nhist else cn_coefficients("CNAB1")

    def _cn_rhs(self, coefs, MXh, LXh, Fh):
        """sum_{j>=1} (c_j F_j - (a_j/dt) M X_j - b_j L X_j), accumulated from zero in that order
        (zero coefficients skipped); one kernel for real float64 states on the GPU."""
        a, b, c = coefs
        dt, sysm = self.dt, self.system
        terms = []
        for j in range(1, len(a)):
            if c[j] != 0:
                terms.append(("+", c[j], Fh[j - 1]))
            if a[j] != 0:
                terms.append(("-", a[j] / dt, MXh[j - 1]))
            if b[j] != 0:
                terms.append(("-", b[j], LXh[j - 1]))
        return rk_rhs(self.xp, self.xp.zeros(sysm.n, dtype=sysm.dtype), terms)

    def _cn_step(self) -> None:
        """One CN-family multistep step (Wang & Ruuth 2008):
        (a_0/dt M + b_0 L) X^{n+1} = sum_{j>=1} c_j F^{n+1-j} - (a_j/dt) M X^{n+1-j} - b_j L X^{n+1-j}."""
        self._push_cn_history()
        coefs = self._cn_coefficients_now(len(self._MXh))
        rhs = self._cn_rhs(coefs, self._MXh, self._LXh, self._Fh)
        rhs = self.impose_bc(rhs, self.t + self.dt, coefs[1][0])
        self.X = self._solve_state(coefs[0][0] / self.dt, coefs[1][0], rhs)

    def _rk_step(self, X0: np.ndarray, t: float, dt: float, tableau) -> np.ndarray:
        return self.rk_stages(X0, t, dt, tableau)[-1]

    @renames("conatus.timestep.integrator:IVPSolver.rk_stages")
    def rk_stages(self, x0: np.ndarray, t: float, dt: float, tableau=None, F=None) -> list:
        """Stage states [X_0, X_1, ..., X_s] of one IMEX Runge–Kutta step (X_s = next state).
        The single implementation of the RK arithmetic: :meth:`step` and the adjoint
        (``conatus.adjoint.FlowMap``) both use it, so tangent/adjoint replay exactly the forward
        stages. ``F``: optional replacement of :meth:`F` with the same signature (X, t)."""
        if tableau is None and self._rk is None:
            raise OptionError(f"rk_stages: scheme {self.scheme} is multistep, it has no Runge–Kutta "
                              "stages (no one-step map)",
                              fix=f"use an IMEX Runge–Kutta scheme ({', '.join(RK_SCHEMES)}), or "
                                  "pass a tableau explicitly")
        A, H, c = tableau if tableau is not None else self._rk
        Fn = self.F if F is None else F
        s = len(c) - 1
        sysm = self.system
        X = [x0]
        Fs: list = [None] * (s + 1)
        LX: list = [None] * (s + 1)
        # GPU: M x0 and L X_k are applied inside the right-hand-side kernel (lincomb.CsrApply)
        dev = self.backend == "cuda"
        MX0 = CsrApply(sysm.M, x0) if dev else sysm.M @ x0
        bcv = self._bc_stage_values(t, dt, H, c)   # (None unless bc_time='rk' with data)
        for i in range(1, s + 1):
            j = i - 1
            if any(A[m][j] != 0 for m in range(j + 1, s + 1)):
                Fs[j] = Fn(X[j], t + c[j] * dt)
            if any(H[m][j] != 0 for m in range(j + 1, s + 1)):
                LX[j] = CsrApply(sysm.L, X[j]) if dev else sysm.L @ X[j]
            terms = []
            for k in range(i):
                if A[i][k] != 0:
                    terms.append(("+", dt * A[i][k], Fs[k]))
                if H[i][k] != 0:
                    terms.append(("-", dt * H[i][k], LX[k]))
            rhs = rk_rhs(self.xp, MX0, terms)      # same roundings, one kernel on the GPU
            rhs = self.impose_bc(rhs, t + c[i] * dt, dt * H[i][i],   # (no-op if none)
                                 None if bcv is None else bcv(i))     # bc_time='rk'
            X.append(self._solve_state(1.0, dt * H[i][i], rhs))
        return X

    def step(self) -> None:
        """One step of the declared scheme (start-up steps of a multistep scheme included)."""
        check_params_unchanged(self, "IVPSolver.step")
        if self.order:
            need_startup = self.startup == "rk443" and self.iteration < self.order - 1
            if need_startup:
                # record history at the current state, then advance with RK443 substeps
                self._MXh.appendleft(self.system.M @ self.X)
                self._Fh.appendleft(self.F(self.X, self.t))
                h = self.dt / self.startup_substeps
                X, t = self.X, self.t
                for _ in range(self.startup_substeps):
                    X = self._rk_step(X, t, h, rk_tableau("RK443"))
                    t += h
                self.X = X
            else:
                self._sbdf_step(self.order)
        elif self.family == "cn":
            if self.startup == "rk443" and self.iteration < self._nhist - 1:
                # record the history at the current state, then advance with RK443 substeps
                self._push_cn_history()
                h = self.dt / self.startup_substeps
                X, t = self.X, self.t
                for _ in range(self.startup_substeps):
                    X = self._rk_step(X, t, h, rk_tableau("RK443"))
                    t += h
                self.X = X
            else:
                self._cn_step()
        else:
            self.X = self._rk_step(self.X, self.t, self.dt, self._rk)
        self.iteration += 1
        self.t = self._t0 + (self.iteration - self._it0) * self.dt   # no accumulated round-off
        if self._fused is not None and self.backend == "cuda":
            self._fused.check_real_projection()       # device-accumulated check (CPU: immediate)

    @property
    def _t0(self) -> float:
        return getattr(self, "_t_origin", 0.0)

    @property
    def _it0(self) -> int:
        return getattr(self, "_it_origin", 0)

    def set_dt(self, dt: float) -> None:
        """Change the step (Runge–Kutta only: multistep coefficients are constant-step here)."""
        if self.order:
            raise OptionError(f"{self.scheme} is constant-step (variable-step SBDF coefficients "
                              "not implemented): dt cannot change",
                              where=f"IVPSolver.set_dt({dt!r})",
                              why="the multistep history was built with the old dt",
                              fix="build a new solver with the new dt from the current state "
                                  "(solver.X), or use an IMEX Runge–Kutta scheme (RK222, RK443, "
                                  "RKSMR, RKGFY), whose step can change", doc="docs/site/guide/timesteppers.md")
        if self.family == "cn":
            raise OptionError(f"{self.scheme} is constant-step (variable-step CN coefficients not "
                              "implemented): dt cannot change",
                              where=f"IVPSolver.set_dt({dt!r})",
                              why="the multistep history was built with the old dt",
                              fix="use an IMEX Runge–Kutta scheme (" + ", ".join(RK_SCHEMES)
                                  + "), or restart from the current state with a new solver")
        check_dt_backend(dt, self.backend, f"IVPSolver.set_dt({dt!r})")
        self._t_origin, self._it_origin = self.t, self.iteration
        self.dt = float(dt)
        self._check_rotation_step()

    def _rotation_text(self) -> str:
        """explain() line of an implicit constant-vector cross term, or ''."""
        rot = getattr(self, "_rotation", None)
        if rot is None:
            return ""
        rate, term = rot
        x = rate * self.dt
        return (f"implicit rotation {term}: |W| = {rate:g}, |W| dt = {x:.3g} (stable at any dt; "
                # audit "assurances": "its waves are resolved in time" for any
                # scheme at |W| dt <= 1, while SBDF1 there damps them by 29 % per step
                + ("dt is below the rotation time 1/|W|; the scheme still damps and slows its "
                   "waves by its own error, of order (|W| dt)^p: halve dt to measure it)" if x <= 1 else
                   "the implicit stages damp and slow its waves of frequency omega with omega dt > 1"
                   ": stability, not accuracy; dt < 1/|W| follows them)"))

    def _check_rotation_step(self) -> None:
        """One UnderResolvedWarning per solver when |W| dt > 1 for an implicit rotation term."""
        rot = getattr(self, "_rotation", None)
        if rot is None or self._rotation_warned or rot[0] * self.dt <= 1:
            return
        from ..errors import UnderResolvedWarning, _warn_user
        self._rotation_warned = True
        _warn_user(self._rotation_text() + "; warned once, recorded by explain()",
                   UnderResolvedWarning)

    @renames("conatus.timestep.integrator:IVPSolver.run")
    def run(self, steps: int, callback: Callable[[IVPSolver], None] | None = None,
            every: int | None = None, *, record=None, cache: bool = False):
        """Advance ``steps`` steps; ``callback(self)`` after every step whose iteration is a
        multiple of ``every``.

        ``record`` (run registry, ``conatus.runs``): True, a run name, a dict of options or an open
        run (``with cn.runs.track(...) as run``) records this call (numerics, problem hash, code,
        start/end state) and returns the run; None (default) records nothing and returns None.
        ``cache=True`` (off by default, needs ``record``) reuses the end state of an identical
        recorded computation instead of stepping, with a warning saying what was reused.

        Supply ``callback`` and ``every`` together; otherwise the call is refused before any step
        is taken. The call frequency costs time and changes what a monitor sees, so it is not
        chosen for the user."""
        steps = self._run_arguments(steps, callback, every)
        if record is not None or cache:              # run registry (explicit opt-in only)
            from ..runs.integrations import ivp_run
            return ivp_run(self, steps, callback, every, record=record, cache=cache)
        fe = self._finite_every
        for _ in range(steps):
            self.step()
            if fe and self.iteration % fe == 0:
                self._check_finite("IVPSolver.run")
            if callback is not None and self.iteration % every == 0:
                callback(self)
        if fe and steps:
            self._check_finite("IVPSolver.run (end of the call)")

    def _run_arguments(self, steps: int, callback: Callable[[IVPSolver], None] | None = None,
                       every: int | None = None, *, record=None, cache: bool = False) -> int:
        """The checks of :meth:`run` on its arguments, without side effects; returns the number
        of steps. Shared by ``run`` and by ``python -m conatus agent check``."""
        check_callback(callback, every, "IVPSolver.run")
        return check_steps(steps, "IVPSolver.run", "steps")

    @renames("conatus.timestep.integrator:IVPSolver.run_graph")
    def run_graph(self, steps: int) -> None:
        """Advance ``steps`` steps by replaying one captured CUDA graph per step (backend='cuda'
        only). Same arithmetic as :meth:`run`: one step is captured once, then replayed; this
        removes the per-kernel launch/Python latency. Explicit opt-in. A time-dependent F is
        refused (a captured step would freeze t) unless the solver refreshes its time factors in
        device slots before each replay (spherical f_backend='fused' with a Runge--Kutta scheme:
        ``sphere_fused.GraphScalars``; same factors as :meth:`step`, rows built per block of
        ``_GRAPH_TABLE_ROWS`` steps).

        SBDF: the multistep history lives in fixed device buffers during the replay (rotated by
        copies inside the graph, same values as the Python deque) and is written back to the
        deque afterwards, so :meth:`run`, checkpoints and restarts continue identically. Steps
        before the history is full (start-up) are taken with :meth:`step` first."""
        if self.backend != "cuda":
            raise OptionError("run_graph needs backend='cuda' (it replays a captured CUDA graph)",
                              fix="IVPSolver(..., backend='cuda'), or run(steps) on the CPU",
                              doc="docs/site/guide/timesteppers.md")
        gts = None
        if self.time_parameter:
            # a time-dependent F can be replayed only when the solver feeds its time factors
            # through device slots refreshed before each replay (spherical planned F, B2)
            gts = self._graph_time_support()
            if gts is None or self._rk is None:
                raise OptionError("run_graph: F depends on t (time_parameter); a captured step would "
                                  "freeze t", fix="run(steps) (one Python call per step, t set at "
                                                   "every stage); spherical IVPs with f_backend='fused', "
                                                   "a Runge--Kutta scheme, the generated product kernels, the direct "
                                                   "cuBLAS library and no time-dependent boundary "
                                                   "data replay a graph", doc="docs/site/guide/timesteppers.md")
        steps = check_steps(steps, "IVPSolver.run_graph", "steps")
        check_params_unchanged(self, "IVPSolver.run_graph")
        # The graph runs on its own non-blocking stream, which does not wait for work queued on
        # the caller's stream: without this barrier the replay (and the capture's warm-up) could
        # read X, or the history, before the previous plain steps have finished writing them
        # (seen as non-reproducible replays once the steps got fast).
        self.xp.cuda.get_current_stream().synchronize()
        multistep = self.family in ("sbdf", "cn")
        if multistep:
            while steps > 0 and not self._sbdf_steady():
                self.step()                  # start-up (lower order or RK443 substeps)
                steps -= 1
            if steps <= 0:
                return
            self.xp.cuda.get_current_stream().synchronize()   # start-up steps done (see above)
        key = (self.dt, self.scheme)
        g = getattr(self, "_graph", None)
        if g is None or g["key"] != key:
            g = self._graph = self._capture_graph()
            g["key"] = key
        with g["stream"]:
            g["X"][...] = self.X
            if multistep:
                self._history_to_buffers(g)
            for i0 in range(0, steps, self._GRAPH_TABLE_ROWS):
                i1 = min(steps, i0 + self._GRAPH_TABLE_ROWS)
                nsl = 0
                if gts is not None:          # time factors of these steps, stage times of rk_stages
                    # step start times as step() uses them: self.t for the first, then the
                    # origin formula (no accumulated round-off); rows built per block of steps
                    # (bounded device memory however long the run)
                    ts = [self.t if i == 0 else self._t0 + (self.iteration + i - self._it0) * self.dt
                          for i in range(i0, i1)]
                    tab = gts.table(ts, self.plan.params)
                    nsl = tab.shape[1]
                for i in range(i1 - i0):
                    if nsl:
                        gts.dev[:nsl] = tab[i]   # ordered on the graph's stream, before the replay
                    g["graph"].launch(g["stream"])
            g["stream"].synchronize()
        # Copies out on the caller's stream (the replay is finished): enqueued on the graph's
        # non-blocking stream, after its last synchronize, they raced with whatever the caller
        # ran next on its own stream (the reverse).
        self.X = g["X"].copy()
        if multistep:
            self._buffers_to_history(g)
        self.iteration += steps
        self.t = self._t0 + (self.iteration - self._it0) * self.dt
        if self._fused is not None:
            self._fused.check_real_projection()
        if self._finite_every:
            # one device reduction per run_graph call (no per-step host sync inside the graph:
            # a blow-up is reported at the end of the call, with the last iteration checked)
            self._check_finite("IVPSolver.run_graph (end of the call)")

    def _sbdf_steady(self) -> bool:
        """History long enough for a full-order multistep step (SBDF or CN; no start-up left)."""
        q = self._nhist
        startup_left = self.startup == "rk443" and self.iteration < q - 1
        return not startup_left and len(self._MXh) >= q - 1

    def _capture_graph(self) -> dict:
        cp = self.xp
        pool = cp.cuda.MemoryPool()          # private pool: graph temporaries never alias user arrays
        stream = cp.cuda.Stream(non_blocking=True)
        g: dict = dict(stream=stream, pool=pool)
        with cp.cuda.using_allocator(pool.malloc), stream:
            g["X"] = Xbuf = self.X.copy()
            if self.order:
                n, q = self.system.n, self.order
                g["MXh"] = cp.zeros((q, n), dtype=self.system.dtype)
                g["Fh"] = cp.zeros((q, n), dtype=self.system.dtype)
                self._history_to_buffers(g)
                self._sbdf_graph_step(g)                              # warm-up: plans, factors, pool
                self._history_to_buffers(g)
                Xbuf[...] = self.X
                stream.synchronize()
                stream.begin_capture()
                self._sbdf_graph_step(g)
            elif self.family == "cn":
                n, q = self.system.n, self._nhist
                for k in ("MXh", "LXh", "Fh"):
                    g[k] = cp.zeros((q, n), dtype=self.system.dtype)
                self._history_to_buffers(g)
                self._cn_graph_step(g)                                # warm-up: plans, factors, pool
                self._history_to_buffers(g)
                Xbuf[...] = self.X
                stream.synchronize()
                stream.begin_capture()
                self._cn_graph_step(g)
            elif self.time_parameter:        # time factors in device slots (_graph_time_support)
                gts = self._graph_time_support()
                Fst = gts.stage_F(self.t, self.dt, self._rk[2])
                gts.begin(capturing=False)
                try:
                    self.rk_stages(Xbuf, self.t, self.dt, self._rk, F=Fst)    # warm-up (kernels)
                    stream.synchronize()
                    gts.begin(capturing=True)
                    stream.begin_capture()
                    g["keep"] = Xn = self.rk_stages(Xbuf, self.t, self.dt, self._rk, F=Fst)[-1]
                    Xbuf[...] = Xn
                finally:
                    gts.end()
            else:
                self._rk_step(Xbuf, self.t, self.dt, self._rk)       # warm-up: plans, factors, pool
                stream.synchronize()
                stream.begin_capture()
                g["keep"] = Xn = self._rk_step(Xbuf, self.t, self.dt, self._rk)
                Xbuf[...] = Xn
            g["graph"] = stream.end_capture()
        return g

    def _graph_time_support(self):
        """Device slots of the time factors of F for a captured step (run_graph of a
        time-dependent F), or None: not supported by this solver (refused)."""
        return None

    def _sbdf_graph_step(self, g: dict) -> None:
        """One full-order SBDF step on the fixed buffers (mirror of :meth:`_sbdf_step`): rotate
        the history by copies, push M X^n and F(X^n), accumulate the RHS in the same order."""
        sysm, dt, q = self.system, self.dt, self.order
        X, MXh, Fh = g["X"], g["MXh"], g["Fh"]
        for j in range(q - 1, 0, -1):
            MXh[j] = MXh[j - 1]
            Fh[j] = Fh[j - 1]
        MXh[0] = sysm.M @ X
        Fh[0] = self.F(X, self.t)
        a, c = sbdf_coefficients(q)
        rhs = sbdf_rhs(self.xp, sysm.n, sysm.dtype,
                       [(c[j], Fh[j - 1], a[j] / dt, MXh[j - 1]) for j in range(1, q + 1)])
        X[...] = self._solve_state(a[0] / dt, 1.0, rhs)

    def _cn_graph_step(self, g: dict) -> None:
        """One full-history CN step on the fixed buffers (mirror of :meth:`_cn_step`)."""
        sysm, q = self.system, self._nhist
        X, MXh, LXh, Fh = g["X"], g["MXh"], g["LXh"], g["Fh"]
        for j in range(q - 1, 0, -1):
            MXh[j] = MXh[j - 1]
            LXh[j] = LXh[j - 1]
            Fh[j] = Fh[j - 1]
        MXh[0] = sysm.M @ X
        LXh[0] = sysm.L @ X
        Fh[0] = self.F(X, self.t)
        a, b, _c = self._cn
        X[...] = self._solve_state(a[0] / self.dt, b[0], self._cn_rhs(self._cn, MXh, LXh, Fh))

    def _history_to_buffers(self, g: dict) -> None:
        for j, (mx, f) in enumerate(zip(self._MXh, self._Fh, strict=True)):
            g["MXh"][j] = mx
            g["Fh"][j] = f
        if "LXh" in g:
            for j, lx in enumerate(self._LXh):
                g["LXh"][j] = lx

    def _buffers_to_history(self, g: dict) -> None:
        # After a full-order step the deque holds exactly ``_nhist`` entries (see _sbdf_step,
        # _cn_step; _nhist = order for SBDF).
        q = self._nhist
        self._MXh = deque(g["MXh"][j].copy() for j in range(q))
        self._Fh = deque(g["Fh"][j].copy() for j in range(q))
        if "LXh" in g:
            self._LXh = deque(g["LXh"][j].copy() for j in range(q))

    # ------------------------------------------------------------------ checkpoint / restart
    def _checkpoint_payload(self) -> tuple[dict, dict]:
        """(meta, host arrays) of a checkpoint: state, multistep history and clock. Shared by the
        two file formats (``.npz`` and HDF5 ``.h5``), so both restart bit-for-bit."""
        meta = dict(scheme=self.scheme, dt=self.dt, startup=self.startup,
                    startup_substeps=self.startup_substeps, n=self.system.n,
                    iteration=self.iteration, t=self.t, t_origin=self._t0, it_origin=self._it0,
                    reduced=self.system.reduced, ordering=self.system.ordering,
                    problem=self.plan.problem.name, nhist=len(self._MXh),
                    nhist_LX=len(self._LXh), bc_time=self.bc_time)
        # what the state is a state of (model, equations, parameters, resolution, geometry):
        # compared at restart, with the differences allowed so far
        from ..io._identity import checkpoint_identity
        meta["identity"] = checkpoint_identity(self)
        if getattr(self, "_restart_changes", None):
            meta["restart_changes"] = list(self._restart_changes)
        # the time the chain of this state started from (set_state(t=...), sim.initial(t=...)):
        # compared by resume_or_start(..., t=...); None when the state was set otherwise
        t_init = getattr(self, "_t_initial", None)
        meta["t_initial"] = None if t_init is None else float(t_init)
        h = (lambda a: a) if self.backend == "cpu" else self.xp.asnumpy
        arrays = {"X": h(self.X)}
        for i, (mx, f) in enumerate(zip(self._MXh, self._Fh, strict=True)):
            arrays[f"MX{i}"] = h(mx)
            arrays[f"F{i}"] = h(f)
        for i, lx in enumerate(self._LXh):              # CN family: L X history
            arrays[f"LX{i}"] = h(lx)
        return meta, arrays

    def save_checkpoint(self, path: str | Path, *, grids: bool = False) -> Path:
        """State + multistep history + clock: a restart continues bit-for-bit.

        Format chosen by the suffix: ``.h5`` / ``.hdf5`` writes an HDF5 file readable without
        Conatus (``conatus.io``: state, history, per-unknown coefficients, provenance; with
        ``grids=True`` also the grid values of each unknown); any other path is a NumPy ``.npz``
        (suffix added if missing, the historical format). Both are written to a temporary file
        then renamed, so a job killed while writing never leaves a truncated checkpoint."""
        path = Path(path)
        check_checkpoint_suffix(path, "save_checkpoint")
        from ..errors import check_flag
        grids = check_flag(grids, "grids", "save_checkpoint()")
        if self._finite_every:                 # never write a NaN checkpoint
            self._check_finite(f"save_checkpoint({path.name})")
        meta, arrays = self._checkpoint_payload()
        if path.suffix in (".h5", ".hdf5"):
            from ..io.checkpoint import write_checkpoint
            return write_checkpoint(path, meta, arrays, solver=self, grids=grids)
        if grids:
            raise OptionError("save_checkpoint(grids=True) needs an HDF5 path (.h5)",
                              fix="save_checkpoint('state.h5', grids=True)",
                              doc="docs/site/guide/io.md")
        if not str(path).endswith(".npz"):
            path = Path(str(path) + ".npz")
        part = path.with_name(path.name[:-4] + ".part.npz")
        np.savez(part, meta=json.dumps(meta), **arrays)
        os.replace(part, path)
        return path

    def load_checkpoint(self, path: str | Path, *, allow: Any = ()) -> None:
        """Restart from :meth:`save_checkpoint` (``.npz`` or ``.h5``): every numerical option
        must match the solver's (a restart never changes the numerics silently), and so must what
        the state is a state of: model, equations, parameters, resolution and geometry, and the data of every given field and the user functions (0.1.1;
        ``conatus.io._identity``). ``allow`` names what may differ deliberately: parameter,
        given-field or function names (``allow=("kappa",)``) or ``"parameters"``, ``"data"``,
        ``"functions"``, ``"model"``, ``"equations"``, ``"geometry"``; an allowed difference is
        printed by :meth:`explain` and carried by every later checkpoint."""
        from ..io._identity import allowable_names, check_allow, compare_identity
        # parameter, given-field and function names (0.1.1, lot A: given data are compared)
        allow = check_allow(allow, allowable_names(self), "load_checkpoint")
        path = Path(path)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        from ..io._h5 import require_file
        require_file(path, "load_checkpoint")
        if path.suffix in (".h5", ".hdf5"):
            from ..io.checkpoint import read_checkpoint
            meta, arrays = read_checkpoint(path)
        else:
            with np.load(path) as d:
                meta = json.loads(str(d["meta"]))
                arrays = {k: d[k] for k in d.files if k != "meta"}
        changed = compare_identity(meta, self, allow, "load_checkpoint", path)
        self._apply_checkpoint(meta, arrays)
        self._restart_changes = list(meta.get("restart_changes", [])) + [
            f"restart from {path.name} (t = {float(meta['t']):g}) with {change} (allowed by "
            f"allow={allow!r})" for change in changed]
        # read by cn.io.CheckpointSeries: is a checkpoint directory this run's own?
        self._restart_path, self._restart_iteration = path, self.iteration

    def _apply_checkpoint(self, meta: dict, arrays: Mapping[str, np.ndarray]) -> None:
        for key in ("scheme", "dt", "startup", "startup_substeps", "n", "reduced", "ordering"):
            mine = dict(scheme=self.scheme, dt=self.dt, startup=self.startup,
                        startup_substeps=self.startup_substeps, n=self.system.n,
                        reduced=self.system.reduced, ordering=self.system.ordering)[key]
            if mine != meta[key]:
                raise OptionError(f"checkpoint mismatch on {key}: {meta[key]!r} vs {mine!r} "
                                  "(a restart never changes the numerics silently)",
                                  fix=f"build the solver with {key}={meta[key]!r} to restart "
                                      "from this checkpoint", doc="docs/site/guide/timesteppers.md")
        if self.time_parameter and self._bc_data() is not None:
            old = meta.get("bc_time", "step")         # older checkpoints: the step convention
            if old != self.bc_time:
                raise OptionError(f"checkpoint mismatch on bc_time: {old!r} vs {self.bc_time!r} "
                                  "(time-dependent boundary data; a restart never changes the "
                                  "numerics silently)",
                                  fix=f"build the solver with bc_time={old!r} to restart from "
                                      "this checkpoint", doc="docs/site/guide/timesteppers.md")
        dev = (lambda a: np.array(a, copy=True)) if self.backend == "cpu" else self.xp.asarray
        self.X = dev(arrays["X"])
        self._MXh = deque(dev(arrays[f"MX{i}"]) for i in range(meta["nhist"]))
        self._Fh = deque(dev(arrays[f"F{i}"]) for i in range(meta["nhist"]))
        self._LXh = deque(dev(arrays[f"LX{i}"]) for i in range(meta.get("nhist_LX", 0)))
        self.iteration, self.t = int(meta["iteration"]), float(meta["t"])
        self._t_origin, self._it_origin = float(meta["t_origin"]), int(meta["it_origin"])
        t_init = meta.get("t_initial")             # None: written before the record
        self._t_initial = None if t_init is None else float(t_init)

    def _repr_html_(self) -> str:       # Jupyter display (conatus.notebook)
        from ..notebook import explain_html
        return explain_html(self)

    def __repr__(self) -> str:
        """One line: scheme, step, backend, size, where the run is. Safe on a solver
        whose construction stopped half-way (it is printed in tracebacks)."""
        def g(name: str):
            return getattr(self, name, None)
        n = getattr(g("system"), "n", "?")
        t, dt = g("t"), g("dt")
        return (f"{type(self).__name__}(scheme={g('scheme')!r}, "
                f"dt={dt if dt is None else format(dt, 'g')}, backend={g('backend')!r}, unknowns={n}, "
                f"t={t if t is None else format(t, 'g')}, iteration={g('iteration')}; .explain())")

    # ------------------------------------------------------------------ explain
    @property
    def effective_order(self) -> int:
        """Global order of the run with its startup: the nominal order of
        the scheme, except that ``startup='lower_order'`` starts a multistep scheme with a
        first-order step (SBDF1, CNAB1), whose O(dt^2) local error caps the global order at 2
        (SBDF3/SBDF4 converge at order 2, not 3/4); ``startup='rk443'`` keeps the nominal order."""
        p = _SCHEME_INFO[self.scheme][3]
        if getattr(self, "family", "rk") != "rk" and getattr(self, "startup", None) == "lower_order":
            p = min(p, 2)
        return min(p, self._bc_order_cap())

    #: Global order of a scheme with time-dependent boundary data, per bc_time mode, where it is
    #: below the scheme's (audit invariance; measured on u_t = u_yy / 2 with both walls
    #: moving, dt = 1/40 .. 1/640, tests/invariance/test_inv_time_order.py): 'step' (Dedalus) carries g
    #: with the explicit weights, first order for RK222/RK443/RKSMR (1.02 .. 1.24), and CNAB2 /
    #: CNLF2 lose their second order without a clean rate (1.4 .. 2.3: stated 1, as the guide's
    #: "about first order"); 'stage' gives RK443 its stage order 2. 'rk' keeps the nominal orders
    #: (RK443: 2.73 .. 2.91, converging to 3).
    _BC_ORDER = {"step": {"RK222": 1, "RK443": 1, "RKSMR": 1, "CNAB1": 1, "RK111": 1,
                          "CNAB2": 1, "CNLF2": 1},
                 "stage": {"RK443": 2}}

    def _bc_order_cap(self) -> int:
        """The order cap of the boundary-data convention (99 without time-dependent data)."""
        if not getattr(self, "time_parameter", None) or self._bc_data() is None:
            return 99
        try:
            mode = self._bc_resolve()[0]
        except OptionError:                   # refused data: the refusal is the message
            return 99
        return self._BC_ORDER.get(mode, {}).get(self.scheme, 99)

    def _startup_text(self) -> str:
        """", startup=..." for a multistep scheme (what the first steps do), "" for RK."""
        if self.family == "rk":
            return ""
        if self.startup == "rk443":
            return f", startup=rk443 ({self.startup_substeps} RK443 substeps)"
        if self.family == "cn" and self._nhist > 1:
            return ", startup=lower_order (first step CNAB1)"
        if self.effective_order < _SCHEME_INFO[self.scheme][3]:
            return (f", startup=lower_order (first steps SBDF1..SBDF{self.order - 1}: effective "
                    f"global order {self.effective_order}, not {self.order}; startup='rk443' "
                    "keeps it)")
        return f", startup={self.startup}"

    def _scheme_text(self) -> str:
        eff = getattr(self, "effective_order", None) if self.scheme in _SCHEME_INFO else None
        note = ("" if eff is None or eff >= _SCHEME_INFO[self.scheme][3] else
                "\n  " + self._effective_order_text(eff))
        return "\n  scheme: " + scheme_description(self.scheme, eff) + note

    def _effective_order_text(self, eff: int) -> str:
        """Why the global order is ``eff``, below the scheme's: the start-up or the boundary data."""
        if self._bc_order_cap() <= eff:
            mode = self._bc_resolve()[0]
            return (f"effective global order {eff} with time-dependent boundary data under "
                    f"bc_time='{mode}'" + (" (the Dedalus convention); bc_time='auto' imposes them "
                                           "at the stage times" if mode == "step" else
                                           "; bc_time='rk' keeps the nominal order"))
        return (f"effective global order {eff} with startup='lower_order' (first step first "
                "order); startup='rk443' gives the nominal order")

    def explain(self, *, full: bool = False) -> str:
        """Scheme, step, backend and the finite-state check, in a few lines;
        ``full=True``: also the system layout (groups, buckets, bands), the evaluator of F,
        boundary data and what runs where. Subclasses write their full text in
        ``_explain_full``."""
        text = self._explain_full()
        if full:
            return text
        head = text.splitlines()[0]
        eff = getattr(self, "effective_order", None) if self.scheme in _SCHEME_INFO else None
        lines = [head, "  scheme: " + scheme_description(self.scheme, eff)]
        if eff is not None and eff < _SCHEME_INFO[self.scheme][3]:
            # the start-up or the boundary data cap the observed order (review finding 17, audit
            # invariance): said where the order is
            lines.append("  " + self._effective_order_text(eff))
        if getattr(self, "_fused", None) is not None or getattr(self, "_sfused", None) is not None:
            lines.append("  F: fused kernels (f_backend='fused')")
        # fallbacks and accuracy notes of F are warnings: kept in the short text (
        # coefficient_time='step_start' is first order in time, "parity reduction not used")
        for note in (getattr(self, "_fused_note", ""), getattr(self, "_fc2d_note", "")):
            if note:
                lines.append("  " + note)
        # the boundary-data line carries the order of the scheme with time-dependent boundary
        # values (bc_time='step': about first order; 'stage': RK443 order 2): kept whole
        bc = self._bc_text().strip()
        if bc:
            lines.append("  " + bc)
        lines += ["  " + n for n in self._short_notes() if n]
        lines += self._restart_lines()
        lines.append(self._finite_text())
        return "\n".join(lines)

    def _restart_lines(self) -> list[str]:
        """explain() lines of the differences a restart was allowed to make (``allow=``)."""
        return ["  restarted with a declared change: " + c
                for c in getattr(self, "_restart_changes", None) or []]

    def _short_notes(self) -> list[str]:
        """Decisions and warnings that the short explain() keeps (projections applied to the
        solution, spurious-mode checks); subclasses add theirs."""
        rot = [self._rotation_text()] if getattr(self, "_rotation", None) is not None else []
        sym = getattr(getattr(getattr(self, "plan", None), "problem", None), "symmetry", None)
        if sym is None:
            return rot
        return rot + ["symmetry subspace: the state is restricted to it (it hides symmetry-breaking "
                "instabilities: check in the full space; explain(full=True) for the reduction)"]

    def _explain_full(self) -> str:
        """The full explain() text of this solver."""
        st = self._startup_text()
        return (f"IVPSolver(scheme={self.scheme}, dt={self.dt:g}, constant step{st}, "
                f"backend={self.backend})" + self._scheme_text() + "\n"
                + self.system.explain()
                + f"\n  F: pseudospectral {'NumPy' if self.backend == 'cpu' else 'CuPy (GPU)'} evaluator, products on each basis' declared "
                  "dealiased grid; linear solves: C++ bordered banded, batched per bucket"
                + ("" if self._fused is None else "\n" + self._fused_text())
                + (f"\n  {self._fused_note}" if getattr(self, "_fused_note", "") else "")
                + ("\n" + self._fc2d.explain() if getattr(self, "_fc2d", None) is not None else "")
                + (f"\n  {self._fc2d_note}" if getattr(self, "_fc2d_note", "") else "")
                + self._bc_text()
                + ("\n  " + self._rotation_text() if getattr(self, "_rotation", None) else "")
                + "\n" + self._explain_step_loop()
                + "".join("\n" + r for r in self._restart_lines())
                + "\n" + self._finite_text())

    #: class default for solvers whose __init__ does not set it (subclasses built elsewhere)
    _finite_every = FINITE_CHECK_EVERY

    def _check_finite(self, where: str) -> None:
        """The finite-state check of this solver (one reduction over the state), then the
        run-time resolution check (:meth:`_check_tails`); the distributed solver overrides it
        with a collective decision."""
        check_state_finite(self, where)
        if not getattr(self, "_tail_warned", False):
            self._check_tails()

    def _tail_plan(self) -> list | None:
        """(variable index, name, [(coord, axis, basis)]) of the unknowns the run-time
        resolution check measures; None when the plan is not a Cartesian one (polar, sphere:
        declared by explain())."""
        if "_tail_dirs" in self.__dict__:
            return self._tail_dirs
        from ..discretize.plan import Plan
        from ..ir.tails import basis_family as _family
        out = None
        plan = getattr(self, "plan", None)
        if isinstance(plan, Plan) and not getattr(plan, "is_polar", False) \
                and not getattr(plan, "is_sphere", False):
            out = []
            taus = set(getattr(plan, "taus", ()) or ())
            coords = plan.domain.coordsys.coords
            for k, v in enumerate(self.system.variables):
                if v in taus or v.domain is None:
                    continue
                dirs = []
                for i, c in enumerate(coords):
                    b = v.domain.basis(c)
                    if b is None or b.size <= 2:
                        continue
                    try:
                        _family(b)
                    except Exception:  # noqa: BLE001 -- a basis without mode numbers is not judged
                        continue
                    dirs.append((c.name, v.rank + i, b))
                if dirs:
                    out.append((k, v.name, dirs))
        self._tail_dirs = out
        return out

    def _check_tails(self) -> None:
        """Run-time resolution check (RUN_TAIL_TOL): the relative
        spectral tail of each unknown along each direction, from the coefficients in memory (one
        reduction per unknown and direction; on the GPU the small envelopes come back in one
        transfer, at the cadence of the finite check, which already synchronizes). Warns once
        per solver; a detection, never a change."""
        dirs = self._tail_plan()
        if not dirs:
            return
        import math

        from ..ir.tails import envelope_tail, mode_envelope, mode_numbers, tail_window
        xp = getattr(self, "xp", np)
        cs = self.system.scatter(self.X, only=frozenset(n for _k, n, _d in dirs))
        envs, meta = [], []
        for k, name, dd in dirs:
            a = xp.abs(cs[k])
            for cname, axis, b in dd:
                e = xp.moveaxis(a, axis, 0).reshape(a.shape[axis], -1).max(axis=1)
                envs.append(e)
                meta.append((name, cname, b))
        host = xp.concatenate(envs)
        host = host.get() if hasattr(host, "get") else np.asarray(host)
        top = float(host.max()) if host.size else 0.0
        if not top > 0 or not math.isfinite(top):
            return
        bad, short, pos = [], [], 0
        for (name, cname, b), e in zip(meta, envs, strict=True):
            n = int(e.shape[0])
            raw = host[pos:pos + n]
            pos += n
            mnum, M = mode_numbers(b)
            E = mode_envelope(raw, mnum, M)
            emax = float(E.max()) if E.size else 0.0
            if not emax > RUN_TAIL_FLOOR * top:
                continue
            # a tail window that covers every mode number holds the mean too: its "tail" is 1
            # whatever the data (earlier versions: a constant on N = 4 Fourier modes,
            # M = 2, was "under-resolved, tail 1.0"): not judged, like a too-short window
            if (M - tail_window(M) < 1
                    or int(np.count_nonzero(mnum >= M - tail_window(M))) < RUN_TAIL_MIN_MODES):
                short.append(f"{name} along {cname} (N = {b.size})")
                continue
            tail = envelope_tail(E, emax)
            if tail > RUN_TAIL_TOL:
                bad.append(f"{name} along {cname}: tail {tail:.1e} (N = {b.size})")
        if short and not getattr(self, "_tail_short", None):
            self._tail_short = ("not judged (tail window of fewer than "
                                f"{RUN_TAIL_MIN_MODES} coefficients): " + "; ".join(short))
        if bad:
            self._tail_warned = True
            self._tail_note = (f"under-resolved at t = {self.t:.6g} (iteration {self.iteration}): "
                               + "; ".join(bad))
            from ..errors import UnderResolvedWarning, _warn_user
            _warn_user(
                f"the run is under-resolved at t = {self.t:.6g}: " + "; ".join(bad)
                + f" (relative spectral tail above RUN_TAIL_TOL = {RUN_TAIL_TOL:g}; an amplitude: "
                "the largest of the last 10 % of the coefficients over the largest one, so the "
                "fields are not resolved to better than about the tail itself); its "
                "numbers are not converged in space: raise N along these directions "
                "(resolution_report(sim.plan, sim.solver) estimates how many modes). Warned "
                "once; recorded by explain().", UnderResolvedWarning)

    def _finite_text(self) -> str:
        """explain() line of the finite-state check."""
        if not self._finite_every:
            return ("  finite-state check: off (check_finite=False): a blown-up run steps on "
                    "with NaN values")
        dflt = " (default)" if self._finite_every == FINITE_CHECK_EVERY else ""
        text = (f"  finite-state check: every {self._finite_every} steps{dflt}, at the end of "
                "each run call and before each checkpoint/output write (NaN/inf -> "
                "NumericalError naming step, time and unknown; check_finite=False turns it off)")
        if getattr(self, "_tail_note", None):
            text += f"\n  resolution check: {self._tail_note} (warned once)"
        elif self._tail_plan():
            text += (f"\n  resolution check: with the finite-state check, the relative spectral "
                     f"tail of each unknown and direction above {RUN_TAIL_TOL:g} warns once "
                     "(none so far)")
        else:
            text += ("\n  resolution check: not available for this plan (no Cartesian spectral "
                     "direction)")
        if getattr(self, "_tail_short", None):       # directions left out (earlier versions: this
            text += f"\n  resolution check: {self._tail_short}"   # line used to break the chain)
        return text

    def _fused_text(self) -> str:
        """explain() of the fused evaluator; in a reflection subspace with the 2D pipeline, the
        parity-reduced evaluator is only the fallback (F runs on the full grid, restricted): it
        used to be printed as if it computed F ("products on the half grid")."""
        txt = self._fused.explain()
        fz = getattr(self, "_fc2d_fused", None)
        if getattr(self, "_fc2d", None) is not None and fz is not None and fz is not self._fused:
            txt = ("fallback evaluator of F (not used while the 2D pipeline below runs): "
                   + txt)
        return txt

    def _explain_step_loop(self) -> str:
        """How one step is executed (what runs where), for explain()."""
        if self.backend != "cuda":
            return ("  step loop: Python loop over steps, NumPy array operations for the IMEX "
                    "right-hand sides, C++ banded solves per bucket")
        rhs = {"rk": "RK stage right-hand side MX0 + sum dt a F - dt h L X",
               "sbdf": "SBDF right-hand side sum c F - (a/dt) M X",
               "cn": "CN multistep right-hand side sum c F - (a/dt) M X - b L X"}[self.family]
        return ("  step loop: Python loop over steps; step() launches every kernel from the host, "
                "run_graph(n) captures one step as a CUDA graph and replays it once per step "
                "(same kernels, same arithmetic; constant dt, F independent of t)\n"
                f"  per stage: {rhs} in one kernel for real float64 states (M X and L X applied "
                "inside it, roundings "
                "identical to the array operations; timestep/lincomb.py), then the batched solve")
