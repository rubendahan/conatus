"""Fused (compiled) evaluation of a time-dependent right-hand side, without freezing t.

``FusedRHS`` (``timestep/fused.py``) compiles the RHS with its parameters frozen, so
``IVPSolver(..., f_backend="fused")`` refuses ``time_parameter``. For a RHS that splits exactly as

    F(X, t) = F_0(X) + sum_g a_g(t) F_g(X)          (:func:`conatus.adjoint.rhs_transpose.split_time`)

the time coefficients a_g(t) are spatial constants. :class:`StageTimeFused` rewrites each RHS as
F_0 + sum_g c_g F_g with c_g a *constant field* (a field on the empty domain, like a pressure
gauge) and compiles that with the unmodified ``FusedRHS``; at every call the value a_g(t) at the
stage time is passed as the data of c_g. The product c_g F_g is then formed on the dealiased
grid by the fused kernel: the same numbers as the evaluator up to rounding (tested), t is never
frozen. It is attached to an evaluator solver (``IVPSolver(..., time_parameter="t",
f_backend="evaluator")``) by :func:`use_stage_time_fused`, which replaces the solver's F by this
object (declared in ``explain()``); FlowMap then uses it for the forward map (tangent and
adjoint keep the matrix-free linearization, as for any fused solver). CUDA-graph capture
(``run_graph``) stays refused: the coefficients are host scalars set per stage.

``coefficient_time="step_start"`` (opt-in, comparison only) evaluates a_g at the start time of
the current step for all its stages: the convention of a Dedalus script that updates a
time-dependent NCC once per step (the reference bracket). It is a
different, first-order-in-time discretization of the forcing, used only to *measure* the effect
of that method difference; the default ``"stage"`` is the consistent one. It is refused for any
call whose time is not inside the solver's current step (e.g. a FlowMap replay), so it cannot
leak into the adjoint.
"""
from __future__ import annotations

import numpy as np

from ..ir import Domain
from ..ir import expr as E
from ..timestep.fused import FusedRHS
from .rhs_transpose import eval_scalar, split_time
from ..errors import OptionError

__all__ = ["StageTimeFused", "use_stage_time_fused"]


class StageTimeFused:
    """Duck-typed replacement of ``IVPSolver._fused`` for a time-dependent RHS (see module)."""

    def __init__(self, solver, coefficient_time: str = "stage") -> None:
        if coefficient_time not in ("stage", "step_start"):
            raise OptionError("coefficient_time must be 'stage' (time coefficients at every RK stage "
                              "time, the consistent default) or 'step_start' (frozen during a step, "
                              "to compare with a Dedalus script that updates its NCC per step)",
                              fix="coefficient_time='stage' (default) or 'step_start'")
        self.coefficient_time = coefficient_time
        tname = solver.time_parameter
        if not tname:
            raise OptionError("StageTimeFused: the solver has no time parameter (use "
                              "f_backend='fused' directly)",
                              fix="IVPSolver(..., f_backend='fused') directly (F does not depend on t)")
        self.tname, self.solver, self.xp = tname, solver, solver.xp
        plan = solver.plan
        cs = plan.domain.coordsys
        self.coefs: list = []                      # a_g (field-free expressions of t)
        self.cfields: list = []                    # c_g (constant fields standing for a_g)
        exprs = []
        for e in solver._rhs_exprs:
            if e is None:
                exprs.append(None)
                continue
            terms = []
            for a, se in split_time(e, tname):
                if a is None:
                    terms.append(se)
                    continue
                if a not in self.coefs:
                    self.coefs.append(a)
                    self.cfields.append(E.Field(f"_stage_coef{len(self.coefs)}", Domain(cs, [])))
                terms.append(E.mul(self.cfields[self.coefs.index(a)], se))
            exprs.append(E.add(*terms) if terms else None)
        self.exprs = exprs                         # the rewritten RHS (memory planner input)
        self.params = {k: v for k, v in dict(plan.params).items() if k != tname}
        self.inner = FusedRHS(plan.domain, plan.problem.equations, exprs, self.params,
                              xp=None if solver.backend == "cpu" else self.xp,
                              dtype=solver.system.dtype)
        self._t = None

    def check_params(self, params) -> None:
        """Record the stage time; recompile only if another parameter changed."""
        p = dict(params)
        self._t = p.pop(self.tname)
        self.inner.check_params(p)

    def values(self, t: float) -> list:
        params = dict(self.solver.plan.params)
        params[self.tname] = t
        return [eval_scalar(a, params) for a in self.coefs]

    def coefficient_time_of(self, t_stage: float) -> float:
        """Time at which the coefficients are evaluated for a call at ``t_stage``."""
        if self.coefficient_time == "stage":
            return t_stage
        t0, dt = float(self.solver.t), float(self.solver.dt)
        # step_start: the stage must belong to the step the solver is taking (t0 is the step start
        # while rk_stages runs); anything else (a FlowMap replay, a direct F call) is refused.
        if not (t0 - 1e-12 * max(1.0, abs(t0)) <= t_stage <= t0 + dt * (1 + 1e-9)):
            raise RuntimeError(
                f"StageTimeFused(coefficient_time='step_start'): F called at t = {t_stage!r} outside "
                f"the current step [{t0!r}, {t0 + dt!r}]; this mode is for forward IVPSolver.step "
                "runs only (use coefficient_time='stage' for FlowMap / adjoints)")
        return t0

    def evaluate(self, data) -> list:
        if self._t is None:
            raise RuntimeError("StageTimeFused.evaluate before check_params (no stage time)")
        d = dict(data)
        for c, v in zip(self.cfields, self.values(self.coefficient_time_of(self._t))):
            d[c] = self.xp.asarray(np.float64(v) if np.isrealobj(v) else np.complex128(v))
        return self.inner.evaluate(d)

    def check_real_projection(self) -> None:
        self.inner.check_real_projection()

    def explain(self) -> str:
        when = (f"at every stage time ({self.tname} never frozen)" if self.coefficient_time == "stage"
                else "at the START of each step, frozen over its stages (coefficient_time="
                     "'step_start': first order in time in the forcing, comparison mode only)")
        return (f"StageTimeFused: fused F with {len(self.coefs)} time coefficient(s) "
                f"{', '.join(str(a) for a in self.coefs)} passed as constant fields {when}\n"
                + self.inner.explain())


def use_stage_time_fused(solver, coefficient_time: str = "stage",
                         transform_batch: int | None = None) -> StageTimeFused:
    """Replace the (evaluator) F of a time-dependent ``IVPSolver`` by :class:`StageTimeFused`
    (explicit opt-in; the solver's ``explain()`` shows it). Returns the attached object.
    ``coefficient_time``: ``"stage"`` (default, consistent) or ``"step_start"`` (comparison with a
    per-step NCC update; forward runs only, see the module docstring). ``transform_batch``: at
    most that many fields per batched transform in F and in the tangent/adjoint transposes
    (``FusedRHS.transform_batch``: smaller transform temporaries, same per-field transforms;
    None: all fields at once)."""
    if solver._fused is not None:
        raise OptionError("use_stage_time_fused: the solver already has a fused F",
                          fix="call use_stage_time_fused on a solver without f_backend='fused'")
    if transform_batch is not None and (int(transform_batch) != transform_batch or transform_batch < 1):
        raise OptionError(f"transform_batch must be a positive integer or None, got {transform_batch!r}",
                          fix="transform_batch=None or a positive integer, e.g. 4")
    f = StageTimeFused(solver, coefficient_time)
    f.inner.transform_batch = None if transform_batch is None else int(transform_batch)
    solver._fused = f
    solver._fused_note = ("F: StageTimeFused (time coefficients as constant fields at the stage times)"
                          if coefficient_time == "stage" else
                          "F: StageTimeFused with coefficient_time='step_start' (time coefficients "
                          "frozen at the start of each step: comparison mode, first order in time)")
    return f
