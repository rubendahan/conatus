"""Matrix-free pseudo-arclength continuation: the corrector is Newton–Krylov–hookstep
(:class:`conatus.ecs.NewtonKrylovHookstep`) on the bordered system, so only residuals and
Jacobian–vector products are needed (large 3D problems where the dense/sparse Jacobian of
:class:`conatus.continuation.PseudoArclength` is out of reach).

Unknowns z = (x, p), norm ‖z‖² = ‖x‖² + (w_p p)² with a **declared** parameter weight w_p. From
the last two converged points the tangent is the secant t = (z_k − z_{k−1}) / ‖·‖, the predictor
z_pred = z_k + ds t, and the corrector solves

    G(x, p) = 0,   ⟨t, z − z_pred⟩ = 0      (x-part plain dot, p-part weighted by w_p²).

A **fold** (limit point in p) is a sign change of Δp between consecutive steps. It is located by
restarting from the point before it with a step divided by ``fold_refine_factor`` (repeated
``fold_refinements`` times), then by the vertex of the parabola p(s) through the three points
around the turning point; the change of the vertex between the last two refinements is reported
as its uncertainty, next to the corrector residuals. Observables (``observe(x, p)``) are
interpolated at the vertex with the same parabola. Every setting is explicit (``explain()``)."""
from __future__ import annotations

import numbers
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .._deprecation import positional_order
from ..ecs.newton import NewtonKrylovHookstep, NewtonNotConverged
from ..errors import OptionError
from .core import _count, real_state
from .._repr import compact_repr

__all__ = ["KrylovContinuationSettings", "KrylovBranchPoint", "KrylovArclength"]


#: Declared defaults of the algorithm controls of KrylovContinuationSettings.
KRYLOV_CONTINUATION_DEFAULTS = dict(adapt=1.3, fast_iterations=3, p_weight=1.0,
                                    fold_refinements=3, fold_refine_factor=4.0)


@dataclass(frozen=True, kw_only=True)
class KrylovContinuationSettings:
    """Steps, budget, parameter weight and fold refinement of :class:`KrylovArclength` (checked
    at construction like :class:`ContinuationSettings`). Problem-scale settings (``ds``,
    ``ds_min``, ``ds_max``, ``max_points``) are required; the algorithm controls have declared
    defaults, marked "(default)" by ``explain()``."""

    ds: float                 # initial arclength step (in the weighted norm)
    ds_min: float
    ds_max: float
    max_points: int
    adapt: float = 1.3        # ds *= adapt after a corrector that took <= fast_iterations
    fast_iterations: int = 3
    p_weight: float = 1.0     # w_p: weight of the parameter in the norm
    fold_refinements: int = 3     # number of step refinements around a fold
    fold_refine_factor: float = 4.0

    def __post_init__(self) -> None:
        # was unchecked: ds_min > ds, adapt <= 1 or a fold_refine_factor <= 1 only showed up as a
        # stalled or degenerate run
        where = "KrylovContinuationSettings"
        if not (0 < self.ds_min <= self.ds <= self.ds_max):
            raise OptionError(f"{where}: need 0 < ds_min <= ds <= ds_max (ds = {self.ds!r}, "
                              f"ds_min = {self.ds_min!r}, ds_max = {self.ds_max!r})",
                              fix="e.g. ds=0.1, ds_min=1e-6, ds_max=0.2")
        if not self.adapt > 1:
            raise OptionError(f"{where}: adapt={self.adapt!r} must be > 1", fix="e.g. adapt=1.3")
        for name in ("fast_iterations", "max_points"):
            _count(getattr(self, name), name, where)
        if isinstance(self.fold_refinements, bool) \
                or not isinstance(self.fold_refinements, numbers.Integral) or self.fold_refinements < 0:
            raise OptionError(f"{where}: fold_refinements={self.fold_refinements!r} must be an "
                              "integer >= 0", fix="e.g. fold_refinements=3")
        if not self.fold_refine_factor > 1:
            raise OptionError(f"{where}: fold_refine_factor={self.fold_refine_factor!r} must be > 1",
                              fix="e.g. fold_refine_factor=4.0")
        if not self.p_weight > 0:
            raise OptionError(f"{where}: p_weight={self.p_weight!r} must be > 0",
                              fix="the weight of the parameter in the arclength norm, e.g. 1.0")

    def explain(self) -> str:
        """Every setting on one line; defaults marked."""
        def d(name: str) -> str:
            return " (default)" if getattr(self, name) == KRYLOV_CONTINUATION_DEFAULTS[name] else ""
        return (f"  ds = {self.ds:g} in [{self.ds_min:g}, {self.ds_max:g}], x{self.adapt:g}"
                f"{d('adapt')} when the corrector takes <= {self.fast_iterations}"
                f"{d('fast_iterations')} its, /2 on failure; max points {self.max_points}; "
                f"parameter weight w_p = {self.p_weight:g}{d('p_weight')}; fold: "
                f"{self.fold_refinements}{d('fold_refinements')} refinements by "
                f"/{self.fold_refine_factor:g}{d('fold_refine_factor')}, parabola vertex")


# v0.1 positional order, accepted for one cycle with a DeprecationWarning (conatus._deprecation)
positional_order(KrylovContinuationSettings, (
    "ds", "ds_min", "ds_max", "adapt", "fast_iterations", "max_points", "p_weight",
    "fold_refinements", "fold_refine_factor"))


@dataclass
class KrylovBranchPoint:
    """A converged point: arclength, state, parameter, reported residual, observables."""
    __repr__ = compact_repr   # one line, arrays by shape

    s: float
    x: np.ndarray
    p: float
    residual: float           # reported residual (``report`` callback) at convergence
    iterations: int
    observables: dict = field(default_factory=dict)


class KrylovArclength:
    """``residual(x, p)``, ``jvp(x, v, p)`` (∂G/∂x v), ``dGdp(x, p)``; ``newton``: corrector
    settings; ``report(x, p)``: the residual reported for each point (e.g. the unpreconditioned
    full-space residual); ``observe(x, p)`` optional dict of scalars recorded per point."""

    def __init__(self, residual: Callable, jvp: Callable, dGdp: Callable,
                 settings: KrylovContinuationSettings, newton: NewtonKrylovHookstep, *,
                 report: Callable, observe: Callable | None = None, log=None,
                 on_point: Callable | None = None) -> None:
        self.G, self.Jv, self.Gp = residual, jvp, dGdp
        self.cfg, self.newton = settings, newton
        self.report, self.observe, self.log = report, observe, log
        self.on_point = on_point          # called after every accepted branch point (e.g. I/O)
        self.points: list[KrylovBranchPoint] = []
        self.folds: list[dict] = []
        self.refused: list[tuple[float, str]] = []
        self.floor_failures: list[float] = []   # corrector failures stagnating within 10x of tol
        self.stop_reason = ""                     # set by run()

    # ------------------------------------------------------------------ pieces
    def _norm(self, dx, dp) -> float:
        return float(np.sqrt(np.dot(dx, dx) + (self.cfg.p_weight * dp) ** 2))

    def _secant(self, a: KrylovBranchPoint, b: KrylovBranchPoint):
        dx, dp = b.x - a.x, b.p - a.p
        nrm = self._norm(dx, dp)
        return dx / nrm, dp / nrm

    def _correct(self, x_pred, p_pred, tx, tp):
        w2 = self.cfg.p_weight ** 2
        n = x_pred.size

        def R(z):
            x, p = z[:n], z[n]
            return np.concatenate([self.G(x, p), [np.dot(tx, x - x_pred) + w2 * tp * (p - p_pred)]])

        def J(z, d):
            x, p = z[:n], z[n]
            return np.concatenate([self.Jv(x, d[:n], p) + d[n] * self.Gp(x, p),
                                   [np.dot(tx, d[:n]) + w2 * tp * d[n]]])
        z0 = np.concatenate([x_pred, [p_pred]])
        res = self.newton.solve(R, J, z0, scale=max(1.0, float(np.linalg.norm(x_pred))))
        return res.z[:n], float(res.z[n]), res.iterations

    def _point(self, s, x, p, its) -> KrylovBranchPoint:
        obs = self.observe(x, p) if self.observe else {}
        pt = KrylovBranchPoint(s, x, p, float(self.report(x, p)), its, obs)
        if self.log:
            self.log(f"point s = {s:.6g}: p = {p:.10g}, residual {pt.residual:.2e}, {its} its, {obs}")
        return pt

    def _step(self, a, b, ds):
        tx, tp = self._secant(a, b)
        x, p, its = self._correct(b.x + ds * tx, b.p + ds * tp, tx, tp)
        return self._point(b.s + ds, x, p, its)

    # ------------------------------------------------------------------ driver
    def run(self, x0, p0: float, x1, p1: float, *, p_bounds: tuple[float, float]) -> list:
        """Continue from two converged points (x0, p0), (x1, p1) (the direction of travel).

        A complex state is refused unless its imaginary part is round-off (``core.IMAG_ROUNDOFF``);
        it used to be dropped with a mere ComplexWarning."""
        cfg = self.cfg
        x0 = real_state(x0, "KrylovArclength.run")
        x1 = real_state(x1, "KrylovArclength.run")
        a = self._point(0.0, x0, float(p0), 0)
        b = self._point(self._norm(x1 - x0, p1 - p0), x1, float(p1), 0)
        self.points = [a, b]
        ds = cfg.ds
        while len(self.points) < cfg.max_points:
            a, b = self.points[-2], self.points[-1]
            try:
                c = self._step(a, b, ds)
            except (NewtonNotConverged, FloatingPointError, np.linalg.LinAlgError) as e:
                self.refused.append((ds, str(e)[:200]))
                if getattr(e, "stagnated_near_tol", False):
                    # the corrector stalled just above tol: halving ds does not help and the step
                    # never grows back (seen on the 3D Couette continuations). Counted
                    # and said once; the numerics are unchanged.
                    self.floor_failures.append(float(e.relative_residual))
                    if self.log and len(self.floor_failures) == 1:
                        self.log("continuation: the corrector stagnates at ‖R‖/scale = "
                                 f"{e.relative_residual:.2e} just above its tolerance "
                                 f"{e.tol:g}; steps will stay small (see explain())")
                ds /= 2
                if ds < cfg.ds_min:
                    self.stop_reason = "ds < ds_min after corrector failure"
                    break
                continue
            self.points.append(c)
            if (b.p - a.p) * (c.p - b.p) < 0:
                self.folds.append(self._locate_fold(len(self.points) - 3))
            if self.on_point is not None:
                self.on_point(self)
            if not (p_bounds[0] <= c.p <= p_bounds[1]):
                self.stop_reason = f"p = {c.p:.6g} left the bounds {p_bounds}"
                break
            ds = min(ds * cfg.adapt, cfg.ds_max) if c.iterations <= cfg.fast_iterations else ds
        else:
            self.stop_reason = f"max_points = {cfg.max_points} reached"
        return self.points

    @staticmethod
    def _vertex(pts):
        s = np.array([q.s for q in pts])
        p = np.array([q.p for q in pts])
        c = np.polyfit(s - s[1], p, 2)
        sv = -c[1] / (2 * c[0])
        return float(np.polyval(c, sv)), float(sv + s[1])

    def _locate_fold(self, i: int) -> dict:
        """Points i, i+1, i+2 bracket a turning point in p (Δp changes sign)."""
        cfg = self.cfg
        trio = self.points[i:i + 3]
        pv, sv = self._vertex(trio)
        history = [pv]
        note = ""
        # the refinement restarts from the secant (points[i-1], points[i]). For a fold in the very
        # first step (i = 0) there is no point before the bracket: the restart used the secant of
        # a point with itself (0/0) and every refinement failed without a word. Now said.
        refinements = cfg.fold_refinements if i > 0 else 0
        if i == 0 and cfg.fold_refinements:
            note = ("not refined: the fold lies in the first step (no point before the bracket); "
                    "start the continuation further from the fold")
        base_prev, base = (self.points[i - 1], trio[0]) if i > 0 else (trio[0], trio[0])
        ds = (trio[1].s - trio[0].s)
        for _ in range(refinements):
            ds /= cfg.fold_refine_factor
            seq = [base_prev, base]
            ok = False
            for _k in range(int(4 * cfg.fold_refine_factor) + 4):
                try:
                    seq.append(self._step(seq[-2], seq[-1], ds))
                except (NewtonNotConverged, FloatingPointError, np.linalg.LinAlgError):
                    break
                if len(seq) >= 4 and (seq[-2].p - seq[-3].p) * (seq[-1].p - seq[-2].p) < 0:
                    ok = True
                    break
            if not ok:
                break
            trio = seq[-3:]
            pv, sv = self._vertex(trio)
            history.append(pv)
            base_prev, base = seq[-4], seq[-3]          # restart before the new trio
        out = {"p": pv, "uncertainty": abs(history[-1] - history[-2]) if len(history) > 1 else None,
               "history": history, "residuals": [q.residual for q in trio],
               "ds_final": float(trio[1].s - trio[0].s), "note": note}
        if trio[0].observables:
            s = np.array([q.s for q in trio])
            for key in trio[0].observables:
                c = np.polyfit(s - s[1], [q.observables[key] for q in trio], 2)
                out[key] = float(np.polyval(c, sv - s[1]))
        return out

    def explain(self) -> str:
        """Points, parameter range, stop reason, settings, corrector failures and folds."""
        prange = (f"p in [{min(q.p for q in self.points):.6g}, {max(q.p for q in self.points):.6g}]"
                  if self.points else "not run yet")       # min() of no point used to raise
        lines = [f"KrylovArclength: {len(self.points)} points, {prange}, "
                 f"stop: {self.stop_reason or '-'}", self.cfg.explain(),
                 "  " + self.newton.explain(),
                 f"  corrector failures (step halved): {len(self.refused)}"]
        if self.floor_failures:
            lines.append(
                f"  ! {len(self.floor_failures)} of them stagnated within 10x of the corrector "
                f"tolerance (‖R‖/scale down to {min(self.floor_failures):.1e} against tol "
                f"{self.newton.tol:g}): the tolerance is below the attainable residual floor, so "
                "the step was halved for nothing and cannot grow back. Declare a reachable "
                "corrector tolerance (about 3x the floor) or a tighter GMRES tolerance.")
        for f in self.folds:
            u = "" if f["uncertainty"] is None else f" ± {f['uncertainty']:.1e}"
            lines.append(f"  fold at p = {f['p']:.8g}{u} (parabola vertex, final ds {f['ds_final']:.2g}, "
                         f"residuals {max(f['residuals']):.1e})"
                         + (f"; {f['note']}" if f.get("note") else ""))
        return "\n".join(lines)
