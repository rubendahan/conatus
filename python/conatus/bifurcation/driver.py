"""Generic pseudo-arclength continuation with test functions (MatCont-style), for the bifurcation core.

The curve is H(z) = 0, H : R^{N+1} -> R^N (z gathers state, extra unknowns and the active
parameter(s)). From a converged point z_k with unit tangent t_k, predictor z_k + ds t_k, Newton
corrector on the bordered system [H_z ; t_k^T] (Keller 1977, Allgower & Georg 1990), next tangent
from [H_z ; t_k^T] t = (0, 1), oriented along t_k. This is the scheme of
``conatus.continuation.PseudoArclength`` without singling out a parameter, so the same driver
follows equilibria (z = (x, p)), periodic orbits (z = (u, T, p)) and codimension-2 curves
(z = (x, p1, p2) plus the minimally augmented test functions).

**Events.** Each test function φ_k(point) is evaluated at every accepted point; a sign change
between two points is an event, located by a secant in arclength on φ_k (each iterate = one
corrected point, same hyperplane as the step), and reported with the value |φ_k| and the residual
at the located point, ``converged`` and, if False, the reason. A ``classify`` callback may relabel
or discard an event after location (e.g. a neutral saddle is not a Hopf point); discarded events
are kept in ``rejected`` with their reason, never dropped silently.

Settings: :class:`conatus.continuation.ContinuationSettings` (every step and tolerance explicit,
event budget/tolerances shown by ``explain()``). The weighted norm ‖z‖² = Σ w_i z_i² takes declared
weights (default 1, shown).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..continuation.core import ContinuationSettings
from ..errors import ConvergenceError
from .._repr import compact_repr

__all__ = ["CurvePoint", "CurveEvent", "Curve", "ArclengthDriver"]


@dataclass
class CurvePoint:
    """One accepted (or event) point of a curve: arclength, unknowns z, unit tangent, residual,
    corrector iterations, test-function values and per-curve data (eigenvalues, multipliers)."""
    __repr__ = compact_repr   # one line, arrays by shape

    s: float
    z: np.ndarray
    tangent: np.ndarray
    residual: float
    iterations: int
    tests: dict = dc_field(default_factory=dict)
    data: dict = dc_field(default_factory=dict)     # eigenvalues, multipliers, ... (per curve kind)
    status: str = "converged"

    def explain(self) -> str:
        """The point in one sentence: arclength, residual, test functions."""
        tests = ", ".join(f"{k} = {v:.3g}" for k, v in self.tests.items()
                          if isinstance(v, (int, float)))
        return (f"curve point ({self.status}) at arclength s = {self.s:.6g}: residual "
                f"{self.residual:.1e} after {self.iterations} corrector iteration(s)"
                + (f"; test functions {tests}" if tests else "")
                + (f"; data: {', '.join(self.data)}" if self.data else "")
                + f"; unknowns of {np.size(self.z)} entries in .z")


@dataclass
class CurveEvent:
    """A located sign change of a test function (``converged``/``reason`` tell how well)."""

    kind: str                         # "fold", "branch point", "hopf", "pd", "ns", "lpc", "bt", ...
    between: tuple[int, int]
    point: CurvePoint | None
    test_value: float | None
    converged: bool
    reason: str = ""
    detail: str = ""
    data: dict = dc_field(default_factory=dict)

    def explain(self, names: Mapping[str, int] | None = None) -> str:
        """One line: kind, location (``names``: label -> index in z), test value, residual."""
        loc = ""
        if self.point is not None and names:
            loc = " at " + ", ".join(f"{k} = {self.point.z[i]:.12g}" for k, i in names.items())
        conv = "" if self.converged else f" [not converged: {self.reason}]"
        val = "" if self.test_value is None else f", |test| = {abs(self.test_value):.1e}"
        res = "" if self.point is None else f", residual {self.point.residual:.1e}"
        crit = f" [located by {self.data['criterion']}]" if self.data.get("criterion") else ""
        return (f"  {self.kind}{loc}{conv} between points {self.between}{val}{res}"
                f"{(': ' + self.detail) if self.detail else ''}{crit}")


@dataclass
class Curve:
    """A continued curve: points, events, rejected events (with reason), corrector failures."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    kind: str
    names: dict                       # parameter/extra name -> index in z
    points: list = dc_field(default_factory=list)
    events: list = dc_field(default_factory=list)
    rejected: list = dc_field(default_factory=list)
    refused: list = dc_field(default_factory=list)
    stop_reason: str = ""
    settings: ContinuationSettings | None = None
    notes: list = dc_field(default_factory=list)
    context: Any = None               # the object that built the curve (system, orbit problem...)

    def column(self, name: str) -> np.ndarray:
        """Values of the named unknown (a parameter, ``T``, ``kappa``) along the curve."""
        return np.array([p.z[self.names[name]] for p in self.points])

    def events_of(self, kind: str) -> list:
        """The events of one kind (``"fold"``, ``"hopf"``, ``"pd"``, ...)."""
        return [e for e in self.events if e.kind == kind]

    def explain(self) -> str:
        """Range of every named unknown, stop reason, settings, events and rejected events."""
        cols = ", ".join(f"{k} in [{self.column(k).min():.6g}, {self.column(k).max():.6g}]"
                         for k in self.names)
        lines = [f"{self.kind} curve: {len(self.points)} points, {cols}; stop: {self.stop_reason}"]
        lines += ["  " + s for s in self.notes]
        if self.settings is not None:
            lines.append(self.settings.explain())
        lines.append(f"  corrector failures (step halved): {len(self.refused)}; "
                     f"max residual on the curve {max(p.residual for p in self.points):.1e}")
        lines += [e.explain(self.names) for e in self.events] or ["  no event detected"]
        for e, why in self.rejected:
            lines.append(f"  (rejected {e.kind} between {e.between}: {why})")
        return "\n".join(lines)


class ArclengthDriver:
    """Continuation of H(z) = 0 with test functions.

    ``H(z) -> (N,)``, ``Hz(z) -> (N, N+1)`` dense or sparse; ``tests``: name -> φ(point) (float);
    ``data(z) -> dict`` stored on each point (and passed to the tests via ``point.data``);
    ``classify(kind, event) -> (kind, None) | (None, reason)``; ``on_accept(point)`` is called after
    each accepted point (e.g. to update the phase reference of a periodic orbit).
    """

    def __init__(self, H: Callable, Hz: Callable, settings: ContinuationSettings, *,
                 tests: Mapping[str, Callable] | None = None, data: Callable | None = None,
                 classify: Callable | None = None, on_accept: Callable | None = None,
                 weights: np.ndarray | None = None, scale: Callable | None = None) -> None:
        self.H, self.Hz, self.cfg = H, Hz, settings
        self.tests = dict(tests or {})
        self.data_fn, self.classify, self.on_accept = data, classify, on_accept
        self.w = weights
        self.scale_fn = scale
        self.n_H = 0

    def explain(self) -> str:
        """Corrector, norm, residual scale and test functions."""
        tests = ", ".join(sorted(self.tests)) or "none"
        return (f"ArclengthDriver: Newton corrector on H(z) = 0 bordered by t·W(z - z_pred) = 0 "
                f"(z = unknowns and parameter)\n"
                f"  norm: {'unit weights' if self.w is None else 'declared diagonal weights'}; "
                f"residual test max|H| / {'1' if self.scale_fn is None else 'scale(z)'} <= "
                f"{self.cfg.tolerance:g}\n" + self.cfg.explain()
                + f"\n  test functions: {tests}"
                + ("; classification callback" if self.classify is not None else "")
                + f"\n  H evaluations so far: {self.n_H}")

    # --- linear algebra ----------------------------------------------------------------------
    def _w(self, m: int) -> np.ndarray:
        return np.ones(m) if self.w is None else np.asarray(self.w, float)

    def _bordered(self, z: np.ndarray, t: np.ndarray):
        A = self.Hz(z)
        row = (self._w(z.size) * t)[None, :]
        if sp.issparse(A):
            K = sp.vstack([A, sp.csr_matrix(row)], format="csc")
            lu = spla.splu(K)
            return K, lu.solve
        K = np.vstack([np.asarray(A), row])
        lu = sla.lu_factor(K)
        return K, lambda r: sla.lu_solve(lu, r)

    def norm(self, v: np.ndarray) -> float:
        """Weighted norm ‖v‖² = Σ w_i v_i² (declared weights, default 1)."""
        return float(np.sqrt(np.sum(self._w(v.size) * v * v)))

    def tangent(self, z: np.ndarray, t_prev: np.ndarray) -> np.ndarray:
        """Unit tangent at z from [H_z ; t_prev^T] t = (0, 1), oriented along t_prev."""
        _, solve = self._bordered(z, t_prev)
        rhs = np.zeros(z.size)
        rhs[-1] = 1.0
        t = solve(rhs)
        t = t / self.norm(t)
        return t if np.dot(self._w(z.size) * t_prev, t) >= 0 else -t

    def _rel(self, h: np.ndarray, z: np.ndarray) -> float:
        sc = 1.0 if self.scale_fn is None else max(float(self.scale_fn(z)), 1e-300)
        return float(np.max(np.abs(h))) / sc if h.size else 0.0

    def correct(self, z_pred: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float, int]:
        """Newton on H(z) = 0, t·W(z - z_pred) = 0; raises ConvergenceError on failure."""
        z = z_pred.copy()
        w = self._w(z.size)
        rel = nrm = np.inf
        for k in range(self.cfg.max_newton + 1):
            h = np.asarray(self.H(z))
            self.n_H += 1
            N = float(np.sum(w * t * (z - z_pred)))
            rel, nrm = self._rel(h, z), abs(N)
            if rel <= self.cfg.tolerance and nrm <= self.cfg.tolerance:
                return z, rel, k
            if k == self.cfg.max_newton or not np.all(np.isfinite(h)):
                break
            _, solve = self._bordered(z, t)
            dz = solve(-np.concatenate([h, [N]]))
            if not np.all(np.isfinite(dz)):
                break
            z = z + dz
        raise ConvergenceError(f"corrector not converged in {self.cfg.max_newton} its "
                               f"(|H|/scale = {rel:.2e}, |N| = {nrm:.2e})")

    def _point(self, s, z, t, rel, its, status="converged") -> CurvePoint:
        pt = CurvePoint(s, z, t, rel, its, status=status)
        if self.data_fn is not None:
            pt.data = dict(self.data_fn(z))
        pt.tests = {k: float(f(pt)) for k, f in self.tests.items()}
        return pt

    # --- driver ------------------------------------------------------------------------------
    def run(self, z0: np.ndarray, t0: np.ndarray, *, kind: str, names: Mapping[str, int],
            bounds: Mapping[str, tuple[float, float]] | None = None,
            stop: Callable[[CurvePoint], str | None] | None = None) -> Curve:
        """Continue from a converged z0 along the initial direction t0 (re-oriented tangent).

        The start point is accepted when max|H| / scale <= 10 x the corrector tolerance (it is
        recorded with its own residual; a Newton-converged start passes)."""
        cfg = self.cfg
        z0 = np.asarray(z0, float)
        r0 = self._rel(np.asarray(self.H(z0)), z0)
        if r0 > 10 * cfg.tolerance:
            raise ConvergenceError(f"starting point is not converged: |H|/scale = {r0:.2e}",
                                   fix="converge the starting point first (Newton), then continue")
        t = self.tangent(z0, np.asarray(t0, float) / self.norm(np.asarray(t0, float)))
        cv = Curve(kind=kind, names=dict(names), settings=cfg)
        cv.points.append(self._point(0.0, z0, t, r0, 0, "start"))
        if self.on_accept is not None:
            self.on_accept(cv.points[-1])
        ds = abs(cfg.ds)
        while len(cv.points) <= cfg.max_points:
            last = cv.points[-1]
            try:
                z, rel, its = self.correct(last.z + ds * last.tangent, last.tangent)
                tn = self.tangent(z, last.tangent)
                pt = self._point(last.s + ds, z, tn, rel, its)
            except (ConvergenceError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as e:
                cv.refused.append((ds, str(e).splitlines()[0]))
                ds /= 2
                if ds < cfg.ds_min:
                    cv.stop_reason = f"ds < ds_min after corrector failure ({str(e).splitlines()[0]})"
                    break
                continue
            cv.points.append(pt)
            self._events(cv, len(cv.points) - 2)
            if self.on_accept is not None:
                self.on_accept(pt)
            if bounds:
                out = [k for k, (lo, hi) in bounds.items() if not lo <= z[names[k]] <= hi]
                if out:
                    cv.stop_reason = f"{out[0]} = {z[names[out[0]]]:.6g} left {bounds[out[0]]}"
                    break
            if stop is not None:
                why = stop(pt)
                if why:
                    cv.stop_reason = why
                    break
            ds = min(ds * cfg.adapt, cfg.ds_max) if its <= cfg.fast_iterations else ds
        if not cv.stop_reason:
            cv.stop_reason = f"max_points = {cfg.max_points} reached"
        return cv

    def _events(self, cv: Curve, i: int) -> None:
        a, b = cv.points[i], cv.points[i + 1]
        tp_change = bool(a.tangent[-1] * b.tangent[-1] < 0)
        for name in self.tests:
            fa, fb = a.tests[name], b.tests[name]
            if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb >= 0:
                continue
            ev = self.locate(name, a, b, (i, i + 1))
            if tp_change and name not in ("fold", "lpc"):
                ev.data["tp_sign_change"] = True
            if self.classify is not None:
                kind, why = self.classify(name, ev)
                if kind is None:
                    cv.rejected.append((ev, why))
                    continue
                ev.kind = kind
            cv.events.append(ev)

    def locate(self, name: str, a: CurvePoint, b: CurvePoint, between) -> CurveEvent:
        """Secant in arclength on φ_name from a (each iterate = one corrected point)."""
        cfg = self.cfg
        f = self.tests[name]
        s0, s1, f0, f1 = 0.0, b.s - a.s, a.tests[name], b.tests[name]
        best, why = b, f"secant budget event_max_iterations = {cfg.event_max_iterations} exhausted"
        tol = cfg.fold_tol if name == "fold" else cfg.crossing_tol
        conv = abs(f1) <= tol
        for k in range(cfg.event_max_iterations):
            if conv:
                break
            if f1 == f0:
                why = f"secant stalled at iterate {k}"
                break
            s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
            if not (min(0.0, b.s - a.s) - abs(b.s - a.s) <= s2 <= 2 * abs(b.s - a.s)):
                s2 = 0.5 * (s0 + s1)          # keep the iterate near the bracket
            try:
                z, rel, _ = self.correct(a.z + s2 * a.tangent, a.tangent)
                # Exactly at a branch point the bordered matrix is singular: LAPACK then warns
                # (LinAlgWarning, an exactly zero pivot) or returns non-finite values. Both are
                # the expected outcome here, caught as a signal and never printed to the user.
                singular = False
                with np.errstate(all="ignore"), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", sla.LinAlgWarning)
                    try:
                        tz = self.tangent(z, a.tangent)
                    except RuntimeError as err:   # sparse LU (splu): "Factor is exactly singular"
                        if "singular" not in str(err).lower():
                            raise
                        singular, tz = True, None
                for w in caught:
                    if issubclass(w.category, sla.LinAlgWarning):
                        singular = True
                    else:                     # any other warning is passed on unchanged
                        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
                status = "event"
                if singular or not np.all(np.isfinite(tz)):
                    # singular bordered matrix: chord direction of the bracket instead, declared
                    # in the point status (it was not)
                    tz = (b.z - a.z) / self.norm(b.z - a.z)
                    status = "event; tangent = chord of the bracket (singular bordered matrix)"
                pt = self._point(a.s + s2, z, tz, rel, 0, status)
            except (ConvergenceError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as e:
                why = f"corrector failed at secant iterate {k} ({str(e).splitlines()[0]})"
                break
            f2 = pt.tests[name] if name in pt.tests else float(f(pt))
            best = pt
            s0, f0, s1, f1 = s1, f1, s2, f2
            if abs(f2) <= tol:
                conv = True
        ev = CurveEvent(name, between, best, best.tests.get(name), bool(conv), "" if conv else why)
        # the declared threshold that located it, in the result
        ev.data["criterion"] = (f"|test {name}| <= {'fold_tol' if name == 'fold' else 'crossing_tol'}"
                                f" = {tol:g}, secant budget {cfg.event_max_iterations}")
        return ev
