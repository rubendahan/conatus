"""Pseudo-arclength continuation of G(x, p) = 0.

Unknowns z = (x, p). From a converged point z_k with unit tangent t_k (in the declared weighted
norm), the predictor is z_pred = z_k + ds t_k and the corrector is Newton on the bordered system

    [ J      G_p ] [dx]     [ G(x, p)                         ]
    [ t_x^T  t_p ] [dp] = - [ t_x·(x - x_pred) + t_p (p - p_pred) ]

(Keller 1977). The next tangent solves the same bordered matrix with right-hand side (0, 1) and is
oriented along the previous one. Events between consecutive points:

* **fold**: sign change of the parameter component t_p of the tangent (limit point in p); the
  fold is then located by a secant on t_p(s) (each evaluation = one corrected point), reported
  with its own residual;
* **eigenvalue crossings** (optional ``eigen`` callback): change in the number of eigenvalues with
  positive real part; a crossing by a complex pair is reported as Hopf, by a real eigenvalue as a
  real bifurcation/fold (a fold is also a real crossing, reported under both names). Crossings are
  located by a secant in arclength on the real part of the threshold eigenvalue (the one that
  changes sign; eigenvalues set exactly to 0 by the caller as marginal are excluded).

Every setting is required and shown by ``explain()`` (no hidden defaults): initial and bounded
arclength step, adaptation factor and the Newton-iteration threshold that triggers it, corrector
tolerance and budget, FD step for G_p when no analytic G_p is given, and the norm weights. The
event-location settings (secant budget, fold / crossing / Hopf tolerances) have defaults, shown by
``explain()`` too; each located event carries ``converged`` and, if False, the reason.
A point whose corrector fails is recorded with status ``refused`` and the step is halved; below
``ds_min`` the run stops with the reason recorded.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._deprecation import positional_order
from ..errors import DtypeError, InputTypeError, OptionError
# round-off threshold of a complex starting state (declared; shared with FlowProperties),
# re-exported here under its former name
from ..tolerances import IMAG_ROUNDOFF
from .._repr import compact_repr

__all__ = ["ContinuationSettings", "BranchPoint", "Branch", "Event", "PseudoArclength"]


def _count(value, name: str, where: str) -> None:
    """An iteration count or budget is an integer >= 1 (a float used to pass the check and fail
    later in ``range``)."""
    import numbers
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise OptionError(f"{where}: {name}={value!r} must be an integer >= 1",
                          fix=f"e.g. {name}=10 (counts must be integers)")


def real_state(x, where: str) -> np.ndarray:
    """The continued state is real: a complex vector is accepted only when its imaginary part is
    round-off (below :data:`IMAG_ROUNDOFF` x max(1, max|x|), then dropped), else refused.
    ``np.asarray(x, float)`` used to drop any imaginary part with a mere ComplexWarning."""
    x = np.asarray(x)
    if np.iscomplexobj(x):
        im = float(np.max(np.abs(x.imag))) if x.size else 0.0
        scale = max(1.0, float(np.max(np.abs(x)))) if x.size else 1.0
        if im > IMAG_ROUNDOFF * scale:
            raise DtypeError(f"{where}: the starting state is complex (max|Im x| = {im:.2e} > "
                             f"{IMAG_ROUNDOFF:.1e} x max(1, max|x|), the declared round-off)",
                             why="pseudo-arclength continuation here works on real unknowns",
                             fix="continue the real and imaginary parts as separate real unknowns, "
                                 "or pass the real state (x.real) if the imaginary part is spurious")
        x = x.real
    return np.asarray(x, dtype=float)


#: Central finite-difference step of G_p when ``dp_fd="auto"``: h = FD_STEP_REL x
#: max(1, |p|), FD_STEP_REL = eps^(1/3), the step that balances truncation O(h^2) and rounding
#: O(eps/h) of a central difference (Nocedal & Wright, Numerical Optimization, §8.1).
FD_STEP_REL = float(np.finfo(float).eps ** (1 / 3))


def fd_step(dp_fd: Any, p: float) -> float:
    """The central FD step in p: ``dp_fd`` if a number, the declared rule if ``"auto"``."""
    if dp_fd == "auto":
        return FD_STEP_REL * max(1.0, abs(float(p)))
    return float(dp_fd)


def fd_text(dp_fd: Any) -> str:
    """explain() words for the G_p evaluation."""
    if dp_fd is None:
        return "analytic G_p (dp_fd=None)"
    if dp_fd == "auto":
        return (f"central FD, dp = eps^(1/3) max(1, |p|) = {FD_STEP_REL:.2e} max(1, |p|) "
                "(default dp_fd='auto'; an analytic G_p is used when given)")
    return f"central FD, dp = {dp_fd:g} (given)"


@dataclass(frozen=True, kw_only=True)
class ContinuationSettings:
    """Steps, tolerance and budgets of a pseudo-arclength run; ``explain()`` prints them.

    Rule: the accuracy target ``tolerance`` and the problem-scale settings (``ds``,
    ``ds_min``, ``ds_max``, ``max_points``: the parameter scale and the extent of the branch)
    are required; the algorithm controls (``adapt``, ``fast_iterations``, ``max_newton``,
    ``dp_fd``) and the event-location settings have declared defaults (``CONTINUATION_DEFAULTS``),
    marked "(default)" by ``explain()``."""

    ds: float                 # initial arclength step
    ds_min: float             # smallest step before the run stops
    ds_max: float             # largest step
    tolerance: float          # corrector tolerance on max|G| / scale and on |N|
    max_points: int           # number of continuation steps
    adapt: float = 1.3        # step factor (> 1): ds *= adapt if Newton is fast, /= 2 on failure
    fast_iterations: int = 3  # corrector iterations <= this -> "fast" (grow the step)
    max_newton: int = 12      # corrector budget (a failure halves the step, never accepted)
    # central FD step for G_p: "auto" (the rule fd_step), a number, or None (an analytic G_p
    # must be given)
    dp_fd: float | str | None = "auto"
    # event location (were literals). Defaults = the former values, all shown by
    # explain(); an event whose secant stops outside its tolerance is flagged converged=False.
    event_max_iterations: int = 30   # secant budget per event (each iterate = one corrected point)
    fold_tol: float = 1e-12          # fold located when |t_p| <= fold_tol (t is a unit vector)
    crossing_tol: float = 1e-11      # crossing located when |Re λ| <= crossing_tol · max(1, |λ|)
    hopf_imag_tol: float = 1e-8      # Hopf if both near-threshold |Im λ| > hopf_imag_tol · max(1, |λ|)

    def __post_init__(self) -> None:
        # OptionError is a ValueError: the historical ``except ValueError`` keeps working
        where = "ContinuationSettings"
        if not (0 < self.ds_min <= abs(self.ds) <= self.ds_max):
            raise OptionError(f"{where}: need 0 < ds_min <= |ds| <= ds_max (ds = {self.ds!r}, "
                              f"ds_min = {self.ds_min!r}, ds_max = {self.ds_max!r})",
                              fix="e.g. ds=0.1, ds_min=1e-6, ds_max=0.5")
        if not self.adapt > 1:
            raise OptionError(f"{where}: adapt={self.adapt!r} must be > 1 (step growth factor)",
                              fix="e.g. adapt=1.3")
        for name in ("fast_iterations", "max_newton", "max_points", "event_max_iterations"):
            _count(getattr(self, name), name, where)
        if not self.tolerance > 0:
            raise OptionError(f"{where}: tolerance={self.tolerance!r} must be > 0",
                              fix="e.g. tolerance=1e-10 (relative corrector residual)")
        if not min(self.fold_tol, self.crossing_tol, self.hopf_imag_tol) > 0:
            raise OptionError(f"{where}: fold_tol, crossing_tol and hopf_imag_tol must be > 0",
                              fix="leave them at their declared defaults unless needed")
        if self.dp_fd is not None and self.dp_fd != "auto" and not (
                isinstance(self.dp_fd, (int, float)) and self.dp_fd > 0):
            raise OptionError(f"{where}: dp_fd={self.dp_fd!r} must be > 0, 'auto' (the declared "
                              "rule eps^(1/3) max(1, |p|)) or None (analytic G_p)",
                              fix="leave the default dp_fd='auto', or e.g. dp_fd=1e-6")

    # The vocabulary of the rest of the API: an accuracy is ``tol``, a
    # budget ``max_iterations``. The fields keep their v0.1 names (scripts read cfg.tolerance and
    # cfg.max_newton, and dataclasses.replace() takes field names); these read the same values.
    @property
    def tol(self) -> float:
        """The corrector tolerance :attr:`tolerance` under the API-wide name (read-only)."""
        return self.tolerance

    @property
    def max_iterations(self) -> int:
        """The corrector budget :attr:`max_newton` under the API-wide name (read-only)."""
        return self.max_newton

    def _d(self, name: str) -> str:
        """" (default)" when the field holds its declared default."""
        return " (default)" if getattr(self, name) == CONTINUATION_DEFAULTS[name] else ""

    def explain(self) -> str:
        """Every setting, one line per group; defaults marked."""
        d = self._d
        return (f"  ds = {self.ds:g} in [{self.ds_min:g}, {self.ds_max:g}], x{self.adapt:g}"
                f"{d('adapt')} when the corrector takes <= {self.fast_iterations}"
                f"{d('fast_iterations')} its, /2 on failure\n"
                f"  corrector: tolerance {self.tolerance:g}, budget {self.max_newton}"
                f"{d('max_newton')} its; G_p: {fd_text(self.dp_fd)}\n"
                f"  max points {self.max_points}\n"
                f"  events: secant budget event_max_iterations = {self.event_max_iterations}"
                f"{d('event_max_iterations')}, fold_tol = {self.fold_tol:g}{d('fold_tol')} "
                f"(|t_p|), crossing_tol = {self.crossing_tol:g}{d('crossing_tol')} "
                f"(|Re λ| / max(1, |λ|)), hopf_imag_tol = {self.hopf_imag_tol:g}"
                f"{d('hopf_imag_tol')} (|Im λ| / max(1, |λ|))")


# v0.1 positional order, accepted for one cycle with a DeprecationWarning (conatus._deprecation)
positional_order(ContinuationSettings, (
    "ds", "ds_min", "ds_max", "adapt", "fast_iterations", "tolerance", "max_newton", "max_points",
    "dp_fd", "event_max_iterations", "fold_tol", "crossing_tol", "hopf_imag_tol"))

#: Declared defaults of ContinuationSettings (algorithm controls and event location).
CONTINUATION_DEFAULTS = dict(adapt=1.3, fast_iterations=3, max_newton=12, dp_fd="auto",
                             event_max_iterations=30, fold_tol=1e-12, crossing_tol=1e-11,
                             hopf_imag_tol=1e-8)


@dataclass
class BranchPoint:
    """A point of a branch: arclength, state, parameter, unit tangent, residual, status."""
    __repr__ = compact_repr   # one line, arrays by shape

    s: float                  # accumulated arclength
    x: np.ndarray
    p: float
    tangent: np.ndarray       # unit tangent (x-part, p-part last), weighted norm
    residual: float           # max|G(x, p)| / scale at convergence
    iterations: int
    status: str               # "start" | "converged"
    unstable: int | None = None   # eigenvalues with Re > 0 (if an eigen callback is given)

    @property
    def tp(self) -> float:
        """Parameter component of the unit tangent (changes sign at a fold)."""
        return float(self.tangent[-1])

    def explain(self) -> str:
        """The point in one sentence: where, how well solved, how stable."""
        stab = "" if self.unstable is None else f", {self.unstable} unstable eigenvalue(s)"
        return (f"branch point ({self.status}) at arclength s = {self.s:.6g}: p = {self.p:.10g}, "
                f"residual {self.residual:.1e} after {self.iterations} corrector iteration(s), "
                f"tangent p-component {self.tp:+.3g}{stab}; state of {np.size(self.x)} entries "
                "in .x")


@dataclass
class Event:
    """A detected event between two branch points, with its located parameter value."""

    kind: str                 # "fold" | "hopf" | "real crossing"
    between: tuple[int, int]  # indices of the bracketing branch points
    p: float | None = None    # located parameter value (last secant iterate if not converged)
    detail: str = ""
    converged: bool | None = None  # secant reached its declared tolerance (None: not attempted)
    reason: str = ""          # why the location did not converge
    criterion: str = ""       # the declared threshold that located it (in the result)

    def explain(self) -> str:
        """One line: kind, location, convergence of the location, detail, criterion."""
        loc = "" if self.p is None else f" at p = {self.p:.12g}"
        conv = "" if self.converged is not False else f" [not converged: {self.reason}]"
        crit = f" [located by {self.criterion}]" if self.criterion else ""
        return (f"  {self.kind}{loc}{conv} between points {self.between}"
                f"{(': ' + self.detail) if self.detail else ''}{crit}")


@dataclass
class Branch:
    """Result of :meth:`PseudoArclength.run`: points, events, refused steps, stop reason."""
    __repr__ = compact_repr   # one line, arrays by shape (as)

    points: list[BranchPoint] = dc_field(default_factory=list)
    events: list[Event] = dc_field(default_factory=list)
    refused: list[tuple[float, str]] = dc_field(default_factory=list)   # (ds, reason)
    stop_reason: str = ""
    settings: ContinuationSettings | None = None

    @property
    def p(self) -> np.ndarray:
        """Parameter value of every point."""
        return np.array([q.p for q in self.points])

    def folds(self) -> list[Event]:
        """The fold events."""
        return [e for e in self.events if e.kind == "fold"]

    def explain(self) -> str:
        """Points, parameter range, stop reason, settings and events."""
        lines = [f"Pseudo-arclength branch: {len(self.points)} points, "
                 f"p in [{self.p.min():.6g}, {self.p.max():.6g}], stop: {self.stop_reason}"]
        if self.settings is not None:
            lines.append(self.settings.explain())
        lines.append(f"  corrector failures (step halved): {len(self.refused)}")
        lines += [e.explain() for e in self.events] or ["  no event detected"]
        return "\n".join(lines)


class PseudoArclength:
    """Continuation of G(x, p) = 0 with a bordered Newton corrector.

    ``residual(x, p) -> G`` and ``jacobian(x, p) -> dG/dx`` (dense or sparse); ``dGdp`` optional
    (else central FD with the declared step); ``eigen(x, p)`` optional, returns eigenvalues of the
    linearized problem (stability counted as #Re > 0); ``weights`` = diagonal of the norm on x
    (the parameter has weight 1); ``scale(x, p)`` optional residual scale for the relative test.
    """

    def __init__(self, residual: Callable, jacobian: Callable, settings: ContinuationSettings, *,
                 dGdp: Callable | None = None, eigen: Callable | None = None,
                 marginal_tol: float | None = None,
                 weights: np.ndarray | None = None, scale: Callable | None = None) -> None:
        if eigen is not None and (marginal_tol is None or not marginal_tol > 0):
            raise InputTypeError("PseudoArclength: an eigen callback requires an explicit "
                                 "marginal_tol > 0 (eigenvalues with |Re λ| <= marginal_tol are "
                                 "marginal, not counted)",
                                 fix="e.g. marginal_tol=1e-10 (declared, shown by explain())")
        self.marginal_tol = marginal_tol
        if dGdp is None and settings.dp_fd is None:
            raise OptionError("PseudoArclength: dp_fd=None declares an analytic G_p, but no "
                              "dGdp was given", fix="pass dGdp=, or leave dp_fd at its default "
                              "'auto' (central FD, declared step rule)")
        self.G, self.J, self.dGdp_an = residual, jacobian, dGdp
        self.eigen, self.w, self.scale_fn = eigen, weights, scale
        self.cfg = settings
        self.n_residuals = 0

    # --- pieces -----------------------------------------------------------------------------
    def _Gp(self, x: np.ndarray, p: float) -> np.ndarray:
        if self.dGdp_an is not None:
            return np.asarray(self.dGdp_an(x, p))
        h = fd_step(self.cfg.dp_fd, p)
        return (np.asarray(self.G(x, p + h)) - np.asarray(self.G(x, p - h))) / (2 * h)

    def _w(self, n: int) -> np.ndarray:
        return np.ones(n) if self.w is None else np.asarray(self.w, dtype=float)

    def _bordered(self, x: np.ndarray, p: float, t: np.ndarray):
        J = self.J(x, p)
        Gp = self._Gp(x, p)[:, None]
        w = self._w(x.size)
        row = np.concatenate([w * t[:-1], [t[-1]]])[None, :]
        if sp.issparse(J):
            A = sp.bmat([[J, sp.csr_matrix(Gp)],
                         [sp.csr_matrix(row[:, :-1]), sp.csr_matrix(row[:, -1:])]], format="csc")
            return A, lambda r: spla.spsolve(A, r)
        A = np.block([[np.asarray(J), Gp], [row]])
        lu = sla.lu_factor(A)
        return A, lambda r: sla.lu_solve(lu, r)

    def _norm(self, v: np.ndarray) -> float:
        w = self._w(v.size - 1)
        return float(np.sqrt(np.sum(w * np.abs(v[:-1]) ** 2) + abs(v[-1]) ** 2))

    def _rel(self, g: np.ndarray, x: np.ndarray, p: float) -> float:
        sc = 1.0 if self.scale_fn is None else max(float(self.scale_fn(x, p)), 1e-300)
        return float(np.max(np.abs(g))) / sc

    def tangent(self, x: np.ndarray, p: float, t_prev: np.ndarray) -> np.ndarray:
        """Unit tangent at (x, p) from the bordered system, oriented along ``t_prev``."""
        _, solve = self._bordered(x, p, t_prev)
        rhs = np.zeros(x.size + 1)
        rhs[-1] = 1.0
        t = np.real_if_close(solve(rhs))
        t = t / self._norm(t)
        return t if np.dot(np.concatenate([self._w(x.size) * t_prev[:-1], [t_prev[-1]]]), t) >= 0 else -t

    def correct(self, z_pred: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float, int]:
        """Bordered Newton from z_pred on the hyperplane orthogonal to t; raises on failure."""
        z = z_pred.copy()
        w = self._w(z.size - 1)
        for k in range(self.cfg.max_newton + 1):
            x, p = z[:-1], float(z[-1])
            g = np.asarray(self.G(x, p))
            self.n_residuals += 1
            N = float(np.sum(w * t[:-1] * (x - z_pred[:-1])) + t[-1] * (p - z_pred[-1]))
            rel = self._rel(g, x, p)
            if rel <= self.cfg.tolerance and abs(N) <= self.cfg.tolerance:
                return z, rel, k
            if k == self.cfg.max_newton:
                break
            _, solve = self._bordered(x, p, t)
            dz = solve(-np.concatenate([g, [N]]))
            if not np.all(np.isfinite(dz)):
                raise FloatingPointError("non-finite corrector step")
            z = z + np.real_if_close(dz)
        raise RuntimeError(f"corrector not converged in {self.cfg.max_newton} its (|G|/scale = "
                           f"{rel:.2e}, |N| = {abs(N):.2e})")

    # --- driver ---------------------------------------------------------------------------------
    def run(self, x0: np.ndarray, p0: float, direction: float = +1.0,
            p_bounds: tuple[float, float] | None = None) -> Branch:
        """Continue from a converged point (x0, p0); the first tangent points along ``direction``
        in p. Stops after ``max_points`` steps, below ``ds_min`` or outside ``p_bounds``.

        The start is accepted when max|G| / scale <= 10 x the corrector tolerance (recorded with
        its own residual). A complex x0 is refused unless its imaginary part is round-off (``IMAG_ROUNDOFF``)."""
        cfg = self.cfg
        x0 = real_state(x0, "PseudoArclength.run")
        g0 = np.asarray(self.G(x0, p0))
        r0 = self._rel(g0, x0, p0)
        if r0 > 10 * cfg.tolerance:
            # an input refusal (OptionError, a ValueError as before), not a corrector failure
            raise OptionError(f"starting point is not converged: |G|/scale = {r0:.2e}",
                              fix="converge the starting point first (Newton), then continue")
        t = np.zeros(x0.size + 1)
        t[-1] = np.sign(direction) or 1.0
        t = self.tangent(x0, p0, t)
        br = Branch(settings=cfg)
        br.points.append(BranchPoint(0.0, x0, float(p0), t, r0, 0, "start",
                                     self._unstable(x0, p0)))
        ds = abs(cfg.ds)
        while len(br.points) <= cfg.max_points:
            last = br.points[-1]
            z_last = np.concatenate([last.x, [last.p]])
            try:
                z, rel, its = self.correct(z_last + ds * last.tangent, last.tangent)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
                br.refused.append((ds, str(e)))
                ds /= 2
                if ds < cfg.ds_min:
                    br.stop_reason = f"ds < ds_min after corrector failure ({e})"
                    break
                continue
            x, p = z[:-1], float(z[-1])
            tn = self.tangent(x, p, last.tangent)
            pt = BranchPoint(last.s + ds, x, p, tn, rel, its, "converged", self._unstable(x, p))
            br.points.append(pt)
            self._events(br, len(br.points) - 2)
            if p_bounds is not None and not (p_bounds[0] <= p <= p_bounds[1]):
                br.stop_reason = f"p = {p:.6g} left the bounds {p_bounds}"
                break
            ds = min(ds * cfg.adapt, cfg.ds_max) if its <= cfg.fast_iterations else ds
        else:
            br.stop_reason = f"max_points = {cfg.max_points} reached"
        return br

    def _unstable(self, x: np.ndarray, p: float) -> int | None:
        """Number of eigenvalues with Re λ > marginal_tol (marginal ones are not counted)."""
        if self.eigen is None:
            return None
        ev = np.asarray(self.eigen(x, p))
        return int(np.sum(ev.real > self.marginal_tol))

    def explain(self) -> str:
        """Corrector, norm, residual scale, G_p and stability test."""
        gp = ("analytic dG/dp" if self.dGdp_an is not None
              else f"dG/dp: {fd_text(self.cfg.dp_fd)}")
        return (f"PseudoArclength: bordered Newton corrector on G(x, p) = 0 plus the arclength "
                f"condition; {gp}\n"
                f"  norm: {'unit weights' if self.w is None else 'declared diagonal weights'} on x, "
                f"weight 1 on p; residual test max|G| / "
                f"{'1' if self.scale_fn is None else 'scale(x, p)'} <= {self.cfg.tolerance:g}\n"
                + self.cfg.explain() + f"\n  {self.explain_eigen()}\n"
                f"  residual evaluations so far: {self.n_residuals}")

    def explain_eigen(self) -> str:
        """The stability count used (or "no stability callback")."""
        return ("no stability callback" if self.eigen is None else
                f"stability: #Re λ > {self.marginal_tol:g}; |Re λ| <= {self.marginal_tol:g} marginal")

    def _events(self, br: Branch, i: int) -> None:
        a, b = br.points[i], br.points[i + 1]
        if a.tp * b.tp < 0:
            p_f, detail, conv, why = self._locate_fold(a, b)
            br.events.append(Event("fold", (i, i + 1), p_f, detail, conv, why,
                                   f"|t_p| <= fold_tol = {self.cfg.fold_tol:g}"
                                   f"{self.cfg._d('fold_tol')}"))
        if a.unstable is not None and b.unstable is not None and a.unstable != b.unstable:
            d = b.unstable - a.unstable
            kind = "hopf" if abs(d) == 2 and self._complex_crossing(a, b) else "real crossing"
            p_c, loc, conv, why = self._locate_crossing(a, b)
            br.events.append(Event(kind, (i, i + 1), p_c,
                                   f"#Re>0: {a.unstable} -> {b.unstable}; {loc}", conv, why,
                                   f"|Re λ| <= crossing_tol = {self.cfg.crossing_tol:g}"
                                   f"{self.cfg._d('crossing_tol')} x max(|λ|, 1e-3 median |λ|); Hopf if "
                                   f"|Im λ| > hopf_imag_tol = {self.cfg.hopf_imag_tol:g}"
                                   f"{self.cfg._d('hopf_imag_tol')} x max(|λ|, 1e-3 median |λ|)"))

    def _complex_crossing(self, a: BranchPoint, b: BranchPoint) -> bool:
        from .._scales import spectral_floor
        ev = np.asarray(self.eigen(b.x, b.p))
        near = ev[np.argsort(np.abs(ev.real))[:2]]
        return bool(np.all(np.abs(near.imag) > self.cfg.hopf_imag_tol
                           * max(np.max(np.abs(near)), spectral_floor(ev))))

    def _n_marginal(self, x: np.ndarray, p: float) -> int:
        """Number of eigenvalues with |Re λ| <= marginal_tol (persistent marginal modes)."""
        ev = np.asarray(self.eigen(x, p), dtype=complex)
        return int(np.count_nonzero(np.abs(ev.real) <= self.marginal_tol))

    def _threshold_re(self, x: np.ndarray, p: float, m: int, n_marginal: int) -> tuple[float, complex]:
        """Real part of the eigenvalue sitting at the stability threshold between two points: the
        (m+1)-th in decreasing real part, marginal modes removed.

        ``n_marginal`` is the number of marginal eigenvalues at the bracketing points (persistent
        ones, e.g. a translation mode). A secant iterate close to the crossing puts the threshold
        eigenvalue itself within marginal_tol: there are then more marginal eigenvalues than
        ``n_marginal`` and that eigenvalue is returned. (Before, the next eigenvalue kept[m] was
        returned instead, far from 0: a secant iterate landing within marginal_tol of the crossing
        (at the first step when Re λ is linear in s; likely otherwise since crossing_tol <
        marginal_tol) read a wrong value and the secant exhausted its budget;
        tests/exact/test_exact_continuation.py.)"""
        ev = np.asarray(self.eigen(x, p), dtype=complex)
        from .._scales import spectral_floor
        self._spectral_floor = spectral_floor(ev)       # the crossing test's floor (unit of λ)
        marg = np.abs(ev.real) <= self.marginal_tol
        if np.count_nonzero(marg) > n_marginal:
            # persistent marginal modes sit at round-off; the threshold one is the largest |Re λ|
            cand = ev[marg]
            lam = cand[np.argmax(np.abs(cand.real))]
            return float(lam.real), complex(lam)
        kept = ev[~marg]                                        # marginal modes excluded
        kept = kept[np.argsort(-kept.real)]
        if m < kept.size:
            return float(kept[m].real), complex(kept[m])
        # the threshold eigenvalue itself is within marginal_tol: we are at the crossing
        near = ev[np.argmin(np.abs(ev.real))] if ev.size else 0j
        return 0.0, complex(near)

    def _locate_crossing(self, a: BranchPoint, b: BranchPoint
                         ) -> tuple[float | None, str, bool, str]:
        """Secant in arclength on Re λ_threshold(s) = 0 (each trial = one corrected point).
        Returns (p, detail, converged, reason); converged iff |Re λ| <= crossing_tol · max(1, |λ|)."""
        cfg = self.cfg
        m = min(a.unstable, b.unstable)
        za = np.concatenate([a.x, [a.p]])
        s0, s1 = 0.0, b.s - a.s
        n_marg = min(self._n_marginal(a.x, a.p), self._n_marginal(b.x, b.p))
        f0, f1 = (self._threshold_re(a.x, a.p, m, n_marg)[0],
                  self._threshold_re(b.x, b.p, m, n_marg)[0])
        if not (np.isfinite(f0) and np.isfinite(f1)) or f0 * f1 > 0:
            return None, "not located (threshold eigenvalue not bracketed)", False, "not bracketed"
        best, why = (None, np.inf, 0j), f"secant budget event_max_iterations = {cfg.event_max_iterations} exhausted"
        conv = False
        for k in range(cfg.event_max_iterations):
            s2 = s1 - f1 * (s1 - s0) / (f1 - f0) if f1 != f0 else 0.5 * (s0 + s1)
            try:
                z, rel, _ = self.correct(za + s2 * a.tangent, a.tangent)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
                why = f"corrector failed at secant iterate {k} ({e})"
                break
            f2, lam = self._threshold_re(z[:-1], float(z[-1]), m, n_marg)
            if not np.isfinite(f2):
                why = f"non-finite threshold eigenvalue at secant iterate {k}"
                break
            best = (float(z[-1]), abs(f2), lam)
            s0, f0, s1, f1 = s1, f1, s2, f2
            if abs(f2) <= cfg.crossing_tol * max(abs(lam), self._spectral_floor):
                conv, why = True, ""
                break
        if best[0] is None:
            return None, f"not located ({why})", False, why
        return (best[0], f"|Re λ| = {best[1]:.1e} at the located point, Im λ = {best[2].imag:.6g}",
                conv, why)

    def _locate_fold(self, a: BranchPoint, b: BranchPoint) -> tuple[float, str, bool, str]:
        """Secant in arclength on t_p(s) = 0, from a (each trial = one corrected point).
        Returns (p, detail, converged, reason); converged iff |t_p| <= fold_tol. Without any
        successful iterate the bracketing point b is reported, with converged = False."""
        cfg = self.cfg
        za = np.concatenate([a.x, [a.p]])
        s0, s1, f0, f1 = 0.0, b.s - a.s, a.tp, b.tp
        best = (b.p, abs(f1), b.residual)
        conv, why = abs(f1) <= cfg.fold_tol, ""
        if not conv:
            why = f"secant budget event_max_iterations = {cfg.event_max_iterations} exhausted"
        for k in range(cfg.event_max_iterations):
            if conv:
                break
            if f1 == f0:
                why = f"secant stalled (equal t_p values) at iterate {k}"
                break
            s2 = s1 - f1 * (s1 - s0) / (f1 - f0)
            try:
                z, rel, _ = self.correct(za + s2 * a.tangent, a.tangent)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as e:
                why = f"corrector failed at secant iterate {k} ({e})"
                break
            t2 = self.tangent(z[:-1], float(z[-1]), a.tangent)
            best = (float(z[-1]), abs(t2[-1]), rel)
            s0, f0, s1, f1 = s1, f1, s2, float(t2[-1])
            if abs(f1) <= cfg.fold_tol:
                conv, why = True, ""
        return (best[0], f"|t_p| = {best[1]:.1e} at the located point, |G|/scale = {best[2]:.1e}",
                bool(conv), why)
