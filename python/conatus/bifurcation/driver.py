"""Generic pseudo-arclength continuation with test functions (MatCont-style), for the bifurcation core.

The curve is H(z) = 0, H : R^{N+1} -> R^N (z gathers state, extra unknowns and the active
parameter(s)). From a converged point z_k with unit tangent t_k, predictor z_k + ds t_k, Newton
corrector on the bordered system [H_z ; t_k^T] (Keller 1977, Allgower & Georg 1990), next tangent
from [H_z ; t_k^T] t = (0, 1), oriented along t_k. This is the scheme of
``conatus.continuation.PseudoArclength`` without singling out a parameter, so the same driver
follows equilibria (z = (x, p)), periodic orbits (z = (u, T, p)) and codimension-2 curves
(z = (x, p1, p2) plus the minimally augmented test functions).

**Events.** Each test function φ_k(point) is evaluated at every accepted point; a sign change
between two points is an event. A test returns a number, or :class:`Factors`: the factors of a
characteristic product (eigenvalues, sums of eigenvalues, multipliers + 1, ...), whose sign is
that of the product of the factors above their declared rounding level (exact neutral modes are
left out and counted). The fold and LPC tests are located by a secant on the tangent component
until |t_p| <= fold_tol (resp. crossing_tol). Every other test is located by a safeguarded secant
(Illinois, with bisection when the bracket does not halve) on its sign bracket, until the bracket
is narrower than ``crossing_tol`` times the step: the size of the smallest factor depends on the
spectators and on the time unit, the sign change does not. Each iterate is one corrected point,
on the hyperplane of the step. The event reports the bracket and, for factors, the value at both
ends of the factor that changed sign, ``converged`` and, if False, the reason.

**Crossing count.** A curve may store ``data["crossing_count"]`` on each point (the number of
eigenvalues across the stability boundary) and give the driver the multiplicity of each event
kind. When the count changes by more than the located events explain (two crossings in one step,
which leave the sign of a product unchanged), the step is bisected until each part is explained;
a change left unexplained in a bracket narrower than ``crossing_tol`` times the step is reported
as an ``"unresolved crossing"`` event (``converged=False``), never dropped. A ``classify`` callback may relabel
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
from ..errors import ConvergenceError, OptionError
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
    factors: dict = dc_field(default_factory=dict)  # test name -> Factors (factor tests only)

    def explain(self) -> str:
        """The point in one sentence: arclength, residual, test functions."""
        tests = ", ".join(f"{k} = {v:.3g}" for k, v in self.tests.items()
                          if isinstance(v, (int, float)))
        return (f"curve point ({self.status}) at arclength s = {self.s:.6g}: residual "
                f"{self.residual:.1e} after {self.iterations} corrector iteration(s)"
                + (f"; test functions {tests}" if tests else "")
                + (f"; data: {', '.join(self.data)}" if self.data else "")
                + f"; unknowns of {np.size(self.z)} entries in .z")


def factor_test(factors: np.ndarray) -> float:
    """Test function built from the factors of a characteristic product: the sign of the product
    (accumulated phase, so nothing underflows) and the size |z| / (1 + |z|) of its smallest factor.

    Its value is the distance of the critical factor to zero. The product itself carries every
    other factor: adding decoupled stable variables moved a located event by 6e-4 and marked it
    converged (external review of 0.2.3). The function is flat where another factor is the
    smallest; ``locate`` then bisects its sign bracket."""
    z = np.asarray(factors, dtype=complex)
    if z.size == 0:
        return 1.0
    size = np.abs(z)
    if not np.all(size > 0):
        return 0.0
    return float(np.sign(np.cos(float(np.sum(np.angle(z))))) * np.min(size / (1.0 + size)))


@dataclass
class Factors:
    """Factors of a characteristic product, the value of a factor test function.

    ``values``: the factors (complex; the product is real). ``zero``: factors with |f| <= zero are
    at rounding level (an exactly neutral mode, e.g. a decoupled y' = 0): they are left out of the
    sign and counted in :attr:`zeros`, so a persistent neutral mode neither hides nor creates a
    sign change."""

    values: np.ndarray
    zero: float = 0.0

    def _split(self):
        z = np.asarray(self.values, dtype=complex).ravel()
        small = np.abs(z) <= self.zero
        return z[~small], int(small.sum())

    @property
    def zeros(self) -> int:
        """Number of factors at rounding level (left out of the sign)."""
        return self._split()[1]

    def signed(self) -> float:
        """Sign of the product of the factors above rounding level, times the smallest of them
        (the smallest positive float when every factor is at rounding level). Never 0."""
        z, _ = self._split()
        if z.size == 0:
            return np.finfo(float).tiny
        sg = float(np.sign(np.cos(float(np.sum(np.angle(z))))))
        return (sg or 1.0) * max(float(np.abs(z).min()), np.finfo(float).tiny)

    def real(self, everything: bool = False) -> np.ndarray:
        """The real factors above rounding level (``everything``: all of them), sorted
        (|Im f| <= 1e-8 |f|)."""
        z = np.asarray(self.values, dtype=complex).ravel() if everything else self._split()[0]
        r = z[np.abs(z.imag) <= 1e-8 * np.abs(z)].real
        return np.sort(r)


#: Number of factors each event kind moves across the stability boundary (crossing-count check).
#: An upper bound (a zero-Hopf point moves 2 on a fold curve, 1 on a Hopf curve).
MULTIPLICITY = {"fold": 1, "branch point": 1, "hopf": 2, "pd": 1, "ns": 2, "lpc": 1, "bpc": 1,
                "bt": 1, "zh": 2, "hh": 2}
#: Tests located on their value (unit tangent component), not on a sign bracket.
VALUE_TESTS = ("fold", "lpc")


def _sign_change(fa: float, fb: float) -> bool:
    """A sign change from fa to fb (a step ending exactly on a root belongs to that interval)."""
    return bool(np.isfinite(fa) and np.isfinite(fb) and fa != 0 and np.sign(fa) != np.sign(fb))


def _crossing_pair(fa: Factors | None, fb: Factors | None) -> tuple[float, float] | None:
    """The factor that changed sign between two points: (value at a, value at b), or None.

    The real factors above rounding level are matched by an assignment of least total cost
    Σ |f_a − f_b|^(1/2): a concave cost keeps a factor that did not move (a spectator) with
    itself rather than pairing the values in sorted order, which put a slow spectator between the
    two ends of the crossing factor. Of the matched pairs that change sign, the one closest to 0."""
    if fa is None or fb is None:
        return None
    ra, rb = fa.real(everything=True), fb.real(everything=True)
    from scipy.optimize import linear_sum_assignment
    if ra.size != rb.size:
        # a complex factor with |Im f| near 1e-8 |f| (a slow mode plus a stiff one) counted as
        # real at one end only (0.3.1 review): match all the factors as complex numbers, and
        # take as real those with |Im f| at rounding level of the largest factor
        za = np.asarray(fa.values, dtype=complex).ravel()
        zb = np.asarray(fb.values, dtype=complex).ravel()
        if za.size != zb.size or za.size == 0 or za.size > 400:
            return None
        i, j = linear_sum_assignment(np.sqrt(np.abs(za[:, None] - zb[None, :])))
        tiny = 1e-8 * max(float(np.abs(za).max()), float(np.abs(zb).max()))
        keep = (np.abs(za[i].imag) <= tiny) & (np.abs(zb[j].imag) <= tiny)
        ra, rb = za[i][keep].real, zb[j][keep].real
        i = j = np.arange(ra.size)
    elif ra.size == 0 or ra.size > 400:
        return None
    else:
        i, j = linear_sum_assignment(np.sqrt(np.abs(ra[:, None] - rb[None, :])))
    flip = np.flatnonzero((np.sign(ra[i]) != np.sign(rb[j]))
                          | ((np.abs(ra[i]) <= fa.zero) != (np.abs(rb[j]) <= fb.zero)))
    if flip.size == 0:
        return None
    k = flip[np.argmin(np.abs(ra[i][flip]) + np.abs(rb[j][flip]))]
    return float(ra[i][k]), float(rb[j][k])


def _match(ref: np.ndarray, other: np.ndarray) -> np.ndarray | None:
    """Indices that reorder ``other`` onto ``ref`` (least total |ref_i − other_j|), or None."""
    if ref.size != other.size:
        return None
    from scipy.optimize import linear_sum_assignment
    _, j = linear_sum_assignment(np.abs(np.asarray(ref)[:, None] - np.asarray(other)[None, :]))
    return j


def flipped(ev: CurveEvent, values: np.ndarray, key: str, factors: Callable) -> np.ndarray | None:
    """Which factors changed sign across the final bracket of ``ev``.

    ``values``: the eigenvalues (or multipliers) at the located point, ``key``: where the bracket
    ends store them in ``point.data``, ``factors(values)``: the factors built from them (any shape).
    The ends are matched to ``values`` by least distance (the bracket is narrow). Returns a boolean
    array of the shape of ``factors(values)``, or None when the bracket ends are not stored or
    differ in size. Used to classify an event by the factor that crossed, not by the factor
    closest to 0 at the located point, which a slow spectator can be (hunt 5A, K2)."""
    ends = ev.data.get("bracket_ends")
    if ends is None:
        return None
    out = []
    for q in ends:
        v = np.asarray(q.data.get(key, ()))
        j = _match(np.asarray(values), v)
        if j is None:
            return None
        out.append(np.real(factors(v[j])))
    return (out[0] < 0) != (out[1] < 0)


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
        if "bracket" in self.data:
            val = f", bracket {self.data['bracket']:.1e} of the step"
            if "crossing_factor" in self.data:
                f0, f1 = self.data["crossing_factor"]
                val += f", factor {f0:+.2e} -> {f1:+.2e} across it"
            if "rounding_band" in self.data:
                val += (f" (located on the factor itself: it is within the rounding level "
                        f"{self.data['rounding_band']:.1e} near the crossing)")
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
                 weights: np.ndarray | None = None, scale: Callable | None = None,
                 scale_text: str = "scale(z)", count_check: bool = False) -> None:
        self.H, self.Hz, self.cfg = H, Hz, settings
        self.tests = dict(tests or {})
        self.data_fn, self.classify, self.on_accept = data, classify, on_accept
        self.w = weights
        self.scale_fn, self.scale_text = scale, scale_text
        # compare data["crossing_count"] with the located events (MULTIPLICITY): only when every
        # family that moves the count is an active test
        self.count_check = count_check
        # least-squares (minimum-norm) bordered solves: set by run() when the bordered matrix is
        # exactly singular at the start (an exactly neutral decoupled mode, y' = 0)
        self.min_norm = False
        self.n_H = 0

    def explain(self) -> str:
        """Corrector, norm, residual scale and test functions."""
        tests = ", ".join(sorted(self.tests)) or "none"
        return (f"ArclengthDriver: Newton corrector on H(z) = 0 bordered by t·W(z - z_pred) = 0 "
                f"(z = unknowns and parameter)\n"
                f"  norm: {'unit weights' if self.w is None else 'declared diagonal weights'}; "
                f"residual test max|H| / {'1' if self.scale_fn is None else self.scale_text} <= "
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
        if self.min_norm:
            # an exactly neutral direction: minimum-norm solves keep it fixed (declared in notes)
            return K, lambda r: np.linalg.lstsq(K, r, rcond=None)[0]
        lu = sla.lu_factor(K)
        return K, lambda r: sla.lu_solve(lu, r)

    def _exactly_singular(self, z: np.ndarray, t: np.ndarray) -> bool:
        """The dense bordered matrix at z has an exactly zero pivot (LAPACK's LinAlgWarning)."""
        A = self.Hz(z)
        if sp.issparse(A):
            return False
        K = np.vstack([np.asarray(A), (self._w(z.size) * t)[None, :]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", sla.LinAlgWarning)
            sla.lu_factor(K)
        return any(issubclass(w.category, sla.LinAlgWarning) for w in caught)

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
        if self.scale_fn is None or not h.size:
            return float(np.max(np.abs(h))) if h.size else 0.0
        sc = np.asarray(self.scale_fn(z), float)
        if sc.ndim == 0:
            return float(np.max(np.abs(h))) / max(float(sc), 1e-300)
        # one scale per equation (0.3.1 review): a stiff decoupled row no longer loosens the others
        return float(np.max(np.abs(h) / np.maximum(sc, 1e-300)))

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
        for k, f in self.tests.items():
            r = f(pt)
            if isinstance(r, Factors):
                pt.factors[k] = r
                pt.tests[k] = r.signed()
            else:
                pt.tests[k] = float(r)
        return pt

    def _at(self, a: CurvePoint, s: float, b: CurvePoint) -> CurvePoint:
        """The corrected point at arclength a.s + s on the hyperplane of a's tangent. Exactly at a
        branch point the bordered matrix is singular: LAPACK then warns (LinAlgWarning, an exactly
        zero pivot) or returns non-finite values, both caught here; the tangent is then the chord
        of the bracket (a, b), declared in the point status."""
        z, rel, _ = self.correct(a.z + s * a.tangent, a.tangent)
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
            tz = (b.z - a.z) / self.norm(b.z - a.z)
            status = "event; tangent = chord of the bracket (singular bordered matrix)"
        return self._point(a.s + s, z, tz, rel, 0, status)

    # --- driver ------------------------------------------------------------------------------
    def run(self, z0: np.ndarray, t0: np.ndarray, *, kind: str, names: Mapping[str, int],
            bounds: Mapping[str, tuple[float, float]] | None = None,
            stop: Callable[[CurvePoint], str | None] | None = None) -> Curve:
        """Continue from a converged z0 along the initial direction t0 (re-oriented tangent).

        The start point is accepted when max|H| / scale <= 10 x the corrector tolerance (it is
        recorded with its own residual; a Newton-converged start passes)."""
        cfg = self.cfg
        for k, b in (bounds or {}).items():
            # a reversed or non-finite interval stopped the curve at its start point without a
            # word (r031r: the input fuzz case passed on an earlier refusal)
            try:
                lo, hi = (float(v) for v in b)
            except (TypeError, ValueError):
                raise OptionError(f"bounds[{k!r}] = {b!r}: a pair (low, high) of numbers") from None
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise OptionError(f"bounds[{k!r}] = {b!r}: low < high, both finite",
                                  why="the curve stops when the parameter leaves [low, high]",
                                  fix=f"bounds=({min(lo, hi)}, {max(lo, hi)})")
        z0 = np.asarray(z0, float)
        r0 = self._rel(np.asarray(self.H(z0)), z0)
        if r0 > 10 * cfg.tolerance:
            raise ConvergenceError(f"starting point is not converged: |H|/scale = {r0:.2e}",
                                   fix="converge the starting point first (Newton), then continue")
        t0n = np.asarray(t0, float) / self.norm(np.asarray(t0, float))
        notes = []
        if self._exactly_singular(z0, t0n):
            self.min_norm = True
            notes.append("the bordered Jacobian [H_z; t] is exactly singular at the start (a "
                         "direction on which H does not depend, e.g. an exactly neutral decoupled "
                         "mode y' = 0): Newton steps and tangents are minimum-norm least-squares "
                         "solutions, which keep that direction fixed")
        t = self.tangent(z0, t0n)
        cv = Curve(kind=kind, names=dict(names), settings=cfg, notes=notes)
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
        step = b.s - a.s
        budget = [self.cfg.event_max_iterations]

        def count(p):
            c = p.data.get("crossing_count") if p.data else None
            return None if c is None else int(c)

        def resolve(p, q):
            events, rejected = self._events_in(p, q, (i, i + 1), step)
            cp, cq = count(p), count(q)
            if not self.count_check or cp is None or cq is None:
                return events, rejected
            need = abs(cq - cp)
            have = sum(MULTIPLICITY.get(e.kind, 0) for e in events)
            if need <= have:
                return events, rejected
            if (q.s - p.s) > self.cfg.crossing_tol * step and budget[0] > 0:
                budget[0] -= 1
                try:
                    m = self._at(p, 0.5 * (q.s - p.s), q)
                except (ConvergenceError, FloatingPointError, np.linalg.LinAlgError,
                        RuntimeError) as e:
                    why = f"corrector failed at the midpoint ({str(e).splitlines()[0]})"
                else:
                    e1, r1 = resolve(p, m)
                    e2, r2 = resolve(m, q)
                    return e1 + e2, r1 + r2
            else:
                why = ("bracket narrower than crossing_tol x step" if budget[0] > 0 else
                       f"bisection budget event_max_iterations = {self.cfg.event_max_iterations} "
                       "exhausted")
            un = CurveEvent("unresolved crossing", (i, i + 1), q, None, False,
                            f"the crossing count changes from {cp} to {cq} between arclength "
                            f"{p.s:.12g} and {q.s:.12g} (relative width {(q.s - p.s) / step:.1e}) "
                            f"and the located events account for {have}: two crossings at one "
                            f"point (a double eigenvalue), or crossings whose signs cancel ({why})")
            un.data.update(count=(cp, cq), bracket=(q.s - p.s) / step)
            return events + [un], rejected

        events, rejected = resolve(a, b)
        cv.events += events
        cv.rejected += rejected

    def _events_in(self, a: CurvePoint, b: CurvePoint, between, step: float):
        """Locate and classify the sign changes of every test between a and b."""
        events, rejected = [], []
        tp_change = bool(a.tangent[-1] * b.tangent[-1] < 0)
        for name in self.tests:
            if not _sign_change(a.tests[name], b.tests[name]):
                continue
            fa, fb = a.factors.get(name), b.factors.get(name)
            if fa is not None and fb is not None and np.size(fa.values) != np.size(fb.values):
                # the set of factors changed (a multiplier became resolved): the sign change says
                # nothing about a crossing (hunt 5A, K5); a real crossing still moves the count
                continue
            ev = self.locate(name, a, b, between, step=step)
            if tp_change and name not in VALUE_TESTS:
                ev.data["tp_sign_change"] = True
            if self.classify is not None:
                kind, why = self.classify(name, ev)
                if kind is None:
                    rejected.append((ev, why))
                    continue
                ev.kind = kind
            events.append(ev)
        return events, rejected

    def locate(self, name: str, a: CurvePoint, b: CurvePoint, between,
               step: float | None = None) -> CurveEvent:
        """Locate the sign change of φ_name between a and b (each iterate = one corrected point).

        Fold and LPC: secant on the value until |φ| <= fold_tol (LPC: crossing_tol). Other tests:
        Illinois secant on the sign bracket, bisection when the bracket did not halve in two
        iterations, until the bracket is narrower than crossing_tol x ``step`` (default b.s - a.s)."""
        cfg = self.cfg
        f = self.tests[name]
        width0 = b.s - a.s
        step = width0 if step is None else step
        value_mode = name in VALUE_TESTS
        tol = cfg.fold_tol if name == "fold" else cfg.crossing_tol
        lo, hi = [0.0, a.tests[name], a], [width0, b.tests[name], b]   # [s, f, point]
        best, why = b, f"budget event_max_iterations = {cfg.event_max_iterations} exhausted"
        conv = abs(hi[1]) <= tol if value_mode else False
        last_side, widths = 0, [width0]
        tol_w = cfg.crossing_tol * step
        wlo = whi = 1.0                       # Illinois weights of the two ends
        for k in range(cfg.event_max_iterations):
            w = hi[0] - lo[0]
            if conv or (not value_mode and w <= tol_w):
                conv = True
                break
            # secant on the factor that changed sign when it can be identified, else on φ
            pair = None if value_mode else _crossing_pair(lo[2].factors.get(name),
                                                          hi[2].factors.get(name))
            gl, gh = pair if pair is not None else (lo[1], hi[1])
            gl, gh = wlo * gl, whi * gh
            s2 = hi[0] - gh * w / (gh - gl) if gh != gl else 0.5 * (lo[0] + hi[0])
            if not lo[0] <= s2 <= hi[0] or (len(widths) >= 3 and widths[-1] > 0.5 * widths[-3]):
                s2 = 0.5 * (lo[0] + hi[0])    # outside the bracket, or no halving in 2 steps
            if not value_mode:
                # at least half the target width from each end (Dekker): a secant converging from
                # one side then lands on the other side and closes the bracket
                s2 = min(max(s2, lo[0] + 0.5 * tol_w), hi[0] - 0.5 * tol_w)
            try:
                pt = self._at(a, s2, b)
            except (ConvergenceError, FloatingPointError, np.linalg.LinAlgError, RuntimeError) as e:
                why = f"corrector failed at iterate {k} ({str(e).splitlines()[0]})"
                break
            f2 = pt.tests[name] if name in pt.tests else float(f(pt))
            best = pt
            if f2 == 0 or (value_mode and abs(f2) <= tol):
                conv = True
            if np.sign(f2) == np.sign(lo[1]):
                lo, wlo = [s2, f2, pt], 1.0
                whi = whi / 2 if last_side == -1 else whi
                last_side = -1
            else:
                hi, whi = [s2, f2, pt], 1.0
                wlo = wlo / 2 if last_side == 1 else wlo
                last_side = 1
            widths.append(hi[0] - lo[0])
        else:
            conv = conv or (not value_mode and hi[0] - lo[0] <= cfg.crossing_tol * step)
        band = None
        if not value_mode and conv:
            # 0.3.1 review: a crossing factor entering the rounding band (|f| <= 10 n eps max|λ|)
            # leaves the product and flips φ before the factor reaches 0; with a stiff spectator
            # (rate 1e8) a branch point at p = 0 was placed at p = -6.7e-7 with a bracket of 1e-11
            # of the step. The zero of the factor itself is located by tracking it.
            tr = self._track_band(name, a, b, lo, hi, width0, tol_w)
            if tr is not None:
                lo, hi, band = tr
        if not value_mode and conv:
            best = min((lo[2], hi[2]), key=lambda q: abs(q.tests[name]))
            # the end where the factor that changed sign is smaller (|φ| is the smallest factor
            # above rounding level, a spectator once the crossing factor is at rounding level)
            pair = _crossing_pair(lo[2].factors.get(name), hi[2].factors.get(name))
            if pair is not None:
                best = lo[2] if abs(pair[0]) < abs(pair[1]) else hi[2]
            # one corrected point at the secant root inside the final bracket: the bracket ends
            # are within crossing_tol x step, the secant root closer still on a smooth factor
            if pair is not None and pair[0] != pair[1] and hi[0] > lo[0] \
                    and min(abs(pair[0]), abs(pair[1])) > 0:
                s2 = hi[0] - pair[1] * (hi[0] - lo[0]) / (pair[1] - pair[0])
                if lo[0] < s2 < hi[0]:
                    try:
                        pt = self._at(a, s2, b)
                    except (ConvergenceError, FloatingPointError, np.linalg.LinAlgError,
                            RuntimeError):
                        pt = None
                    if pt is not None:
                        q1 = _crossing_pair(lo[2].factors.get(name), pt.factors.get(name))
                        q2 = _crossing_pair(pt.factors.get(name), hi[2].factors.get(name))
                        f3 = q1[1] if q1 is not None else (q2[0] if q2 is not None else None)
                    if pt is not None and f3 is not None and abs(f3) < min(map(abs, pair)):
                        if np.sign(pt.tests[name]) == np.sign(lo[1]):
                            lo = [s2, pt.tests[name], pt]
                        else:
                            hi = [s2, pt.tests[name], pt]
                        best = pt
        ev = CurveEvent(name, between, best, best.tests.get(name), bool(conv), "" if conv else why)
        if value_mode:
            ev.data["criterion"] = (f"|test {name}| <= {'fold_tol' if name == 'fold' else 'crossing_tol'}"
                                    f" = {tol:g}, secant budget {cfg.event_max_iterations}")
        else:
            ev.data["bracket"] = (hi[0] - lo[0]) / step
            ev.data["bracket_ends"] = (lo[2], hi[2])
            ev.data["criterion"] = (f"sign change of {name} bracketed to arclength width <= "
                                    f"crossing_tol x step = {cfg.crossing_tol:g} x {step:.3g}, "
                                    f"budget {cfg.event_max_iterations}")
            fl, fh = lo[2].factors.get(name), hi[2].factors.get(name)
            pair = _crossing_pair(fl, fh)
            if pair is not None:
                ev.data["crossing_factor"] = pair
            if fl is not None and fh is not None:
                if fl.zeros or fh.zeros:
                    ev.data["neutral_factors"] = max(fl.zeros, fh.zeros)
            if band is not None:
                ev.data["rounding_band"] = band
        return ev

    def _track_band(self, name: str, a: CurvePoint, b: CurvePoint, lo: list, hi: list,
                    width0: float, tol_w: float):
        """When the final bracket of φ is a rounding-band entry of the crossing factor (its value
        has the same sign at both ends), bracket the zero of that factor itself: Illinois secant
        on the factor, followed from point to point by its predicted value, between the step
        ends. Returns (lo, hi, band) or None (no band entry, a neutral factor present, or the
        factor cannot be followed)."""
        fl, fh = lo[2].factors.get(name), hi[2].factors.get(name)
        pair = _crossing_pair(fl, fh)
        if pair is None or pair[0] == 0 or np.sign(pair[0]) != np.sign(pair[1]):
            return None
        fa, fb = a.factors.get(name), b.factors.get(name)
        p0 = _crossing_pair(fa, fb)
        if p0 is None or fa.zeros or fb.zeros or np.sign(p0[0]) == np.sign(p0[1]) or 0 in p0:
            return None
        L, H = [0.0, p0[0], a], [width0, p0[1], b]
        # the band-entry bracket is on one side of the zero: start from its end nearer the zero
        for end in ((lo[0], pair[0], lo[2]), (hi[0], pair[1], hi[2])):
            if np.sign(end[1]) == np.sign(L[1]) and end[0] > L[0]:
                L = list(end)
            elif np.sign(end[1]) == np.sign(H[1]) and end[0] < H[0]:
                H = list(end)
        wl = wh = 1.0
        last, widths = 0, [H[0] - L[0]]
        for _ in range(self.cfg.event_max_iterations):
            w = H[0] - L[0]
            if w <= tol_w:
                break
            gl, gh = wl * L[1], wh * H[1]
            s2 = H[0] - gh * w / (gh - gl) if gh != gl else 0.5 * (L[0] + H[0])
            if not L[0] <= s2 <= H[0] or (len(widths) >= 3 and widths[-1] > 0.5 * widths[-3]):
                s2 = 0.5 * (L[0] + H[0])
            s2 = min(max(s2, L[0] + 0.5 * tol_w), H[0] - 0.5 * tol_w)
            try:
                pt = self._at(a, s2, b)
            except (ConvergenceError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
                return None
            fp = pt.factors.get(name)
            if fp is None:
                return None
            vals = np.asarray(fp.values, dtype=complex).ravel()
            if vals.size == 0:
                return None
            pred = L[1] + (H[1] - L[1]) * (s2 - L[0]) / w
            g2 = float(vals[np.argmin(np.abs(vals - pred))].real)
            if g2 == 0:
                L = H = [s2, g2, pt]
                break
            if np.sign(g2) == np.sign(L[1]):
                L, wl = [s2, g2, pt], 1.0
                wh = wh / 2 if last == -1 else wh
                last = -1
            else:
                H, wh = [s2, g2, pt], 1.0
                wl = wl / 2 if last == 1 else wl
                last = 1
            widths.append(H[0] - L[0])
        if H[0] - L[0] > tol_w and L is not H:
            return None
        return ([L[0], L[2].tests[name], L[2]], [H[0], H[2].tests[name], H[2]],
                float(max(fl.zero, fh.zero)))
